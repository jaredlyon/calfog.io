"""Task-local tuning, minimum-category labels and fail-closed ablation invariants."""
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pandas as pd
import pytest
from models._shared import data, study
from models._shared.labels import label_tables
from models._shared.evaluation import fit_calibrator, finalize_model, ordinal_bootstrap
from models._shared.visibility import N_CATEGORIES, hourly_labels


@pytest.fixture(autouse=True)
def synthetic_environment(monkeypatch):
    monkeypatch.setattr(study, 'capture_environment', lambda: {'python': 'synthetic producer'})


def tuning_data():
    times = pd.date_range('2020-01-01', periods=24 * 181, freq='h')
    weather = pd.DataFrame({'time': times, 'custom': np.arange(len(times), dtype=float),
        **{col: 0. for col in data.CAL_COLS + data.AQI_COLS + data.MEASURED_AQ_COLS}})
    issues = times[24::24]
    categories = np.where(np.arange(len(issues)) % 2, 4, 17)
    labels = pd.DataFrame({'issue_time': issues, 'row_key': issues.astype(str),
                           'category': categories, 'fog': categories <= 4})
    return weather, labels


def test_poc_2_nowcast_uses_own_validation(monkeypatch, tmp_path):
    weather, now = tuning_data()
    lead = now.assign(category=np.where(now.category.eq(4), 17, 4), fog=~now.fog)
    monkeypatch.setattr(study, 'load_sources', lambda *a, **kw: (weather, None))
    monkeypatch.setattr(study, 'label_tables', lambda *a: (now, lead))
    tuned = {}
    def tune(prep, site, task, **kwargs):
        tuned[task] = prep.cv.copy()
        return {'selected_on': task}
    monkeypatch.setattr(study, 'tune_cnn', tune)
    received = []
    def evaluate(weather, frame, site, task, family, *, cnn_cfg, **kwargs):
        if family != 'climatology':
            received.append((task, cnn_cfg['selected_on']))
        return {}, [], {}
    monkeypatch.setattr(study, 'evaluate_family', evaluate)
    monkeypatch.setattr(study, 'evaluate_ablation', lambda *a, **kw: {'fold_audit': []})
    metrics = study.run_family('temporal_cnn', sites=('location_10',), weather_cols=['custom'],
                               out_dir=tmp_path, write=False)
    assert received == [('nowcast', 'nowcast'), ('lead_time', 'lead_time')]
    assert not np.array_equal(tuned['nowcast'], tuned['lead_time'])
    assert set(metrics['per_site']['location_10']['cnn_tuning']) == {'nowcast', 'lead_time'}


def test_cnn_tuning_prefix_clips_against_earlier_matched_tests():
    weather, labels = tuning_data()
    cohort = labels.iloc[:160]
    original = study.rolling_origin_folds(labels)
    prep, context, tests = study.tuning_prefix(weather, labels, [cohort], weather_cols=['custom'])
    expected = study.rolling_origin_folds(cohort)[0]['test_start'] - pd.Timedelta(hours=24)
    assert pd.Timestamp(context['cutoff']) == expected
    for split in ('train', 'val'):
        assert (prep.metas[split].issue_time < expected).all()
        assert set(prep.metas[split].row_key) <= set(original[0][split].row_key)
        assert all(not set(prep.metas[split].row_key) & set(test.row_key) for test in tests)
    assert len(prep.cv) < len(original[0]['val'])
    pd.testing.assert_frame_equal(study.rolling_origin_folds(labels)[0]['test'], original[0]['test'])


def test_cnn_tuning_cache_is_shared_only_for_identical_context(monkeypatch, tmp_path):
    weather, labels = tuning_data()
    prep, context, tests = study.tuning_prefix(weather, labels, [labels], weather_cols=['custom'])
    first = study.rolling_origin_folds(labels)[0]
    assert prep.metas['val'].row_key.tolist() == first['val'].row_key.tolist()
    calls = []
    monkeypatch.setattr(study, 'tune_cnn', lambda *a, **kw: calls.append(a[2]) or {'call': len(calls)})
    cache = {}
    def select(prepared, ctx, recipient):
        return study.tuned_cnn_cfg(prepared, 'location_10', 'nowcast', art_dir=tmp_path,
                                    cache=cache, context=ctx, recipient_tests=recipient)
    assert select(prep, context, tests) == select(prep, context, tests) == {'call': 1}
    changed = labels.copy()
    changed.loc[0, 'category'] = 4 if changed.category.iloc[0] == 17 else 17
    updated, new_context, new_tests = study.tuning_prefix(weather, changed, [changed], weather_cols=['custom'])
    assert new_context['train_sha256'] != context['train_sha256']
    assert select(updated, new_context, new_tests) == {'call': 2}
    assert cache['location_10', 'nowcast'] == {'cfg': {'call': 2}, 'context': new_context}
    earlier, earlier_context, earlier_tests = study.tuning_prefix(
        weather, labels, [labels.iloc[:160]], weather_cols=['custom'])
    assert earlier_context['cutoff'] != context['cutoff']
    assert select(earlier, earlier_context, earlier_tests) == {'call': 3}
    with pytest.raises(AssertionError):
        select(prep, context, [prep.metas['train']])


def test_cnn_tuning_unavailable_prefix_does_not_move_folds():
    weather, labels = tuning_data()
    before = study.rolling_origin_folds(labels)[0]['test']
    with pytest.raises(study.NotEstimable, match='cnn tuning prefix lacks both classes'):
        study.tuning_prefix(weather, labels, [labels.iloc[:120]], weather_cols=['custom'])
    pd.testing.assert_frame_equal(before, study.rolling_origin_folds(labels)[0]['test'])


def test_poc_3_hourly_loader_keeps_min_category_and_drops_unknown(monkeypatch, tmp_path):
    times = pd.date_range('2024-01-02 08:00', periods=11, freq='h', tz='UTC')
    obs = pd.DataFrame({'time_utc': list(times) + [times[0]+pd.Timedelta(minutes=10)],
                        'cat_lo': [17]*10+[0, 2], 'cat_hi': [17]*10+[17, 2]})
    hourly = hourly_labels(obs)
    weather = pd.DataFrame({'time': pd.date_range('2024-01-01', periods=72, freq='h')})
    folder = tmp_path / 'datasets' / 'weather'
    folder.mkdir(parents=True)
    (folder / 'location_10_weather_appended.csv').touch()
    monkeypatch.setattr(data, 'ROOT', tmp_path)
    monkeypatch.setattr(pd, 'read_csv', lambda path, **kw: hourly.copy() if 'visibility' in str(path) else weather.copy())
    w, vis = data.load_sources('location_10')
    assert len(vis) == 10
    assert vis.iloc[0].category == 2
    assert vis.iloc[0].fog
    now, lead = label_tables(w, vis, 'location_10')
    assert int(now.fog.sum()) == 1
    assert lead.category.tolist() == [2]
    assert lead.fog.tolist() == [True]


@pytest.mark.parametrize('yv', [np.zeros(4, dtype=int), np.ones(4, dtype=int), np.array([], dtype=int), np.array([0, 1])])
def test_full_4_validation_checked_before_any_training(monkeypatch, yv):
    class Reached(Exception):
        pass

    trainer = Mock(side_effect=Reached)

    meta = pd.DataFrame({'row_key': ['synthetic']})
    prep = SimpleNamespace(yv=yv, metas={key: meta for key in ('train', 'val', 'test')})
    monkeypatch.setattr(study, 'rolling_origin_folds', lambda frame: [{key: None for key in ('train', 'val', 'test')}])
    monkeypatch.setattr(study, 'prepare', lambda *a, **kw: prep)
    monkeypatch.setattr(study.FAMILIES['xgboost'], 'fit_predict', trainer)
    if len(np.unique(yv)) == 2:
        with pytest.raises(Reached):
            study.evaluate_ablation(None, None, 'location_10', 'nowcast', 'xgboost', cnn_cfg=None, weather_cols=[], seed=17)
        trainer.assert_called_once()
    else:
        with pytest.raises(study.NotEstimable, match=f'fold 1, arm without:.*validation positives={int(yv.sum())}/{len(yv)}'):
            study.evaluate_ablation(None, None, 'location_10', 'nowcast', 'xgboost', cnn_cfg=None, weather_cols=[], seed=17)
        trainer.assert_not_called()


def test_main_single_class_validation_uses_no_calibrator():
    assert fit_calibrator(np.zeros(4, dtype=int), np.arange(4) / 4) is None
    assert fit_calibrator(np.ones(4, dtype=int), np.arange(4) / 4) is None


@pytest.mark.parametrize('yv', [np.zeros(4, dtype=int), np.ones(4, dtype=int), np.array([], dtype=int)])
@pytest.mark.parametrize('bad_arm', ['without', 'with'])
def test_full_4_nonestimable_comparison_keeps_main_result(monkeypatch, tmp_path, yv, bad_arm):
    from models._shared import report

    frame = pd.DataFrame({'fog': [0, 1]})
    monkeypatch.setattr(study, 'load_sources', lambda *a, **kw: (pd.DataFrame(), None))
    monkeypatch.setattr(study, 'label_tables', lambda *a: (frame, frame.copy()))
    monkeypatch.setattr(study, 'build_windows', lambda *a, **kw: (None, None, frame, None))
    monkeypatch.setattr(study, 'rolling_origin_folds', lambda *a: [{key: None for key in ('train', 'val', 'test')}])
    meta = pd.DataFrame({'row_key': ['synthetic']})

    def prepare(*args, **kwargs):
        arm = 'with' if kwargs.get('aq_cols') else 'without'
        return SimpleNamespace(yv=yv if arm == bad_arm else np.array([0, 1]),
                               metas={key: meta for key in ('train', 'val', 'test')})

    monkeypatch.setattr(study, 'prepare', prepare)
    trainer = Mock(side_effect=AssertionError('ablation training started'))
    monkeypatch.setattr(study.FAMILIES['xgboost'], 'fit_predict', trainer)
    y = np.array([0, 1, 0, 1])
    p = np.array([.1, .8, .3, .9])
    test_meta = pd.DataFrame({'issue_time': pd.date_range('2024-01-01', periods=4, freq='D'), 'fold': 1})
    main_result = finalize_model(y, p, p, test_meta, [{'fold': 1, 'y': y, 'xgboost': p}],
                                 'xgboost', 17, n_boot=4)
    categories = np.where(y, 2, 17)
    pmf = np.zeros((len(y), N_CATEGORIES)); pmf[:, 2] = p; pmf[:, 17] = 1-p
    main_result['ordinal'] = ordinal_bootstrap(categories, pmf, test_meta, n_boot=4, seed=17)
    def evaluate_main(*args, prediction_path=None, **kwargs):
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_path.write_bytes(b'synthetic main prediction')
        return main_result, [{'fold': 1}], {}
    monkeypatch.setattr(study, 'evaluate_family', evaluate_main)
    monkeypatch.setattr(report, 'write_plots', Mock())
    metrics = study.run_family('xgboost', sites=('location_10',), out_dir=tmp_path, n_boot=4)
    trainer.assert_not_called()
    sd = metrics['per_site']['location_10']
    for task in ('nowcast', 'lead_time'):
        assert sd['tasks'][task]['xgboost'] == main_result
        assert 'climatology' in sd['tasks'][task]
        for aq_set in ('nox', 'pm25', 'cams', 'nox_monitor'):
            ablation = sd['aqi_ablation'][aq_set][task]['xgboost']
            assert ablation['status'] == 'not_estimable'
            assert f'fold 1, arm {bad_arm}' in ablation['reason']
            assert f'validation positives={int(yv.sum())}/{len(yv)}' in ablation['reason']
            assert f'{task}_{aq_set}' not in sd['fold_audit']
            assert metrics['aggregate']['aqi_site_summary']['location_10'][aq_set][task] == {}
    assert len(metrics['aggregate']['aqi_not_estimable']) == 8
    text = (tmp_path / 'xgboost_results_v2.txt').read_text(encoding='utf-8')
    assert text.count('AQI ablation: not estimable — ') == 8
    comparison = report.write_comparison({'xgboost': metrics}, tmp_path / 'comparison')
    assert len(comparison['aggregate']['aqi_not_estimable']) == 8
    assert comparison['per_site']['location_10']['tasks']['nowcast']['xgboost'] == main_result
    monkeypatch.setattr(study, 'evaluate_ablation', Mock(side_effect=ValueError('real failure')))
    with pytest.raises(ValueError, match='real failure'):
        study.run_family('xgboost', sites=('location_10',), out_dir=tmp_path, n_boot=4)


def test_unavailable_cnn_prefix_marks_every_recipient_without_training(monkeypatch, tmp_path):
    weather, labels = tuning_data()
    monkeypatch.setattr(study, 'load_sources', lambda *a, **kw: (weather, None))
    monkeypatch.setattr(study, 'label_tables', lambda *a: (labels, labels))
    def unavailable(*args, **kwargs):
        raise study.NotEstimable('cnn tuning prefix lacks both classes')
    monkeypatch.setattr(study, 'tuning_prefix', unavailable)
    monkeypatch.setattr(study, 'tune_cnn', Mock(side_effect=AssertionError('tuning began')))
    monkeypatch.setattr(study, 'evaluate_ablation', Mock(side_effect=AssertionError('ablation began')))
    def baseline(*args, **kwargs):
        assert args[4] == 'climatology'
        return {'ordinal': {}}, [], {}
    monkeypatch.setattr(study, 'evaluate_family', baseline)
    result = study.run_family('temporal_cnn', sites=('location_10',), weather_cols=['custom'],
                               out_dir=tmp_path, write=False)
    sd = result['per_site']['location_10']
    for task in ('nowcast', 'lead_time'):
        expected = {'status': 'not_estimable', 'reason': 'cnn tuning prefix lacks both classes'}
        assert sd['tasks'][task]['temporal_cnn'] == expected
        assert sd['tasks'][task]['climatology'] == {'ordinal': {}}
        assert all(tasks[task]['temporal_cnn'] == expected for tasks in sd['aqi_ablation'].values())
