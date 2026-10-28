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


def test_poc_2_nowcast_uses_own_validation(monkeypatch, tmp_path):
    labels = {task: pd.DataFrame({'fog': [0, 1]}) for task in ('nowcast', 'lead_time')}
    for task, frame in labels.items():
        frame.attrs['task'] = task
    monkeypatch.setattr(study, 'load_sources', lambda *a, **kw: (None, None))
    monkeypatch.setattr(study, 'label_tables', lambda *a: tuple(labels.values()))
    monkeypatch.setattr(study, 'build_windows', lambda w, frame, *a, **kw: (None, None, frame, None))
    monkeypatch.setattr(study, 'rolling_origin_folds', lambda frame: [{key: frame.attrs['task'] for key in ('train', 'val', 'test')}])
    monkeypatch.setattr(study, 'prepare', lambda w, parts, **kw: SimpleNamespace(cv=parts['val']))
    tuned = []

    def tune(prep, site, task, **kw):
        tuned.append((task, prep.cv))
        return {'selected_on': prep.cv}

    monkeypatch.setattr(study, 'tune_cnn', tune)
    received = []

    class StopProbe(Exception):
        pass

    def evaluate(weather, frame, site, task, family, *, cnn_cfg, **kw):
        if family != 'climatology':
            received.append((task, cnn_cfg['selected_on']))
            if task == 'lead_time':
                raise StopProbe
        return {}, [], {}

    monkeypatch.setattr(study, 'evaluate_family', evaluate)
    monkeypatch.setattr(study, 'evaluate_ablation', lambda *a, **kw: {'fold_audit': []})
    with pytest.raises(StopProbe):
        study.run_family('temporal_cnn', sites=('location_10',), out_dir=tmp_path)
    assert tuned == [('nowcast', 'nowcast'), ('lead_time', 'lead_time')]
    assert received == [('nowcast', 'nowcast'), ('lead_time', 'lead_time')]


def test_cnn_tuning_cache_is_shared_per_site_and_task(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(study, 'tune_cnn', lambda prep, site, task, **kw: calls.append((site, task, prep.cv)) or {'selected_on': prep.cv})
    cache = {}
    first = SimpleNamespace(cv='nowcast')
    assert study.tuned_cnn_cfg(first, 'location_10', 'nowcast', art_dir=tmp_path, cache=cache) == {'selected_on': 'nowcast'}
    assert study.tuned_cnn_cfg(SimpleNamespace(cv='ignored'), 'location_10', 'nowcast', art_dir=tmp_path, cache=cache) == {'selected_on': 'nowcast'}
    study.tuned_cnn_cfg(SimpleNamespace(cv='lead_time'), 'location_10', 'lead_time', art_dir=tmp_path, cache=cache)
    assert calls == [('location_10', 'nowcast', 'nowcast'), ('location_10', 'lead_time', 'lead_time')]


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
    monkeypatch.setattr(study, 'evaluate_family', lambda *a, **kw: (main_result, [{'fold': 1}], {}))
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
