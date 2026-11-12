"""Shared registry, parameterized windows and one-family orchestration contracts."""
from copy import deepcopy
import hashlib
import json
import numpy as np
import pandas as pd
import pytest
from models._shared import study
from models._shared.data import WEATHER_COLS, CAL_COLS, AQI_COLS, MEASURED_AQ_COLS, AQ_SETS, PRIMARY_FAMILIES
from models._shared.families import FAMILIES, FamilyOutput, seed_for
from models._shared.windows import build_windows, tree_feature_names, tree_view, fit_preprocess
from models._shared.visibility import N_CATEGORIES


def test_registry_contract():
    assert set(FAMILIES) == {'climatology', 'logistic_regression', 'random_forest', 'xgboost',
                             'xgboost_focal_loss', 'temporal_cnn', 'ensemble', 'persistence', 'analog',
                             'xgboost_ordinal', 'xgboost_aft', 'ebm', 'xgboost_pooled'}
    for module in FAMILIES.values():
        assert isinstance(module.NEEDS_CNN_CFG, bool)
        assert callable(module.fit_predict)
    assert set(study.FAMILY_DIRS) == set(FAMILIES) - {'climatology'}
    assert tuple(study.FAMILY_DIRS)[:3] == PRIMARY_FAMILIES
    assert study.PREDICTOR_SETS['xgboost'] == 'weather'
    assert study.PREDICTOR_SETS['persistence'] == 'observed visibility history'
    assert study.PREDICTOR_SETS['xgboost_pooled'] == 'weather + observed visibility (own + 4 neighbors), pooled sites'
    parts = ('xgboost', 'location_10', 'nowcast', 2)
    assert seed_for(*parts) == int(hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest()[:8], 16)


def test_custom_weather_columns_do_not_rebind_default_features():
    times = pd.date_range('2024-01-01', periods=30, freq='h')
    weather = pd.DataFrame({'time': times, 'custom': np.arange(30), **{c: np.zeros(30) for c in CAL_COLS}})
    part = pd.DataFrame({'issue_time': [times[-1]], 'category': [17]})
    X, y, meta, cols = build_windows(weather, part, weather_cols=['custom'])
    assert cols == ['custom'] + CAL_COLS
    assert X.shape == (1, 24, 5)
    assert WEATHER_COLS[0] == 'temperature_2m'
    assert len(tree_feature_names(cols)) == tree_view(X).shape[1]
    np.testing.assert_array_equal(tree_view(X)[0, :len(cols)], X[0, -1])


def test_preprocessing_is_fit_only_on_training_timesteps():
    X = np.array([[[0.], [2.]], [[2.], [4.]]], dtype=np.float32)
    held = np.full((1, 2, 1), 1000., dtype=np.float32)
    (train, val), imputer, scaler = fit_preprocess(X, held)
    np.testing.assert_allclose(scaler.mean_, [2.])
    np.testing.assert_allclose(train.mean(), 0., atol=1e-6)
    assert val.min() > 100


def test_run_family_routes_one_family_and_baseline_without_tuning(monkeypatch, tmp_path):
    times = pd.date_range('2024-01-01', periods=120 * 24, freq='h')
    weather = pd.DataFrame({'time': times, 'custom': np.sin(np.arange(len(times)) / 24),
                             **{c: np.zeros(len(times)) for c in CAL_COLS + AQI_COLS + MEASURED_AQ_COLS + ['nox_mon_06-019-0001']}})
    labels = pd.DataFrame({'issue_time': times, 'fog': np.arange(len(times)) % 2,
                          'category': np.where(np.arange(len(times)) % 2, 2, 17),
                          'row_key': times.astype(str), 'site': 'location_10',
                          'dpd_c': 2., 'wind_kmh': 3.})
    monkeypatch.setattr(study, 'load_sources', lambda site, **kw: (weather, None))
    monkeypatch.setattr(study, 'label_tables', lambda *a: (labels, labels.copy()))
    calls = []

    def predict(prep, *, cnn_cfg, seed):
        calls.append((len(prep.cols), seed))
        def pmf(n):
            p = np.zeros((n, N_CATEGORIES))
            p[:, 2] = np.linspace(.1, .9, n)
            p[:, 17] = 1-p[:, 2]
            return p
        return FamilyOutput(pmf(len(prep.cv)), pmf(len(prep.ct)),
                            {name: 100 / len(prep.cols) for name in prep.cols})

    monkeypatch.setattr(FAMILIES['xgboost'], 'fit_predict', predict)
    monkeypatch.setattr(study, 'tuned_cnn_cfg', lambda *a, **kw: (_ for _ in ()).throw(AssertionError('tabular family tuned CNN')))
    from models._shared import report
    monkeypatch.setattr(report, 'write_plots', lambda *a, **kw: None)
    metrics = study.run_family('xgboost', sites=('location_10',), weather_cols=['custom'], out_dir=tmp_path, n_boot=8)
    assert metrics['n_boot'] == 8
    assert metrics['weather_cols'] == ['custom']
    assert metrics['predictor_set'] == 'weather'
    assert len(calls) == 54  # 2 tasks x (3 main folds + 4 AQ sets x 2 arms x 3 folds)
    sd = metrics['per_site']['location_10']
    for task in ('nowcast', 'lead_time'):
        assert set(sd['tasks'][task]) == {'xgboost', 'climatology'}
        assert sd['cnn_cfg'][task] is None
        assert len(sd['fold_audit'][task]) == 3
        result = sd['tasks'][task]['xgboost']
        assert result['valid_ap_resamples'] <= 8
        assert np.isclose(sum(result['importance'].values()), 100)
        assert result['ordinal']['rps_ci'][0] <= result['ordinal']['rps'] <= result['ordinal']['rps_ci'][1]
        assert set(sd['aqi_ablation']) == set(AQ_SETS) | {'nox_monitor'}
        for aq_set in sd['aqi_ablation']:
            ablation = sd['aqi_ablation'][aq_set][task]['xgboost']
            assert ablation['bootstrap_resamples'] == 8
            assert ablation['identical_issue_keys']
            assert {'with', 'without'} <= ablation.keys()
            assert ablation['delta_rps_ci'][0] <= ablation['delta_rps'] <= ablation['delta_rps_ci'][1]
        for arm in ('main', 'climatology', 'nox_with', 'nox_without', 'nox_monitor_with', 'nox_monitor_without'):
            assert (tmp_path / 'xgboost_predictions_v2' / f'location_10__{task}__{arm}.npz').is_file()
    path = tmp_path / 'xgboost_metrics_v2.json'
    assert json.loads(path.read_text())['family'] == 'xgboost'
    assert (tmp_path / 'xgboost_results_v2.txt').is_file()
    text = (tmp_path / 'xgboost_results_v2.txt').read_text(encoding='utf-8')
    assert 'RPS:' in text and 'Capped NLL (eps=1e-9, nats):' in text and 'Within one category:' in text
    assert 'category-representative meters log10 MAE' in text
    assert 'Predictor set: weather' in text
    comparison = report.write_comparison({'xgboost': metrics}, tmp_path / 'comparison')
    assert comparison['aggregate']['nowcast']['xgboost']['mean_rps'] >= 0
    markdown = (tmp_path / 'comparison' / 'model_comparison_v2.md').read_text(encoding='utf-8')
    assert '| RPS (95% CI) |' in markdown and 'Paired delta RPS' in markdown
    assert '| Predictor set |' in markdown
    assert all(label in markdown for label in ('NOx (primary)', 'PM2.5 (secondary)', 'CAMS (appendix)'))
    assert 'NOx conditional on monitor ID (sensitivity)' in markdown
    assert 'both arms contain identical weather/calendar inputs and monitor-ID indicators' in markdown
    assert 'mean(-log(clip(p_true_category, 1e-9, 1)))' in markdown
    assert comparison['per_site']['location_10']['environments']['xgboost'] == sd['environment']
    assert comparison['nox_value_descriptives']


def test_reuse_retrains_binary_only_metrics(monkeypatch, tmp_path):
    import sys
    from models import run_all_v2
    saved = tmp_path / 'xgboost_metrics_v2.json'
    saved.write_text(json.dumps({'per_site': {'location_10': {'tasks': {'nowcast': {
        'xgboost': {'ap': .5}}}}}}), encoding='utf-8')
    retrained = []
    monkeypatch.setattr(run_all_v2, 'FAMILY_DIRS', {'xgboost': tmp_path})
    monkeypatch.setattr(sys, 'argv', ['run_all_v2.py', '--reuse'])
    monkeypatch.setattr(run_all_v2, 'run_family', lambda f, **kw: retrained.append(f) or {})
    monkeypatch.setattr(run_all_v2, 'write_comparison', lambda *a: None)
    run_all_v2.main()
    assert retrained == ['xgboost']


def _report_result():
    point = {'expected_cost_per_issue': .03, 'mean_threshold': .5, 'precision': .4,
             'recall': .6, 'f1': .48, 'alert_rate': .1}
    return {'ap': .4, 'ap_ci': [.3, .5], 'roc_auc': .6, 'roc_auc_ci': [.5, .7],
            'brier': .03, 'calibration': {'brier_after': .025, 'ece_after': .01},
            'ordinal': {'rps': .1, 'rps_ci': [.08, .12], 'nll': 1.2, 'within_one': .8,
                        'median_category_mae': 1., 'median_meters_mae_log10': .2, 'median_meters_n': 10},
            'precision': .4, 'recall': .6, 'f1': .48, 'cost_threshold': .5,
            'climatology_ap': .05, 'cost_operating_points': {str(r): dict(point) for r in (1, 3, 5, 10, 20)}}


def _report_run(family='xgboost', *, sites=('location_10',)):
    return {'family': family, 'per_site': {
        site: {'cohort': {}, 'tasks': {
            task: {family: _report_result(), 'climatology': _report_result()}
            for task in ('nowcast', 'lead_time')},
            'aqi_ablation': {}, 'fold_audit': {'nowcast': [], 'lead_time': []}}
        for site in sites}}


def _nox_ablation(*, n=100, positives=5):
    without, with_ = _report_result(), _report_result()
    with_['ap'], with_['brier'], with_['calibration']['brier_after'] = .5, .02, .015
    for point in with_['cost_operating_points'].values():
        point['expected_cost_per_issue'] = .02
    return {'n_test': n, 'pos_count': positives, 'with': with_, 'without': without,
            'delta_ap': .1, 'delta_ap_ci': [.05, .15], 'delta_rps': -.01, 'delta_rps_ci': [-.02, 0.]}


def test_nox_value_rows_numeric_and_equal_airport_means():
    from models._shared.report import nox_value_rows
    run = _report_run(sites=('location_10', 'location_9'))
    for site, positives in (('location_10', 5), ('location_9', 90)):
        run['per_site'][site]['aqi_ablation'] = {
            'nox': {'nowcast': {'xgboost': _nox_ablation(positives=positives)}}}
    before = deepcopy(run)
    rows = nox_value_rows(run)
    assert run == before
    assert len(rows) == 18  # two airports plus means, five cost rows and one Brier row each
    cost = next(r for r in rows if r['site'] == 'location_10' and r.get('ratio') == 1)
    assert cost['cost_without'] == .03 and cost['cost_with'] == .02
    assert cost['delta_cost'] == pytest.approx(-.01)
    assert cost['delta_value'] == pytest.approx(.2)
    high = next(r for r in rows if r['site'] == 'location_9' and r.get('ratio') == 20)
    assert high['delta_value'] == pytest.approx(.1)  # denominator min(20*.9, .1), not 18
    brier = next(r for r in rows if r['site'] == 'location_10' and r['kind'] == 'brier')
    assert brier['delta_bss'] == pytest.approx(.01 / (.05 * .95))
    assert brier['calibrated_brier_without'] == .025 and brier['calibrated_brier_with'] == .015
    assert brier['calibrated_delta_bss'] == pytest.approx(.01 / (.05 * .95))
    assert brier['bss_without'] == pytest.approx(1 - .03 / (.05 * .95))
    assert brier['relative_ap_change'] == pytest.approx(.25)
    mean = next(r for r in rows if r['scope'] == 'equal_airport_mean' and r.get('ratio') == 20)
    assert mean['delta_value'] == pytest.approx((.01 / .95 + .01 / .1) / 2)
    assert mean['n_airports'] == mean['n_airports_by_value']['delta_value'] == 2
    json.dumps(rows, allow_nan=False)


@pytest.mark.parametrize('positives', [0, 100])
def test_nox_value_rows_degenerate_prevalence_and_identical_arms(positives):
    from models._shared.report import nox_value_rows
    run = _report_run(sites=('location_10', 'location_9'))
    for site, pos in (('location_10', positives), ('location_9', 5)):
        result = _nox_ablation(positives=pos)
        result['with'] = deepcopy(result['without'])
        result['with']['ap'] = result['without']['ap'] = 0.
        run['per_site'][site]['aqi_ablation'] = {'nox': {'lead_time': {
            'xgboost': result, 'ebm': {'status': 'not_estimable', 'reason': 'insufficient classes'},
            'logistic_regression': {'status': 'not_applicable', 'reason': 'excluded fixture'},
            'random_forest': result}}}
    rows = nox_value_rows(run)
    assert {r['family'] for r in rows} == {'xgboost'}
    for row in rows:
        if row['kind'] == 'cost':
            assert row['delta_cost'] == 0.
            assert row['delta_value'] is None if row['site'] == 'location_10' else row['delta_value'] == 0.
        else:
            assert row['relative_ap_change'] is None
            assert row['delta_bss'] is None if row['site'] == 'location_10' else row['delta_bss'] == 0.
    mean = next(r for r in rows if r['scope'] == 'equal_airport_mean' and r['kind'] == 'brier')
    assert mean['n_airports'] == 2
    assert mean['n_airports_by_value']['delta_bss'] == 1
    json.dumps(rows, allow_nan=False)


def test_comparison_rejects_unequal_rosters_before_writing(tmp_path):
    from models._shared.report import write_comparison
    out = tmp_path / 'comparison'
    with pytest.raises(ValueError, match='unequal site rosters'):
        write_comparison({'xgboost': _report_run(),
                          'ebm': _report_run('ebm', sites=('location_10', 'location_9'))}, out)
    assert not out.exists()


def test_comparison_forwards_producing_environments_and_legacy_null(monkeypatch, tmp_path):
    from models._shared import report
    monkeypatch.setattr(report, 'write_plots', lambda *a, **kw: None)
    runs = {family: _report_run(family, sites=('location_10', 'location_9'))
            for family in ('xgboost', 'ebm')}
    for family, run in runs.items():
        for site, sd in run['per_site'].items():
            sd['environment'] = {'python': f'producer {family}/{site}'}
    del runs['ebm']['per_site']['location_9']['environment']
    combined = report.write_comparison(runs, tmp_path)
    for site, sd in combined['per_site'].items():
        assert sd['environments'] == {
            family: run['per_site'][site].get('environment') for family, run in runs.items()}
    assert combined['per_site']['location_9']['environments']['ebm'] is None


def test_comparison_descriptives_leave_earlier_sections_unchanged(monkeypatch, tmp_path):
    from models._shared import report
    monkeypatch.setattr(report, 'write_plots', lambda *a, **kw: None)
    run = _report_run()
    report.write_comparison({'xgboost': run}, tmp_path / 'baseline')
    baseline = (tmp_path / 'baseline' / 'model_comparison_v2.md').read_text(encoding='utf-8')
    assert report.nox_value_rows(run) == []
    run['per_site']['location_10']['aqi_ablation'] = {
        'nox': {'nowcast': {'xgboost': _nox_ablation()}}}
    # Remove only the new rows to isolate the new section from the existing AQ table/verdict.
    full = report.write_comparison({'xgboost': run}, tmp_path / 'full')
    assert len(full['nox_value_descriptives']) == 12
    text = (tmp_path / 'full' / 'model_comparison_v2.md').read_text(encoding='utf-8')
    preamble = ('Descriptive, retrospective misclassification-loss sensitivities at validation-selected thresholds '
                '(calibrated probabilities). Cost ratios are assumptions, not measured user costs. Raw Brier uses '
                'uncalibrated probabilities; Brier skill is relative to sample climatology. No inference is attached '
                'and no value is a practical-significance threshold.')
    assert preamble in text
    assert '### Per-airport descriptives' in text and '### Equal-airport means' in text
    assert text.split('## Matched AQI comparisons')[0] == baseline.split('## Matched AQI comparisons')[0]
    original_rows = report.nox_value_rows
    monkeypatch.setattr(report, 'nox_value_rows', lambda combined: [])
    report.write_comparison({'xgboost': run}, tmp_path / 'no_rows')
    no_rows = (tmp_path / 'no_rows' / 'model_comparison_v2.md').read_text(encoding='utf-8')
    heading = '## NOx decision-loss and Brier descriptives (primary families)'
    assert text.split(heading)[0] == no_rows.split(heading)[0]
    assert text.split('## Operational verdict')[1] == no_rows.split('## Operational verdict')[1]
    assert original_rows(run) == full['nox_value_descriptives']
    assert 'Capped NLL (eps=1e-9, nats) = mean(-log(clip(p_true_category, 1e-9, 1)))' in text


def test_main_not_estimable_survives_reports(monkeypatch, tmp_path):
    from models._shared import report
    monkeypatch.setattr(report, 'write_plots', lambda *a, **kw: None)
    run = _report_run('temporal_cnn')
    unavailable = {'status': 'not_estimable', 'reason': 'cnn tuning prefix lacks both classes'}
    for sd in run['per_site'].values():
        for task in sd['tasks']:
            sd['tasks'][task]['temporal_cnn'] = dict(unavailable)
    run['aggregate'] = report.aggregate(run)
    assert 'temporal_cnn' not in run['aggregate']['nowcast']
    report.write_family_report(run, tmp_path / 'family.txt')
    combined = report.write_comparison({'temporal_cnn': run}, tmp_path / 'comparison')
    assert combined['aggregate']['site_task_summary']['location_10']['nowcast']['temporal_cnn'] == unavailable
    assert unavailable['reason'] in (tmp_path / 'family.txt').read_text(encoding='utf-8')
    assert unavailable['reason'] in (tmp_path / 'comparison' / 'model_comparison_v2.md').read_text(encoding='utf-8')


def test_nox_value_rows_cover_primary_tasks_and_sample_climatology():
    from models._shared.report import nox_value_rows
    run = _report_run()
    result = _nox_ablation()
    for arm in ('with', 'without'):
        result[arm]['brier'] = .05 * .95
        result[arm]['calibration']['brier_after'] = .05 * .95
    run['per_site']['location_10']['aqi_ablation'] = {
        'nox': {task: {family: deepcopy(result) for family in PRIMARY_FAMILIES}
                for task in ('nowcast', 'lead_time')},
        'pm25': {'nowcast': {'xgboost': _nox_ablation()}}}
    rows = nox_value_rows(run)
    airport = [row for row in rows if row['scope'] == 'airport']
    assert len(airport) == 3 * 2 * 6
    for family in PRIMARY_FAMILIES:
        for task in ('nowcast', 'lead_time'):
            selected = [row for row in airport if row['family'] == family and row['task'] == task]
            assert {row['ratio'] for row in selected if row['kind'] == 'cost'} == {1, 3, 5, 10, 20}
            brier, = [row for row in selected if row['kind'] == 'brier']
            for key in ('bss_without', 'bss_with', 'delta_bss',
                        'calibrated_bss_without', 'calibrated_bss_with', 'calibrated_delta_bss'):
                assert brier[key] == 0.
