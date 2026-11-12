"""Synthetic measured-AQ, evaluation strata and saved-prediction contracts."""
import json
import sys

import numpy as np
import pandas as pd
import pytest

from models import run_all_v2
from models._shared import data, report, study
from models._shared.data import AQ_MIN_HOURS, AQ_SETS, CAL_COLS, PRIMARY_FAMILIES
from models._shared.labels import label_tables
from models._shared.windows import build_windows, prepare

from models._shared import artifacts


@pytest.fixture(autouse=True)
def observed_environment(monkeypatch):
    monkeypatch.setattr(study, 'capture_environment', lambda: {'python': 'synthetic producer'})

def weather_frame(n=72):
    times = pd.date_range('2024-01-01', periods=n, freq='h')
    return pd.DataFrame({'time': times, 'temperature_2m': np.arange(n, dtype=float),
                         'dew_point_2m': 0., 'wind_speed_10m': 3.,
                         **{col: 0. for col in CAL_COLS}})


def test_aq_twenty_hours_and_identical_arms_without_stratum_features():
    weather = weather_frame(48).assign(nox_ppb=10., pm25_ugm3=5.)
    weather.loc[[0, 1, 2, 3, 24, 25, 26, 27, 28], 'nox_ppb'] = np.nan
    # Eligibility is per channel, not the count of jointly complete hours.
    weather.loc[[4, 5, 6, 7], 'pm25_ugm3'] = np.nan
    labels = pd.DataFrame({'issue_time': weather.time.iloc[[23, 47]].to_numpy(),
                           'category': [2, 17], 'row_key': ['kept', 'dropped'],
                           'dpd_c': [123., 456.], 'wind_kmh': [789., 987.]})
    _, _, matched, cols = build_windows(weather, labels, aq_cols=['nox_ppb', 'pm25_ugm3'],
                                        aq_min_hours=AQ_MIN_HOURS,
                                        weather_cols=['temperature_2m'])
    assert matched.row_key.tolist() == ['kept']
    assert cols == ['temperature_2m'] + CAL_COLS + ['nox_ppb', 'pm25_ugm3']
    parts = {key: labels for key in ('train', 'val', 'test')}
    yes = prepare(weather, parts, aq_cols=['nox_ppb', 'pm25_ugm3'], aq_min_hours=20,
                  weather_cols=['temperature_2m'])
    no = prepare(weather, yes.metas, weather_cols=['temperature_2m'])
    for split in parts:
        assert yes.metas[split].row_key.tolist() == no.metas[split].row_key.tolist() == ['kept']
    assert not {'dpd_c', 'wind_kmh'} & set(yes.cols + no.cols)
    assert np.isfinite(yes.Xtr).all()  # Missing AQ hours were training-median imputed.
    assert yes.metas['test'].dpd_c.tolist() == [123.]


def test_strata_use_issue_hour_and_complete_verification_window():
    weather = weather_frame()
    vis = pd.DataFrame({'time': weather.time, 'category': 17})
    now, lead = label_tables(weather, vis, 'location_10')
    issue = pd.Timestamp('2024-01-01 18:00')
    assert now.loc[now.issue_time.eq(issue), 'dpd_c'].item() == 18.
    row = lead.loc[lead.issue_time.eq(issue)].iloc[0]
    assert row.dpd_c == np.mean(np.arange(24, 34))
    assert row.wind_kmh == 3.
    weather.loc[weather.time.eq('2024-01-02 04:00'), 'wind_speed_10m'] = np.nan
    _, incomplete = label_tables(weather, vis, 'location_10')
    row = incomplete.loc[incomplete.issue_time.eq(issue)].iloc[0]
    assert np.isnan(row.wind_kmh) and row.dpd_c == np.mean(np.arange(24, 34))
    weather = weather.loc[~weather.time.eq('2024-01-02 05:00')]
    _, missing_hour = label_tables(weather, vis, 'location_10')
    row = missing_hour.loc[missing_hour.issue_time.eq(issue)].iloc[0]
    assert np.isnan(row.dpd_c) and np.isnan(row.wind_kmh)


def test_predictions_roundtrip_exact_arrays_and_dtypes(tmp_path):
    times = pd.date_range('2024-01-01 18:00', periods=3, freq='D')
    categories = np.array([0, 4, 17])
    # probabilities that differ below half precision must survive the round trip (AP is rank-based)
    pmf = np.random.default_rng(0).dirichlet(np.ones(18), size=3)
    pmf[1, :5] = pmf[0, :5] + 1e-6
    records = []
    for fold, indices in ((1, [0, 1]), (2, [2])):
        records.append({'c': categories[indices], 'pmf': pmf[indices],
                        'meta': pd.DataFrame({'issue_time': times[indices], 'fold': fold,
                                              'dpd_c': np.array([1., np.nan, 7.])[indices],
                                              'wind_kmh': np.array([2., 4., 8.])[indices]})})
    path = tmp_path / 'predictions' / 'site__nowcast__main.npz'
    study.save_predictions(records, path)
    expected = {'issue_time': times.to_numpy(dtype='datetime64[ns]').astype(np.int64),
                'fold': np.array([1, 1, 2], dtype=np.int8),
                'c': categories.astype(np.int8), 'pmf': pmf,
                'dpd_c': np.array([1., np.nan, 7.], dtype=np.float32),
                'wind_kmh': np.array([2., 4., 8.], dtype=np.float32)}
    with np.load(path, allow_pickle=False) as saved:
        assert set(saved.files) == set(expected)
        for key, value in expected.items():
            assert saved[key].dtype == value.dtype
            np.testing.assert_array_equal(saved[key], value)


def completed_metrics(tmp_path, family='xgboost'):
    tasks = {task: {family: {'ordinal': {}}, 'climatology': {'ordinal': {}}}
             for task in run_all_v2.TASKS}
    sets = set(AQ_SETS) | ({'nox_monitor'} if family in PRIMARY_FAMILIES else set())
    ablation = {name: {task: {family: {'status': 'not_estimable', 'reason': 'fixture',
                  **({'contrast': 'nox_given_monitor'} if name == 'nox_monitor' else {})}}
                  for task in run_all_v2.TASKS} for name in sets}
    saved = {'family': family, 'inputs': {'location_10': {'source': 'fixture'}},
             'per_site': {'location_10': {'tasks': tasks, 'aqi_ablation': ablation}}}
    for name in study.prediction_inventory(saved):
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b'prediction')
    artifacts.atomic_write_json(artifacts.completion_path(tmp_path, family), {
        'schema_version': 1, 'family': family, 'sites': ['location_10'], 'inputs': saved['inputs'],
        'metrics_sha256': artifacts.canonical_hash(saved),
        'prediction_sha256': {name: artifacts.file_sha256(tmp_path / name)
                               for name in study.prediction_inventory(saved)}})
    return saved


@pytest.mark.parametrize('defect', ['none', 'empty', 'missing', 'changed', 'metric', 'roster',
                                  'inputs', 'obsolete', 'completion', 'contrast'])
def test_reuse_requires_one_complete_generation(tmp_path, defect):
    saved = completed_metrics(tmp_path)
    prediction = next((tmp_path / 'xgboost_predictions_v2').glob('*.npz'))
    inputs, sites = saved['inputs'], ['location_10']
    if defect == 'empty':
        for path in prediction.parent.glob('*.npz'):
            path.unlink()
    elif defect == 'missing':
        prediction.unlink()
    elif defect == 'changed':
        prediction.write_bytes(b'PREDICTION')
    elif defect == 'metric':
        saved['changed'] = True
    elif defect == 'roster':
        sites = list(data.SITES)
    elif defect == 'inputs':
        inputs = {'location_10': {'source': 'changed'}}
    elif defect == 'obsolete':
        (prediction.parent / 'location_6__nowcast__main.npz').write_bytes(b'old')
    elif defect == 'completion':
        artifacts.completion_path(tmp_path, 'xgboost').unlink()
    elif defect == 'contrast':
        del saved['per_site']['location_10']['aqi_ablation']['nox_monitor']['nowcast']['xgboost']['contrast']
    reason = run_all_v2.stale_reason('xgboost', tmp_path, saved, sites=sites, expected_inputs=inputs)
    assert (reason is None) == (defect == 'none')


def test_reuse_retrains_legacy_instead_of_aborting(monkeypatch, tmp_path):
    family = 'xgboost'
    (tmp_path / f'{family}_metrics_v2.json').write_text('{}', encoding='utf-8')
    retrained = []
    monkeypatch.setattr(run_all_v2, 'FAMILY_DIRS', {family: tmp_path})
    monkeypatch.setattr(run_all_v2, 'input_identity', lambda *a: {})
    monkeypatch.setattr(sys, 'argv', ['run_all_v2.py', '--reuse'])
    monkeypatch.setattr(run_all_v2, 'run_family', lambda f, **kw: retrained.append(f) or {})
    monkeypatch.setattr(run_all_v2, 'write_comparison', lambda *a: None)
    run_all_v2.main()
    assert retrained == [family]


def test_parallel_shards_share_tuned_cnn_and_isolate_failures(monkeypatch, tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    sites = ['location_6', 'location_10']
    seen = {}
    monkeypatch.setattr(run_all_v2, 'FAMILY_DIRS', {
        family: tmp_path / family for family in ('xgboost', 'temporal_cnn', 'ensemble')})

    def fake_run_family(family, *, sites, cnn_cache, write):
        (site,) = sites
        assert write is False
        if family == 'xgboost' and site == 'location_10':
            raise ValueError('boom')
        seen[family, site] = dict(cnn_cache)
        cfg = {task: cnn_cache.get((site, task), {'cfg': f'{site}-{task}', 'context': {'site': site}})
               for task in run_all_v2.TASKS}
        return {'family': family, 'inputs': {site: {'identity': site}}, 'per_site': {site: {
            'cnn_cfg': {t: r['cfg'] for t, r in cfg.items()},
            'cnn_tuning': {t: r['context'] for t, r in cfg.items()}}}}

    written = {}
    monkeypatch.setattr(run_all_v2, 'ProcessPoolExecutor', lambda jobs, **kw: ThreadPoolExecutor(jobs))
    monkeypatch.setattr(run_all_v2, 'run_family', fake_run_family)
    monkeypatch.setattr(run_all_v2, 'write_family', lambda m: written.setdefault(m['family'], m))
    with pytest.raises(RuntimeError, match="xgboost.*boom"):
        run_all_v2.run_parallel(['xgboost', 'temporal_cnn', 'ensemble'], sites, {}, jobs=3)
    # ensemble ran after temporal_cnn tuned each site and received exactly those configs
    for site in sites:
        assert {task: seen['ensemble', site].get((site, task)) for task in run_all_v2.TASKS} == {
            task: {'cfg': f'{site}-{task}', 'context': {'site': site}} for task in run_all_v2.TASKS}
    # complete families are written in site order; the failed family is not
    assert set(written) == {'temporal_cnn', 'ensemble'}
    assert list(written['ensemble']['per_site']) == sites
    assert written['ensemble']['inputs'] == {site: {'identity': site} for site in sites}


@pytest.mark.parametrize('family', [f for f in study.FAMILY_DIRS if f != 'xgboost_pooled'])
def test_monitor_one_hot_arm_only_in_primary_families(monkeypatch, tmp_path, family):
    weather = weather_frame().assign(nox_ppb=10., pm25_ugm3=3.,
                                     **{'nox_mon_06-019-0002': 0., 'nox_mon_06-019-0001': 1.})
    labels = pd.DataFrame({'fog': [False, True] * 20})
    monkeypatch.setattr(study, 'load_sources', lambda *a, **kw: (weather, None))
    monkeypatch.setattr(study, 'label_tables', lambda *a: (labels, labels.copy()))
    monkeypatch.setattr(study.FAMILIES[family], 'NEEDS_CNN_CFG', False)
    monkeypatch.setattr(study, 'evaluate_family', lambda *a, **kw: ({}, [], {}))
    monkeypatch.setattr(study, 'build_windows', lambda *a, **kw: (None, None, labels, None))
    seen = {}
    def ablation(*args, aq_set, aq_cols, **kwargs):
        seen[aq_set] = (aq_cols, kwargs['weather_cols'])
        return {'fold_audit': []}
    monkeypatch.setattr(study, 'evaluate_ablation', ablation)
    monkeypatch.setattr(report, 'aggregate', lambda *a: {})
    monkeypatch.setattr(report, 'write_family_report', lambda *a: None)
    monkeypatch.setattr(report, 'write_plots', lambda *a, **kw: None)
    metrics = study.run_family(family, sites=('location_10',), out_dir=tmp_path, write=False)
    expected = set(AQ_SETS) | ({'nox_monitor'} if family in PRIMARY_FAMILIES else set())
    assert set(seen) == set(metrics['per_site']['location_10']['aqi_ablation']) == expected
    if family in PRIMARY_FAMILIES:
        assert seen['nox_monitor'][0] == ['nox_ppb']
        assert seen['nox_monitor'][1] == [*data.WEATHER_COLS, 'nox_mon_06-019-0001', 'nox_mon_06-019-0002']
        assert metrics['per_site']['location_10']['aqi_ablation']['nox_monitor']['nowcast'][family]['contrast'] == 'nox_given_monitor'


def test_loader_left_joins_measured_aq_and_builds_monitor_indicators(monkeypatch, tmp_path):
    root = tmp_path / 'datasets'
    for folder in ('weather', 'aqi', 'visibility', 'aq_measured'):
        (root / folder).mkdir(parents=True)
    weather = weather_frame(30)
    weather.to_csv(root / 'weather/location_10_weather.csv', index=False)
    pd.DataFrame({'time': weather.time, **{col: 0. for col in data.AQI_COLS}}).to_csv(
        root / 'aqi/location_10_aqi.csv', index=False)
    pd.DataFrame({'time': weather.time, 'cat_lo': 17, 'cat_hi': 17, 'category': 17, 'fog': False}).to_csv(
        root / 'visibility/location_10_visibility_hourly.csv', index=False)
    monkeypatch.setattr(data, 'ROOT', tmp_path)
    absent, _ = data.load_sources('location_10')
    assert absent[data.MEASURED_AQ_COLS + ['nox_monitor', 'pm25_monitor']].isna().all().all()
    pd.DataFrame({'time': weather.time.iloc[[1, 2]], 'nox_ppb': [7., 8.],
                  'nox_monitor': ['06-019-0001', '06-019-0002'], 'pm25_ugm3': [3., 4.],
                  'pm25_monitor': ['06-019-0003'] * 2}).to_csv(
        root / 'aq_measured/location_10_aq_measured.csv', index=False)
    loaded, _ = data.load_sources('location_10')
    assert loaded.time.tolist() == weather.time.tolist()
    assert loaded.nox_ppb.iloc[2:4].tolist() == [7., 8.]
    assert loaded.nox_ppb.iloc[:2].isna().all()
    assert loaded['nox_mon_06-019-0001'].iloc[:4].tolist() == [0., 0., 1., 0.]
    assert loaded['nox_mon_06-019-0002'].iloc[:4].tolist() == [0., 0., 0., 1.]


@pytest.mark.parametrize('issue', ['2024-01-31 18:00', '2024-07-31 18:00', '2024-02-29 18:00'])
@pytest.mark.parametrize('completed', [19, 20])
def test_loader_window_excludes_issue_start_sentinel(monkeypatch, tmp_path, issue, completed):
    issue = pd.Timestamp(issue)
    times = pd.date_range(issue - pd.Timedelta(hours=47), periods=72, freq='h')
    root = tmp_path / 'datasets'
    for folder in ('weather', 'aqi', 'visibility', 'aq_measured'):
        (root / folder).mkdir(parents=True)
    weather = weather_frame(72).assign(time=times, **{col: 2. for col in data.WEATHER_COLS})
    weather.to_csv(root / 'weather/location_10_weather.csv', index=False)
    pd.DataFrame({'time': times, **{col: 3. for col in data.AQI_COLS}}).to_csv(
        root / 'aqi/location_10_aqi.csv', index=False)
    pd.DataFrame({'time': times, 'cat_lo': 17, 'cat_hi': 17, 'category': 17, 'fog': False}).to_csv(
        root / 'visibility/location_10_visibility_hourly.csv', index=False)
    raw_times = pd.date_range(issue - pd.Timedelta(hours=completed), periods=completed + 1, freq='h')
    measured = pd.DataFrame({'time': raw_times, 'nox_ppb': [7.] * completed + [999999.],
        'pm25_ugm3': [5.] * completed + [888888.], 'nox_monitor': ['old'] * completed + ['future'],
        'pm25_monitor': 'pm'})
    path = root / 'aq_measured/location_10_aq_measured.csv'
    measured.to_csv(path, index=False)
    before = path.read_bytes()
    monkeypatch.setattr(data, 'ROOT', tmp_path)
    loaded, _ = data.load_sources('location_10')
    labels = pd.DataFrame({'issue_time': [issue], 'category': [4], 'row_key': ['issue']})
    X, _, kept, columns = build_windows(loaded, labels, aq_cols=['nox_ppb'], aq_min_hours=20,
                                        weather_cols=['temperature_2m'])
    assert len(kept) == (completed == 20)
    assert loaded.loc[loaded.time.eq(issue), 'nox_ppb'].item() == 7.
    assert loaded.loc[loaded.time.eq(issue), 'nox_mon_old'].item() == 1.
    if len(kept):
        assert X[0, -1, columns.index('nox_ppb')] == 7.
        assert not np.any(X == 999999.)
    assert path.read_bytes() == before
    np.testing.assert_array_equal(loaded.temperature_2m, weather.temperature_2m)
    assert (loaded.pm10 == 3.).all()


def test_float32_origin_roundtrip_preserves_float64_fog_ranks(tmp_path):
    from models._shared.evaluation import pmf_to_fog, safe_ap
    pmf = np.zeros((2, 18), dtype=np.float32)
    pmf[:, 0] = .25
    pmf[:, 1] = [.25, np.nextafter(np.float32(.25), np.float32(np.inf))]
    pmf[:, 17] = 1 - pmf[:, :5].sum(axis=1)
    meta = pd.DataFrame({'issue_time': pd.date_range('2024-01-01', periods=2, freq='h'),
                         'fold': 1, 'dpd_c': np.nan, 'wind_kmh': 1.})
    path = tmp_path / 'roundtrip.npz'
    study.save_predictions([{'c': np.array([17, 0]), 'pmf': pmf, 'meta': meta}], path)
    before = pmf_to_fog(pmf)
    with np.load(path) as saved:
        after = pmf_to_fog(saved['pmf'])
    assert before.dtype == after.dtype == np.float64
    np.testing.assert_array_equal(before, after)
    assert before[0] < before[1]
    assert safe_ap(np.array([0, 1]), before) == safe_ap(np.array([0, 1]), after) == 1.


@pytest.mark.parametrize('family', ['logistic_regression', 'ensemble'])
def test_reuse_rejects_missing_or_failed_optimizer_diagnostics(tmp_path, family):
    saved = completed_metrics(tmp_path, family)
    sd = saved['per_site']['location_10']
    assert run_all_v2.stale_reason(family, tmp_path, saved, sites=['location_10'],
        expected_inputs=saved['inputs']) == 'ordered-logit convergence diagnostics missing or failed'
    diagnostics = dict(method='L-BFGS-B', maxiter=4000, maxfun=20000, maxls=50, maxcor=100,
        gtol=1e-7, ftol=0., nit=5, nfev=6, objective=1., grad_inf_norm=1e-8,
        projected_grad_inf_norm=1e-8, status=0, message='accepted', converged=True)
    info = {'optimization': diagnostics} if family == 'logistic_regression' else {
        'lr_optimization': [{'phase': 'oof', 'block': block, 'optimization': diagnostics}
                            for block in range(2, 6)] +
                           [{'phase': 'refit', 'block': None, 'optimization': diagnostics}]}
    sd['info'] = {task: {'folds': [info] * 3} for task in run_all_v2.TASKS}
    if family == 'ensemble':
        context = {'policy': 'pre_all_tests_v1', 'first_test_start': '2024-01-03 00:00:00',
            'cutoff': '2024-01-02 00:00:00', 'feature_columns': ['temperature_2m'],
            'train_sha256': 'a' * 64, 'valid_sha256': 'b' * 64, 'n_train': 2, 'n_valid': 2,
            'train_end': '2024-01-01 00:00:00', 'valid_end': '2024-01-01 12:00:00'}
        sd['cnn_tuning'] = {task: context for task in run_all_v2.TASKS}
        sd['cnn_cfg'] = {task: {'selected': 1} for task in run_all_v2.TASKS}
    assert run_all_v2._converged(sd, family)
    record = artifacts.load_completion(tmp_path, family)
    record['metrics_sha256'] = artifacts.canonical_hash(saved)
    artifacts.atomic_write_json(artifacts.completion_path(tmp_path, family), record)
    assert run_all_v2.stale_reason(family, tmp_path, saved, sites=['location_10'],
                                  expected_inputs=saved['inputs']) is None
    diagnostics['projected_grad_inf_norm'] = 1e-5
    assert not run_all_v2._converged(sd, family)


@pytest.mark.parametrize('failure', ['plots', 'inputs', 'missing_prediction'])
def test_failed_family_publish_has_no_completion(monkeypatch, tmp_path, failure):
    saved = completed_metrics(tmp_path)
    saved.update(weather_cols=['temperature_2m'], n_boot=2)
    monkeypatch.setattr(study, 'input_identity', lambda *a: saved['inputs']['location_10']
                        if failure != 'inputs' else {'changed': True})
    monkeypatch.setattr(report, 'aggregate', lambda *a: {})
    monkeypatch.setattr(report, 'write_family_report', lambda *a: None)
    def plots(*a, **kw):
        if failure == 'plots':
            raise RuntimeError('plot failed')
    monkeypatch.setattr(report, 'write_plots', plots)
    if failure == 'missing_prediction':
        next((tmp_path / 'xgboost_predictions_v2').glob('*.npz')).unlink()
    with pytest.raises(RuntimeError):
        study.write_family(saved, tmp_path)
    assert artifacts.load_completion(tmp_path, 'xgboost') is None


def test_completion_removes_only_obsolete_core_predictions(monkeypatch, tmp_path):
    saved = completed_metrics(tmp_path)
    saved.update(weather_cols=['temperature_2m'], n_boot=2)
    directory = tmp_path / 'xgboost_predictions_v2'
    obsolete = directory / 'location_6__nowcast__main.npz'
    obsolete.write_bytes(b'old')
    retained = [directory / 'location_6__nowcast__nox_lag_with.npz',
                directory / 'location_6__nowcast__pm25_fem_reference.npz',
                directory / 'location_6__nox.environment.json',
                directory / 'replicates/r1/location_6__nowcast__nox_with.npz']
    for path in retained:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'supplementary')
    monkeypatch.setattr(study, 'input_identity', lambda *a: saved['inputs']['location_10'])
    monkeypatch.setattr(report, 'aggregate', lambda *a: {})
    monkeypatch.setattr(report, 'write_family_report', lambda *a: None)
    monkeypatch.setattr(report, 'write_plots', lambda *a, **kw: None)
    study.write_family(saved, tmp_path)
    assert not obsolete.exists()
    assert all(path.read_bytes() == b'supplementary' for path in retained)
    assert run_all_v2.stale_reason('xgboost', tmp_path, saved, sites=['location_10'],
                                  expected_inputs=saved['inputs']) is None


def test_input_identity_binds_selected_files_absence_features_and_pooled_roster(monkeypatch, tmp_path):
    monkeypatch.setattr(data, 'ROOT', tmp_path)
    folder = tmp_path / 'datasets/weather'
    folder.mkdir(parents=True)
    raw = folder / 'location_10_weather.csv'
    appended = folder / 'location_10_weather_appended.csv'
    raw.write_bytes(b'unselected raw')
    appended.write_bytes(b'selected weather')
    first = study.input_identity('xgboost', ('location_10',), ['temperature_2m'], 2)
    assert 'datasets/weather/location_10_weather.csv' not in first['sources']
    assert first['sources']['datasets/aq_measured/location_10_aq_measured.csv']['sha256'] is None
    raw.write_bytes(b'changed unselected raw')
    assert first == study.input_identity('xgboost', ('location_10',), ['temperature_2m'], 2)
    appended.write_bytes(b'changed selected weather')
    assert first != study.input_identity('xgboost', ('location_10',), ['temperature_2m'], 2)
    assert first != study.input_identity('xgboost', ('location_10',), ['dew_point_2m'], 2)
    assert first != study.input_identity('xgboost', ('location_10',), ['temperature_2m'], 3)
    pooled = study.input_identity('xgboost_pooled', ('location_10',), ['temperature_2m'], 2)
    assert pooled['source_sites'] == sorted(data.SITES)
    assert len(pooled['sources']) == 3 * len(data.SITES) + 1


def test_direct_core_prediction_overwrite_invalidates_completion(tmp_path):
    saved = completed_metrics(tmp_path)
    path = tmp_path / 'xgboost_predictions_v2/location_10__nowcast__main.npz'
    meta = pd.DataFrame({'issue_time': pd.date_range('2024-01-01', periods=2, freq='h'),
                         'fold': 1, 'dpd_c': 2., 'wind_kmh': 3.})
    study.save_predictions([{'c': np.array([4, 17]), 'pmf': np.full((2, 18), 1/18),
                              'meta': meta}], path)
    assert artifacts.load_completion(tmp_path, saved['family']) is None


def test_monitor_prepared_columns_and_pair_keys(monkeypatch, tmp_path):
    from models._shared.families import FamilyOutput
    times = pd.date_range('2024-01-01', periods=24 * 121, freq='h')
    weather = pd.DataFrame({'time': times, 'temperature_2m': 2., 'nox_ppb': 3.,
        'nox_mon_a': np.arange(len(times)) % 2, 'nox_mon_b': (np.arange(len(times)) + 1) % 2,
        **{col: 0. for col in CAL_COLS}})
    issues = times[24::24]
    categories = np.where(np.arange(len(issues)) % 2, 4, 17)
    labels = pd.DataFrame({'issue_time': issues, 'row_key': issues.astype(str), 'category': categories,
                           'dpd_c': 1., 'wind_kmh': 3.})
    seen = []
    def predict(prep, *, cnn_cfg, seed):
        seen.append((seed, list(prep.cols), prep.metas['test'].row_key.tolist()))
        return FamilyOutput(np.full((len(prep.cv), 18), 1/18), np.full((len(prep.ct), 18), 1/18))
    monkeypatch.setattr(study.FAMILIES['xgboost'], 'fit_predict', predict)
    result = study.evaluate_ablation(weather, labels, 'location_10', 'nowcast', 'xgboost',
        cnn_cfg=None, weather_cols=['temperature_2m', 'nox_mon_a', 'nox_mon_b'], aq_cols=['nox_ppb'],
        seed=7, n_boot=2, prediction_dir=tmp_path, aq_set='nox_monitor')
    for without, with_nox in zip(seen[::2], seen[1::2]):
        assert without[0] == with_nox[0] and without[2] == with_nox[2]
        assert {'nox_mon_a', 'nox_mon_b'} <= set(without[1]) & set(with_nox[1])
        assert 'nox_ppb' not in without[1] and 'nox_ppb' in with_nox[1]
    assert {'nox_mon_a', 'nox_mon_b'} <= set(result['feature_columns']['without'])


def test_failed_prediction_write_keeps_old_bytes_and_invalidates_completion(monkeypatch, tmp_path):
    saved = completed_metrics(tmp_path)
    path = tmp_path / 'xgboost_predictions_v2/location_10__nowcast__main.npz'
    previous = path.read_bytes()
    meta = pd.DataFrame({'issue_time': [pd.Timestamp('2024-01-01')], 'fold': 1,
                         'dpd_c': 2., 'wind_kmh': 3.})
    def fail(handle, **kwargs):
        handle.write(b'partial')
        raise OSError('NPZ serialization failed')
    monkeypatch.setattr(study.np, 'savez_compressed', fail)
    with pytest.raises(OSError, match='NPZ serialization failed'):
        study.save_predictions([{'c': np.array([4]), 'pmf': np.full((1, 18), 1/18),
                                  'meta': meta}], path)
    assert path.read_bytes() == previous
    assert not list(path.parent.glob('*.part'))
    assert artifacts.load_completion(tmp_path, saved['family']) is None
