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
    pmf = np.eye(18)[categories]
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
                'c': categories.astype(np.int8), 'pmf': pmf.astype(np.float16),
                'dpd_c': np.array([1., np.nan, 7.], dtype=np.float32),
                'wind_kmh': np.array([2., 4., 8.], dtype=np.float32)}
    with np.load(path, allow_pickle=False) as saved:
        assert set(saved.files) == set(expected)
        for key, value in expected.items():
            assert saved[key].dtype == value.dtype
            np.testing.assert_array_equal(saved[key], value)


@pytest.mark.parametrize('old_nesting,predictions,reusable', [
    (True, True, False), (False, False, False), (False, True, True)])
def test_reuse_only_current_metrics_with_predictions(monkeypatch, tmp_path, old_nesting, predictions, reusable):
    family = 'xgboost'
    tasks = {task: {family: {'ordinal': {}}} for task in ('nowcast', 'lead_time')}
    sets = set(AQ_SETS) | {'nox_monitor'}
    ablation = tasks if old_nesting else {aq_set: tasks for aq_set in sets}
    saved = {'per_site': {'location_10': {'tasks': tasks, 'aqi_ablation': ablation}}}
    (tmp_path / f'{family}_metrics_v2.json').write_text(json.dumps(saved), encoding='utf-8')
    if predictions:
        (tmp_path / f'{family}_predictions_v2').mkdir()
    retrained, compared = [], []
    monkeypatch.setattr(run_all_v2, 'FAMILY_DIRS', {family: tmp_path})
    monkeypatch.setattr(sys, 'argv', ['run_all_v2.py', '--reuse'])
    monkeypatch.setattr(run_all_v2, 'run_family', lambda f, **kw: retrained.append(f) or {'fresh': f})
    monkeypatch.setattr(run_all_v2, 'write_comparison', lambda runs, out: compared.append(runs))
    run_all_v2.main()
    # stale saved metrics are retrained, never silently accepted
    assert retrained == ([] if reusable else [family])
    assert compared[0][family] == (saved if reusable else {'fresh': family})


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
        seen[aq_set] = aq_cols
        return {'fold_audit': []}
    monkeypatch.setattr(study, 'evaluate_ablation', ablation)
    monkeypatch.setattr(report, 'aggregate', lambda *a: {})
    monkeypatch.setattr(report, 'write_family_report', lambda *a: None)
    monkeypatch.setattr(report, 'write_plots', lambda *a, **kw: None)
    metrics = study.run_family(family, sites=('location_10',), out_dir=tmp_path)
    expected = set(AQ_SETS) | ({'nox_monitor'} if family in PRIMARY_FAMILIES else set())
    assert set(seen) == set(metrics['per_site']['location_10']['aqi_ablation']) == expected
    if family in PRIMARY_FAMILIES:
        assert seen['nox_monitor'] == ['nox_ppb', 'nox_mon_06-019-0001', 'nox_mon_06-019-0002']


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
    assert loaded.nox_ppb.iloc[1:3].tolist() == [7., 8.]
    assert np.isnan(loaded.nox_ppb.iloc[0])
    assert loaded['nox_mon_06-019-0001'].iloc[:3].tolist() == [0., 1., 0.]
    assert loaded['nox_mon_06-019-0002'].iloc[:3].tolist() == [0., 0., 1.]
