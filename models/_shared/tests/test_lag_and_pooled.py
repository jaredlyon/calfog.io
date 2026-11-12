"""Synthetic causal-lag, pooled-fold and report regression contracts."""
from types import SimpleNamespace

import numpy as np
import pandas as pd

from models._shared import report, study, windows
from models._shared.data import AQI_COLS, CAL_COLS, SITES
from models._shared.families import FAMILIES, FamilyOutput
from models._shared.labels import add_neighbor_lags, hourly_visibility_lag, label_tables
from models._shared.visibility import N_CATEGORIES


def _weather(times, offset=0):
    return pd.DataFrame({'time': times, 'custom': np.arange(len(times), dtype=float) + offset,
                         **{col: np.zeros(len(times)) for col in CAL_COLS + AQI_COLS}})


def _labels(times, site):
    categories = np.where(np.arange(len(times)) % 2, 2, 17)
    return pd.DataFrame({'issue_time': times, 'category': categories, 'fog': categories <= 4,
                         'row_key': times.astype(str), 'site': site,
                         'lag_category': pd.array(categories, dtype='Int64'),
                         'lag_time': times - pd.Timedelta(hours=1),
                         'nbr_min_cat': 2., 'nbr_mean_cat': 9.5,
                         'nbr_fog_count': 1, 'nbr_reporting': 2})


def _predict(prep, *, cnn_cfg, seed):
    return FamilyOutput(np.full((len(prep.cv), N_CATEGORIES), 1 / N_CATEGORIES),
                        np.full((len(prep.ct), N_CATEGORIES), 1 / N_CATEGORIES))


def test_lag_uses_previous_hour_for_both_tasks_not_issue_hour():
    times = pd.date_range('2024-01-01', periods=72, freq='h')
    vis = pd.DataFrame({'time': times, 'category': np.full(len(times), 17)})
    vis.loc[17, 'category'] = 4
    vis.loc[18, 'category'] = 0  # This target-hour report is not yet available.
    now, lead = label_tables(_weather(times), vis, 'a')
    for table in (now, lead):
        row = table.loc[table.issue_time.eq(times[18])].iloc[0]
        assert row.lag_category == 4
        assert row.lag_time == times[17]
        assert str(table.lag_category.dtype) == 'Int64'
        assert pd.api.types.is_datetime64_any_dtype(table.lag_time)


def test_lag_staler_than_three_hours_is_missing():
    times = pd.date_range('2024-01-01', periods=48, freq='h')
    vis = pd.DataFrame({'time': times[[8, 9, 12]], 'category': [4, np.nan, 0]})
    now, _ = label_tables(_weather(times), vis, 'a')
    row = now.loc[now.issue_time.eq(times[12])].iloc[0]
    assert pd.isna(row.lag_category)
    assert pd.isna(row.lag_time)


def test_hourly_visibility_lag_matches_issue_lag_for_both_tasks():
    times = pd.date_range('2024-01-01', periods=96, freq='h')
    rng = np.random.default_rng(0)
    vis = pd.DataFrame({'time': times, 'category': rng.integers(0, 18, len(times)).astype(float)})
    vis.loc[30:34, 'category'] = np.nan  # a gap longer than the 2 h staleness tolerance
    weather = _weather(times)
    hourly = hourly_visibility_lag(weather, vis).set_axis(weather.time)
    for table in label_tables(weather, vis, 'a'):
        expected = table.lag_category.astype(float).to_numpy()
        np.testing.assert_array_equal(hourly.reindex(table.issue_time).to_numpy(), expected)
    assert np.isnan(hourly.loc[times[35]])  # hours 32–34 are all missing: stale beyond 2 h
    assert hourly.loc[times[30]] == vis.category[29]  # issue 30 sees hour 29, never its own hour


def test_neighbor_aggregates_exclude_self_issue_hour_and_stale_reports():
    issue = pd.Timestamp('2024-01-02T18:00:00')
    labels = {site: pd.DataFrame({'issue_time': [issue, issue + pd.Timedelta(hours=4)]}, index=[8, 2])
              for site in ('a', 'b', 'c')}
    vis = {
        'a': pd.DataFrame({'time': [issue - pd.Timedelta(hours=1)], 'category': [0]}),
        'b': pd.DataFrame({'time': [issue - pd.Timedelta(hours=1), issue], 'category': [2, 17]}),
        'c': pd.DataFrame({'time': [issue - pd.Timedelta(hours=3)], 'category': [10]}),
    }
    actual = add_neighbor_lags(labels, vis)
    for site, expected in {'a': (2., 6., 1, 2), 'b': (0., 5., 1, 2), 'c': (0., 1., 2, 2)}.items():
        frame = actual[site]
        assert frame.index.tolist() == [8, 2]
        assert tuple(frame.loc[8, ['nbr_min_cat', 'nbr_mean_cat', 'nbr_fog_count', 'nbr_reporting']]) == expected
        assert np.isnan(frame.loc[2, 'nbr_min_cat']) and np.isnan(frame.loc[2, 'nbr_mean_cat'])
        assert frame.loc[2, 'nbr_reporting'] == frame.loc[2, 'nbr_fog_count'] == 0
        assert 'nbr_reporting' not in labels[site]  # Input tables stay untouched.


def test_prepare_pooled_fits_once_on_pooled_train_and_preserves_metadata(monkeypatch):
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    times = pd.date_range('2024-01-01', periods=120, freq='h')
    sources = {site: _weather(times, offset) for site, offset in (('b', 100), ('a', 0))}
    parts = {}
    for site in sources:
        labels = _labels(times, site)
        parts[site] = {split: labels.iloc[index].copy() for split, index in
                       {'train': [31, 30], 'val': [81, 80], 'test': [111, 110]}.items()}
    fitted = {'imputer': [], 'scaler': []}
    for cls, name in ((SimpleImputer, 'imputer'), (StandardScaler, 'scaler')):
        original = cls.fit
        def tracked(self, X, *args, _original=original, _name=name, **kwargs):
            fitted[_name].append(X.copy())
            return _original(self, X, *args, **kwargs)
        monkeypatch.setattr(cls, 'fit', tracked)
    prep = windows.prepare_pooled(sources, parts, weather_cols=['custom'])
    assert len(fitted['scaler']) == len(fitted['imputer']) == 1
    raw = np.concatenate([windows.build_windows(sources[s], parts[s]['train'], weather_cols=['custom'])[0]
                          for s in sources])
    np.testing.assert_allclose(fitted['scaler'][0].mean(axis=0), raw.reshape(-1, raw.shape[-1]).mean(axis=0))
    np.testing.assert_allclose(prep.Xtr.mean(axis=(0, 1)), 0., atol=1e-6)
    for split, X in (('train', prep.Xtr), ('val', prep.Xv), ('test', prep.Xt)):
        meta = prep.metas[split]
        assert meta.site.tolist() == ['a', 'b', 'a', 'b']
        assert meta.issue_time.is_monotonic_increasing
        assert set(parts['a'][split]) <= set(meta)
        # The two airport windows must stay separate, not become one interleaved history.
        assert np.all(X[meta.site.eq('b'), :, 0] > X[meta.site.eq('a'), :, 0])
    sources['b'].loc[80, 'custom'] = 100000.
    changed = windows.prepare_pooled(sources, parts, weather_cols=['custom'])
    assert len(fitted['scaler']) == len(fitted['imputer']) == 2
    np.testing.assert_array_equal(prep.Xtr, changed.Xtr)
    assert not np.array_equal(prep.Xv, changed.Xv)
    # The nonpooled route must also preserve every lag/neighbor label column.
    single = windows.prepare(sources['a'], parts['a'], weather_cols=['custom'])
    pd.testing.assert_frame_equal(single.metas['train'], parts['a']['train'].reset_index(drop=True))


def test_evaluate_pooled_applies_target_validation_embargo_and_target_only_test(monkeypatch):
    times = pd.date_range('2024-01-01', periods=168, freq='h')
    sources = {site: _weather(times, offset) for site, offset in (('a', 0), ('b', 100), ('c', 200))}
    labels = {site: _labels(times[24:], site) for site in sources}
    V, T = times[72], times[120]
    target = labels['a']
    fold = {'train': target.loc[target.issue_time < V - pd.Timedelta(hours=24)],
            'val': target.loc[(target.issue_time >= V) & (target.issue_time < T - pd.Timedelta(hours=24))],
            'test': target.loc[target.issue_time.between(T, times[123])]}
    monkeypatch.setattr(study, 'rolling_origin_folds', lambda frame: [fold])
    seen = []
    def predict(prep, **kwargs):
        seen.append(prep)
        train, val, test = (prep.metas[s] for s in ('train', 'val', 'test'))
        assert (train.issue_time < V - pd.Timedelta(hours=24)).all()
        assert train.issue_time.max() == times[47]
        assert set(train.site) == set(sources)
        assert (val.issue_time >= V).all() and (val.issue_time < T - pd.Timedelta(hours=24)).all()
        assert set(val.site) == set(sources)
        assert set(test.site) == {'a'}
        assert test.row_key.tolist() == fold['test'].row_key.tolist()
        return _predict(prep, **kwargs)
    monkeypatch.setitem(FAMILIES, 'pooled_stub', SimpleNamespace(fit_predict=predict))
    result, audit, info = study.evaluate_pooled(sources, labels, 'a', 'nowcast', 'pooled_stub',
                                               weather_cols=['custom'], seed=7, n_boot=2)
    assert len(seen) == 1
    assert audit[0]['pooled_sites'] == {'a': 24, 'b': 24, 'c': 24}
    assert audit[0]['test'] == 4
    assert result['ordinal']['rps'] >= 0
    assert info == {'folds': [{}]}


def test_run_pooled_loads_all_sites_once_and_reports_not_applicable(monkeypatch, tmp_path):
    times = pd.date_range('2024-01-01', periods=168, freq='h')
    loaded, pooled_calls, baseline_sites = [], [], []
    def load(site, **kwargs):
        loaded.append(site)
        return _weather(times), pd.DataFrame({'time': times, 'category': np.where(np.arange(len(times)) % 2, 2, 17)})
    monkeypatch.setattr(study, 'load_sources', load)
    monkeypatch.setattr(study, 'capture_environment', lambda: {'python': 'pooled producer'})
    # Use one small fixed fold, while retaining real label, window, evaluation and reporting paths.
    def folds(labels):
        return [{'train': labels.loc[labels.issue_time < times[48]],
                 'val': labels.loc[labels.issue_time.between(times[72], times[95])],
                 'test': labels.loc[labels.issue_time.between(times[120], times[143])]}]
    monkeypatch.setattr(study, 'rolling_origin_folds', folds)
    def predict(prep, **kwargs):
        pooled_calls.append(prep)
        assert set(prep.metas['train'].site) == set(SITES)
        assert prep.metas['train'].issue_time.max() < times[48]
        assert set(prep.metas['test'].site) == {next(iter(SITES))}
        return _predict(prep, **kwargs)
    monkeypatch.setitem(FAMILIES, 'xgboost_pooled', SimpleNamespace(POOLED=True, NEEDS_CNN_CFG=False, fit_predict=predict))
    original_baseline = FAMILIES['climatology'].fit_predict
    def baseline(prep, **kwargs):
        baseline_sites.append(set(prep.metas['train'].site))
        return original_baseline(prep, **kwargs)
    monkeypatch.setattr(FAMILIES['climatology'], 'fit_predict', baseline)
    monkeypatch.setattr(study, 'evaluate_ablation', lambda *a, **kw: (_ for _ in ()).throw(AssertionError('pooled AQI ablation')))
    monkeypatch.setattr(report, 'write_plots', lambda *a, **kw: None)
    site = next(iter(SITES))
    metrics = study.run_family('xgboost_pooled', sites=(site,), weather_cols=['custom'], out_dir=tmp_path, n_boot=2)
    assert loaded == list(SITES)
    assert len(pooled_calls) == 2
    assert baseline_sites == [{site}, {site}]
    assert metrics['predictor_set'] == study.PREDICTOR_SETS['xgboost_pooled']
    assert metrics['per_site'][site]['environment'] == {'python': 'pooled producer'}
    assert metrics['inputs'][site]['source_sites'] == sorted(SITES)
    for task in ('nowcast', 'lead_time'):
        for aq_set in ('nox', 'pm25', 'cams'):
            assert metrics['per_site'][site]['aqi_ablation'][aq_set][task]['xgboost_pooled'] == {
                'status': 'not_applicable', 'reason': 'pooled multi-site model; AQI matched cohort is per site'}
            assert metrics['aggregate']['aqi_site_summary'][site][aq_set][task] == {}
    assert len(metrics['aggregate']['aqi_not_applicable']) == 6
    assert metrics['aggregate']['aqi_not_estimable'] == []
    text = (tmp_path / 'xgboost_pooled_results_v2.txt').read_text(encoding='utf-8')
    assert 'AQI ablation: not applicable' in text
    assert study.PREDICTOR_SETS['xgboost_pooled'] in text
    assert 'Paired AQI delta' not in text
    combined = report.write_comparison({'xgboost_pooled': metrics}, tmp_path / 'comparison')
    assert len(combined['aggregate']['aqi_not_applicable']) == 6
    markdown = (tmp_path / 'comparison' / 'model_comparison_v2.md').read_text(encoding='utf-8')
    assert '| Predictor set |' in markdown
    assert study.PREDICTOR_SETS['xgboost_pooled'] in markdown
    assert '- AQI not applicable:' in markdown


def test_lag_runner_completed_nox_excludes_issue_hour_sentinel(monkeypatch, tmp_path):
    from models import lag_aware_nox_v2 as runner
    from models._shared import data
    times = pd.date_range('2024-01-01', periods=72, freq='h')
    weather = pd.DataFrame({'time': times, **{col: 1. for col in data.WEATHER_COLS}})
    measured = pd.DataFrame({'time': times[27:48], 'nox_ppb': [7.] * 20 + [999999.],
        'pm25_ugm3': 3., 'nox_monitor': 'old', 'pm25_monitor': 'pm'})
    vis = pd.DataFrame({'time': times, 'cat_lo': 17, 'cat_hi': 17, 'category': 17, 'fog': False})
    root = tmp_path / 'datasets'
    for folder in ('weather', 'aqi', 'visibility', 'aq_measured'):
        (root / folder).mkdir(parents=True)
    (root / 'weather/location_10_weather.csv').touch()
    (root / 'aq_measured/location_10_aq_measured.csv').touch()
    def read_csv(path, **kwargs):
        if 'aq_measured' in str(path):
            return measured.copy()
        if 'visibility' in str(path):
            return vis.copy()
        if 'aqi' in str(path):
            return pd.DataFrame({'time': times, **{col: 0. for col in data.AQI_COLS}})
        return weather.copy()
    monkeypatch.setattr(data, 'ROOT', tmp_path)
    monkeypatch.setattr(pd, 'read_csv', read_csv)
    monkeypatch.setattr(runner, 'capture_environment', lambda: {'python': 'lag producer'})
    monkeypatch.setattr(runner, 'prediction_dir', lambda *a: tmp_path / 'predictions')
    labels = pd.DataFrame({'issue_time': [times[47]], 'category': [4], 'row_key': ['issue']})
    monkeypatch.setattr(runner, 'label_tables', lambda *a: (labels, labels))
    seen = []
    def evaluate(source, matched, site, task, family, **kwargs):
        assert matched.row_key.tolist() == ['issue']
        X, _, _, columns = windows.build_windows(source, matched, aq_cols=['nox_ppb'],
            aq_min_hours=20, weather_cols=runner.LAG_WEATHER_COLS)
        assert X[0, -1, columns.index('nox_ppb')] == 7.
        assert not np.any(X == 999999.)
        seen.append(task)
        return {}
    monkeypatch.setattr(runner, 'evaluate_ablation', evaluate)
    runner.run_unit('xgboost', 'location_10')
    assert seen == ['nowcast', 'lead_time']
