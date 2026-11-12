"""Three-arm PM sensitivity boundaries, eligibility and completion provenance."""
import json
import numpy as np
import pandas as pd
import pytest
from models import pm25_robustness_v2 as runner
from models._shared.data import WEATHER_COLS, CAL_COLS
from models._shared.families import FamilyOutput


def weather_data(n=96):
    times = pd.date_range('2024-01-01', periods=n, freq='h')
    return pd.DataFrame({'time': times, **{col: 1. for col in WEATHER_COLS + CAL_COLS},
                         'pm25_ugm3': np.arange(n, dtype=float) - 1,
                         'pm25_parameter': pd.array([88101] * n, dtype='Int64'),
                         'pm25_method': pd.array([170] * n, dtype='Int64')})


def test_candidate_latest_completed_sample_and_lag24():
    weather = weather_data()
    # At index 66 the corrected input represents the raw sample starting at index 65 (17:00).
    issue = pd.Timestamp('2024-01-03 18:00')
    lagged = runner.candidate_weather(weather, 'pm25_lag24')
    assert weather.loc[weather.time.eq(issue), 'pm25_ugm3'].item() == 65.
    assert lagged.loc[lagged.time.eq(issue), 'pm25_ugm3'].item() == 41.
    assert weather.time.iloc[65] == issue - pd.Timedelta(hours=1)
    assert weather.time.iloc[41] == issue - pd.Timedelta(hours=25)
    assert lagged.pm25_ugm3.iloc[:24].isna().all()
    assert weather.pm25_ugm3.iloc[66] == 65.  # no mutation of reference


@pytest.mark.parametrize('method,parameter,kept', [(170, 88101, True), (204, 88101, True),
    (209, 88101, True), (170, 88502, False), (999, 88101, False)])
def test_fem_masks_without_monitor_reselection(method, parameter, kept):
    weather = weather_data().assign(pm25_method=method, pm25_parameter=parameter)
    selected = runner.candidate_weather(weather, 'pm25_fem')
    assert selected.pm25_ugm3.notna().all() == kept
    assert selected.pm25_method.eq(method).all()
    assert weather.pm25_ugm3.notna().all()


def test_fem_unclassified_nonmissing_sample_fails():
    weather = weather_data()
    weather.loc[25, 'pm25_method'] = pd.NA
    with pytest.raises(ValueError, match='^pm25_fem: unclassified PM2.5 method$'):
        runner.candidate_weather(weather, 'pm25_fem')
    weather.loc[25, 'pm25_ugm3'] = np.nan
    runner.candidate_weather(weather, 'pm25_fem')


@pytest.mark.parametrize('candidate_count,reference_count,expected', [(20, 20, 1), (19, 20, 0), (20, 19, 0)])
def test_both_windows_require_twenty_hours(candidate_count, reference_count, expected):
    weather = weather_data(48)
    candidate = weather.copy()
    weather.loc[:47-reference_count, 'pm25_ugm3'] = np.nan
    candidate.loc[:47-candidate_count, 'pm25_ugm3'] = np.nan
    labels = pd.DataFrame({'issue_time': [weather.time.iloc[47]], 'category': [4], 'row_key': ['issue']})
    assert len(runner.matched_labels(weather, candidate, labels)) == expected


def test_three_arms_preserve_keys_categories_and_seeds(monkeypatch, tmp_path):
    weather = weather_data(24 * 121)
    issues = weather.time.iloc[24::24].reset_index(drop=True)
    c = np.where(np.arange(len(issues)) % 2, 4, 17)
    labels = pd.DataFrame({'issue_time': issues, 'row_key': issues.astype(str), 'category': c,
                          'dpd_c': 2., 'wind_kmh': 3.}).sample(frac=1, random_state=7)
    candidate = runner.candidate_weather(weather, 'pm25_lag24')
    seen = []
    def fit(prep, *, cnn_cfg, seed):
        seen.append((seed, {key: value.row_key.tolist() for key, value in prep.metas.items()}, list(prep.cols)))
        return FamilyOutput(np.full((len(prep.cv), 18), 1/18), np.full((len(prep.ct), 18), 1/18))
    monkeypatch.setattr(runner.FAMILIES['xgboost'], 'fit_predict', fit)
    files = runner.evaluate_check(weather, candidate, labels, 'location_10', 'nowcast', 'xgboost',
                                  'pm25_lag24', tmp_path)
    assert len(files) == 3 and len(seen) == 9
    for offset in range(0, len(seen), 3):
        rows = seen[offset:offset+3]
        assert len({row[0] for row in rows}) == 1
        assert rows[0][1] == rows[1][1] == rows[2][1]
        assert rows[0][2] == rows[1][2]
        assert 'pm25_ugm3' not in rows[2][2]
    with np.load(tmp_path / files[0]) as a, np.load(tmp_path / files[1]) as b, np.load(tmp_path / files[2]) as c:
        for key in ('issue_time', 'fold', 'c'):
            np.testing.assert_array_equal(a[key], b[key])
            np.testing.assert_array_equal(a[key], c[key])


def test_unit_manifest_last_and_not_estimable_status(monkeypatch, tmp_path):
    monkeypatch.setitem(runner.FAMILY_DIRS, 'xgboost', tmp_path)
    monkeypatch.setattr(runner, 'capture_environment', lambda: {'python': 'actual producer'})
    monkeypatch.setattr(runner, 'load_sources', lambda *a, **kw: (weather_data(), None))
    monkeypatch.setattr(runner, 'label_tables', lambda *a: (None, None))
    def unavailable(*args):
        assert not runner.complete('xgboost', 'location_10', 'pm25_lag24')
        raise runner.NotEstimable('insufficient validation classes')
    monkeypatch.setattr(runner, 'evaluate_check', unavailable)
    assert runner.run_unit('xgboost', 'location_10', 'pm25_lag24') == ('xgboost', 'location_10', 'pm25_lag24')
    manifest = json.loads((runner.prediction_dir('xgboost') / 'location_10__pm25_lag24.environment.json').read_text())
    assert manifest['prediction_files'] == []
    assert manifest['environment'] == {'python': 'actual producer'}
    assert all(value == {'status': 'not_estimable', 'reason': 'insufficient validation classes'}
               for value in manifest['tasks'].values())
    assert runner.complete('xgboost', 'location_10', 'pm25_lag24')
    def broken(*args):
        raise RuntimeError('fit failed')
    monkeypatch.setattr(runner, 'evaluate_check', broken)
    with pytest.raises(RuntimeError, match='fit failed'):
        runner.run_unit('xgboost', 'location_10', 'pm25_lag24')
    assert not runner.complete('xgboost', 'location_10', 'pm25_lag24')


def test_single_class_validation_stops_before_any_arm_fits(monkeypatch, tmp_path):
    from unittest.mock import Mock
    weather = weather_data(24 * 121)
    issues = weather.time.iloc[24::24].reset_index(drop=True)
    labels = pd.DataFrame({'issue_time': issues, 'row_key': issues.astype(str), 'category': 17,
                           'dpd_c': 2., 'wind_kmh': 3.})
    fit = Mock(side_effect=AssertionError('training started'))
    monkeypatch.setattr(runner.FAMILIES['xgboost'], 'fit_predict', fit)
    with pytest.raises(runner.NotEstimable, match='validation requires both classes; keep fixed folds'):
        runner.evaluate_check(weather, runner.candidate_weather(weather, 'pm25_lag24'), labels,
                              'location_10', 'nowcast', 'xgboost', 'pm25_lag24', tmp_path)
    fit.assert_not_called()
    assert not list(tmp_path.glob('*.npz'))
