"""Synthetic, CPU-only and network-free checks for the settled long-term design."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location(
    "long_term_trend_v2", Path(__file__).with_name("long_term_trend_v2.py")
)
trend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trend)


def full_winter(year=2000):
    hours = pd.date_range(f"{year}-11-01", f"{year + 1}-04-01", freq="h", inclusive="left")
    weather = pd.DataFrame({"time": hours, **{name: np.zeros(len(hours)) for name in trend.WEATHER_COLS}})
    visibility = pd.DataFrame({"time": hours, "category": np.full(len(hours), 5.0)})
    return weather, visibility


def synthetic_panel():
    rows = []
    for number, site in enumerate(trend.SITES):
        for year in range(1980, 2027):
            step = int(site in trend.STEPS and pd.Timestamp(year, 11, 1) >= pd.Timestamp(trend.STEPS[site]))
            rows.append({"site": site, "winter": year, "year": year,
                         "y": 0.3 * number + 0.012 * (year - 2000) + 0.9 * step,
                         "dpd_c": 3.0, "fog_day_residual": 0.0})
    return pd.DataFrame(rows)


def test_one_month_at_89_percent_excludes_the_entire_winter():
    weather, visibility = full_winter()
    november = visibility.index[visibility.time.dt.month == 11]
    visibility.loc[november[int(0.89 * len(november)):], "category"] = np.nan
    daily, _ = trend.daily_table(weather, visibility, "location_7", start_year=2000, end_year=2000)
    coverage = trend.winter_validity(daily).iloc[0]
    assert coverage.month_coverage["11"] == pytest.approx(0.89, abs=0.005)  # 640 of 720 hours
    assert coverage.month_coverage["12"] == 1
    assert not coverage.valid
    assert not daily.valid.any()
    assert trend.winter_panel(daily, np.full(len(daily), 0.2)).empty
    # The boundary is inclusive, and the winter label stays the November year.
    visibility.loc[november[:int(0.90 * len(november))], "category"] = 5
    daily, _ = trend.daily_table(weather, visibility, "location_7", start_year=2000, end_year=2000)
    assert daily.valid.all()
    assert daily.winter.unique().tolist() == [2000]


def test_any_category_zero_through_four_defines_fog_day_and_unknown_is_not_known():
    weather, visibility = full_winter()
    visibility.loc[0, "category"] = 4
    visibility.loc[24, "category"] = 0
    visibility.loc[48:71, "category"] = np.nan
    visibility.loc[72, "category"] = np.nan
    visibility["category"] = visibility.category.astype("Int64")
    daily, X = trend.daily_table(weather, visibility, "location_6", start_year=2000, end_year=2000)
    assert daily.fog_day.iloc[:4].tolist() == [1, 1, 0, 0]
    assert daily.known_hours.iloc[:4].tolist() == [24, 24, 0, 23]
    assert X.shape == (len(daily), 24, len(trend.WEATHER_COLS))
    # Even with 90% monthly coverage, >=18-hour eligibility is day-specific.
    probability = np.full(len(daily), 0.25)
    probability[2] = np.nan
    panel = trend.winter_panel(daily, probability).iloc[0]
    assert panel.n_days == len(daily) - 1
    assert panel.observed == 2
    assert panel.expected == pytest.approx((len(daily) - 1) * 0.25)
    assert panel.y == pytest.approx(np.log(2.5 / (panel.expected + 0.5)))


def test_leave_one_winter_out_never_preprocesses_or_trains_on_held_out_sites():
    rows, raw = [], []
    for year in (2000, 2001, 2002):
        for site in ("location_6", "location_7"):
            for day in (1, 2):
                rows.append({"site": site, "date": pd.Timestamp(year, 11, day), "winter": year,
                             "valid": True, "known_hours": 24, "fog_day": day % 2})
                raw.append(np.full((24, len(trend.WEATHER_COLS)), year, dtype=np.float32))
    daily = pd.DataFrame(rows)
    calls, model_calls = [], []

    def preprocess(train, test):
        held = np.unique(test)
        assert len(held) == 1
        assert held[0] not in np.unique(train)
        assert len(train) == 8 and len(test) == 4
        transformed, imputer, scaler = trend.fit_preprocess(train, test)
        assert np.all(imputer.statistics_ == np.median(train, axis=(0, 1)))
        calls.append(int(held[0]))
        return transformed, imputer, scaler

    class Prediction:
        classes_ = np.array([0, 1])

        def predict_proba(self, X):
            assert len(X) == 4
            return np.tile([0.7, 0.3], (len(X), 1))

    def fit(family, X, y, *, seed):
        # Five summaries per weather channel, two seasonal terms, two sites;
        # no year or air-quality feature is appended.
        assert X.shape == (8, 5 * len(trend.WEATHER_COLS) + 2 + 2)
        assert np.bincount(y).tolist() == [4, 4]
        model_calls.append(family)
        return Prediction()

    predictions, folds = trend.diagnostic_predictions(daily, np.array(raw), "logistic_regression", fitter=fit, preprocess=preprocess)
    assert calls == [2000, 2001, 2002]
    assert model_calls == ["logistic_regression"] * 3
    assert np.allclose(predictions, 0.3)
    assert all(fold["status"] == "ok" for fold in folds)


def test_panel_ols_recovers_planted_year_slope_and_instrument_steps():
    panel = synthetic_panel()
    result = trend.fit_panel(panel, n_boot=50)
    assert result["status"] == "ok"
    assert result["coefficients"]["year"]["estimate"] == pytest.approx(0.012, abs=1e-12)
    for site in trend.STEPS:
        assert result["coefficients"][f"step_{site}"]["estimate"] == pytest.approx(0.9, abs=1e-12)
    assert result["r_squared"] == pytest.approx(1.0)
    assert result["valid_draws"] == 50
    post = trend.fit_panel(panel.loc[panel.winter >= 1996], n_boot=10)
    assert post["status"] == "ok"
    assert post["coefficients"]["year"]["estimate"] == pytest.approx(0.012, abs=1e-12)
    assert set(post["omitted_absorbed_steps"]) == {f"step_{site}" for site in trend.STEPS}


def test_bootstrap_resamples_whole_winters_jointly_with_all_sites():
    panel = pd.DataFrame({"winter": [1999, 2000, 2001, 1999, 2000, 2001, 2001],
                          "site": ["a", "a", "a", "b", "b", "b", "c"]})

    class Draw:
        def choice(self, winters, size, replace):
            assert winters.tolist() == [1999, 2000, 2001]
            assert size == 3 and replace
            return np.array([2001, 2001, 1999])

    index = trend.winter_bootstrap_indices(panel.winter, Draw())
    assert index.tolist() == [2, 5, 6, 2, 5, 6, 0, 3]
    counts = panel.iloc[index].groupby(["winter", "site"]).size().to_dict()
    assert counts == {(1999, "a"): 1, (1999, "b"): 1,
                      (2001, "a"): 2, (2001, "b"): 2, (2001, "c"): 2}


def test_exact_year_nox_collinearity_is_not_estimable():
    panel = synthetic_panel()
    panel["nox_ppb"] = 80 - 0.5 * (panel.year - 1980)
    result = trend.fit_panel(panel, ("year", "nox_ppb"), n_boot=5)
    assert result["status"] == "not_estimable"
    assert result["reason"] == "collinear"


def test_nox_only_signal_removes_year_association_in_joint_model():
    panel = synthetic_panel()
    rng = np.random.default_rng(41)
    panel["nox_ppb"] = 70 - 0.6 * (panel.year - 1980) + rng.normal(0, 8, len(panel))
    design, _, _ = trend.panel_design(panel, ("year", "nox_ppb"))
    noise = rng.normal(0, 0.03, len(panel))
    noise -= design @ np.linalg.lstsq(design, noise, rcond=None)[0]
    panel["y"] = 0.07 * panel.nox_ppb + noise
    nox = panel[["site", "winter", "nox_ppb"]]
    result = trend.attribution(panel.drop(columns="nox_ppb"), nox, n_boot=150, seed=12)
    models = result["models"]
    joint = models["year_plus_nox"]["coefficients"]
    assert joint["year"]["ci95"][0] < 0 < joint["year"]["ci95"][1]
    assert abs(joint["year"]["estimate"]) < 0.001
    assert joint["nox_ppb"]["estimate"] == pytest.approx(0.07, abs=0.002)
    assert models["year_only"]["coefficients"]["year"]["estimate"] < -0.02
    assert models["year_only"]["n_rows"] == models["nox_only"]["n_rows"] == models["year_plus_nox"]["n_rows"]
    assert result["year_slope_change_with_nox"] > 0.02


def test_measured_nox_absence_and_monthly_65_percent_rule(tmp_path):
    empty, status = trend.measured_nox("location_6", root=tmp_path)
    assert empty.empty and status["status"] == "not_available"
    assert "not available" in status["reason"]
    weather, _ = full_winter()
    hourly = weather[["time"]].copy()
    hourly["nox_ppb"] = np.nan
    for _, part in hourly.groupby(hourly.time.dt.to_period("M")):
        hourly.loc[part.index[:int(np.ceil(len(part) * 0.65))], "nox_ppb"] = 30.0
    folder = tmp_path / "datasets" / "aq_measured"
    folder.mkdir(parents=True)
    path = folder / "location_6_aq_measured.csv"
    hourly.to_csv(path, index=False)
    table, status = trend.measured_nox("location_6", root=tmp_path)
    assert status["status"] == "ok"
    assert table.winter.tolist() == [2000]
    assert table.nox_ppb.tolist() == [30.0]
    november = hourly.index[hourly.time.dt.month == 11]
    hourly.loc[november, "nox_ppb"] = np.nan
    hourly.loc[november[:int(len(november) * 0.64)], "nox_ppb"] = 30.0
    hourly.to_csv(path, index=False)
    table, status = trend.measured_nox("location_6", root=tmp_path)
    assert table.empty and status["status"] == "not_available"
    panel = synthetic_panel()
    assert trend.attribution(panel, table, n_boot=5)["status"] == "not_available"


def test_gray_optimal_day_thresholds_and_shares_are_hand_checkable():
    weather, visibility = full_winter()
    # Only the first four days pass the weather thresholds.
    weather["temperature_2m"] = 5.0
    weather["dew_point_2m"] = 0.0
    weather["wind_speed_10m"] = 8.0
    weather.loc[:95, "temperature_2m"] = 4.0
    weather.loc[:95, "wind_speed_10m"] = 7.0
    # Two optimal fog days, one non-optimal fog day: shares = 2/4 and 2/3.
    visibility.loc[[0, 24, 96], "category"] = 4
    # Equality is NOT optimal; test both boundaries on otherwise good days.
    weather.loc[120:143, "temperature_2m"] = 4.2
    weather.loc[120:143, "wind_speed_10m"] = 7.0
    weather.loc[144:167, "temperature_2m"] = 4.0
    weather.loc[144:167, "wind_speed_10m"] = 7.2
    daily, _ = trend.daily_table(weather, visibility, "location_6", start_year=2000, end_year=2000)
    assert daily.optimal.iloc[:7].tolist() == [True, True, True, True, False, False, False]
    result = trend.gray_replication(daily)["winters"][0]
    assert result["n_fog"] == 3 and result["n_optimal"] == 4 and result["n_both"] == 2
    assert result["fog_share_of_optimal"] == 0.5
    assert result["optimal_share_of_fog"] == pytest.approx(2 / 3)
    daily["optimal"] = False
    daily["fog_day"] = 0
    result = trend.gray_replication(daily)["winters"][0]
    assert result["fog_share_of_optimal"] is None
    assert result["optimal_share_of_fog"] is None


def test_json_markdown_and_png_outputs_include_unavailable_attribution(tmp_path):
    result = {"coverage": [], "nox_availability": {"location_6": {"status": "not_available"}},
              "gray_replication": {"winters": [], "trends": {}},
              "diagnostics": {"logistic_regression": {"winters": [], "folds": [],
                               "attribution": {"status": "not_available", "reason": "measured NOx winters not available"}}}}
    trend.write_outputs(result, tmp_path)
    assert '"status": "not_available"' in (tmp_path / "long_term_trend_v2.json").read_text()
    assert "measured NOx winters not available" in (tmp_path / "long_term_trend_v2.md").read_text(encoding="utf-8")
    for name in ("logistic_regression", "gray"):
        assert (tmp_path / f"long_term_trend_v2_{name}.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
