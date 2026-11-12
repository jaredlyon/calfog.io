"""Synthetic, CPU-only and network-free checks for the settled long-term design."""
import importlib.util
import json
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
    assert result["n_boot"] == 50
    assert result["method"] == "dependent_wild_bootstrap"
    assert result["coefficients"]["year"]["valid_draws"] == 50
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


def test_wild_multipliers_share_winter_and_preserve_calendar_gaps():
    years = np.array([2000, 2000, 2002, 2003])
    seed, length = 17, 3
    normals = np.random.default_rng(seed).standard_normal(4 + length - 1)
    weights = np.array([normals[i:i + length].sum() / np.sqrt(length) for i in range(4)])
    actual = trend._winter_multipliers(years, length, np.random.default_rng(seed))
    assert actual[0] == actual[1]
    np.testing.assert_allclose(actual, weights[[0, 0, 2, 3]], rtol=0, atol=1e-15)


def test_wild_bootstrap_matches_fixed_design_refit():
    panel = synthetic_panel()
    panel["y"] += np.random.default_rng(1).normal(0, .1, len(panel))
    X, names, _ = trend.panel_design(panel)
    y = panel.y.to_numpy()
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    P = np.linalg.pinv(X)
    h = np.diag(X @ P)
    length = int(np.ceil(47 ** (1 / 3)))
    weights = trend._winter_multipliers(panel.winter.to_numpy(), length, np.random.default_rng(3))
    refit = np.linalg.lstsq(X, X @ beta + (y - X @ beta) / np.sqrt(1 - h) * weights, rcond=None)[0]
    result = trend.fit_panel(panel, n_boot=1, seed=3)
    for j, name in enumerate(names):
        radius = abs(refit[j] - beta[j])
        assert result["coefficients"][name]["ci95"] == pytest.approx([beta[j] - radius, beta[j] + radius], abs=1e-12)
    assert result["calendar_span"] == 47
    assert set(result["dependence_sensitivity"]) == {"1", "3", "4", "5"}


def _sparse_panel():
    return pd.DataFrame({"site": ["a"] * 6 + ["b"], "winter": list(range(2000, 2006)) + [2000],
                         "nox_ppb": [0, 1, 2, 3, 4, 5, 2],
                         "y": [3, 5, 7, 9, 11, 13, 7]})


def test_pairs_winter_keeps_supported_slope_despite_missing_nuisance(monkeypatch):
    monkeypatch.setattr(trend, "winter_bootstrap_indices", lambda years, rng: np.array([1, 2, 3, 4, 5, 5]))
    result = trend.fit_panel(_sparse_panel(), ("nox_ppb",), steps={}, n_boot=2)
    slope = result["coefficients"]["nox_ppb"]
    assert slope["estimate"] == pytest.approx(2)
    assert slope["ci95"] is None
    assert slope["inference_status"] == "not_available"
    assert slope["inference_reason"] == "unit_leverage"
    assert slope["pairs_winter"]["valid_draws"] == 2
    assert slope["pairs_winter"]["ci95"] == pytest.approx([2, 2])
    nuisance = result["coefficients"]["intercept_b"]["pairs_winter"]
    assert nuisance["ci95"] is None
    assert nuisance["inference_reason"] == "incomplete_bootstrap_support"
    assert result["pairs_winter_full_rank_draws"] == 0


def test_pairs_winter_refuses_incomplete_slope_support(monkeypatch):
    monkeypatch.setattr(trend, "winter_bootstrap_indices", lambda years, rng: np.zeros(7, dtype=int))
    result = trend.fit_panel(_sparse_panel(), ("nox_ppb",), steps={}, n_boot=2)
    slope = result["coefficients"]["nox_ppb"]
    assert slope["estimate"] == pytest.approx(2)
    assert slope["pairs_winter"]["ci95"] is None
    assert slope["pairs_winter"]["p"] is None
    assert slope["pairs_winter"]["valid_draws"] == 0


def test_rank_calendar_bootstrap_matches_independent_length_one():
    years = np.arange(2000, 2008)
    values = np.array([0, 2, 1, 4, 3, 8, 5, 9], dtype=float)
    slope = trend.stats.theilslopes(values, years).slope
    tau = trend.stats.kendalltau(years, values).statistic
    residual = values - np.median(values - slope * years) - slope * years
    rng = np.random.default_rng(8)
    slopes, taus = [], []
    for _ in range(12):
        sample = residual[rng.integers(0, 8, size=8)]
        slopes.append(trend.stats.theilslopes(sample, years).slope)
        taus.append(0 if np.ptp(sample) == 0 else trend.stats.kendalltau(years, sample).statistic)
    radius = np.quantile(np.abs(slopes), .95)
    expected_p = (1 + np.count_nonzero(np.abs(taus) >= abs(tau))) / 13
    result = trend.monotone_trend(years, values, n_boot=12, seed=8)
    one = result["dependence_sensitivity"]["1"]
    assert one["ci95"] == pytest.approx([slope - radius, slope + radius])
    assert one["mann_kendall_p"] == expected_p


def test_sparse_kvis_length_five_is_unavailable_without_redrawing(monkeypatch):
    class GapDraw:
        def integers(self, low, high, size):
            return np.full(size, 5)
    monkeypatch.setattr(trend.np.random, "default_rng", lambda seed: GapDraw())
    result = trend.monotone_trend([1992, 1993, 2015], [1, 2, 3], n_boot=4)
    five = result["dependence_sensitivity"]["5"]
    assert result["calendar_span"] == 24
    assert five["inference_status"] == "not_available"
    assert five["ci95"] is None and five["mann_kendall_p"] is None
    assert five["n_boot"] == 4


def test_constant_rank_series_has_zero_slope_tau_and_p_one():
    result = trend.monotone_trend(np.arange(10), np.ones(10), n_boot=5)
    assert result["slope"] == result["kendall_tau"] == 0
    assert result["ci95"] == [0, 0]
    assert result["mann_kendall_p"] == 1


def test_site_rank_trends_are_separate_and_empty_is_empty():
    frame = pd.DataFrame({"site": ["a"] * 3 + ["b"] * 3,
                          "winter": [1980, 1981, 1982, 2000, 2001, 2002],
                          "value": [0, .1, .2, .2, .1, 0]})
    frame.attrs.update(n_boot=3, seed=7)
    result = trend.site_trends(frame, "value")
    assert set(result) == {"a", "b"}
    for site in ("a", "b"):
        part = frame.loc[frame.site == site]
        assert result[site] == trend.monotone_trend(part.winter, part.value, n_boot=3, seed=7)
    assert result["a"]["slope"] == pytest.approx(.1)
    assert result["b"]["slope"] == pytest.approx(-.1)
    assert trend.site_trends(frame.iloc[:0], "value") == {}


def test_gray_integration_and_rendering_have_no_pooled_rank(tmp_path):
    weather, visibility = full_winter()
    daily, _ = trend.daily_table(weather, visibility, "a", start_year=2000, end_year=2000)
    daily = pd.concat([daily, daily.assign(site="b")], ignore_index=True)
    daily.attrs["n_boot"] = 2
    gray = trend.gray_replication(daily)
    assert all(set(trends) == {"a", "b"} for trends in gray["trends"].values())
    result = {"coverage": [], "nox_availability": {}, "gray_replication": gray,
              "diagnostics": {"logistic_regression": {"winters": [], "folds": []}}}
    trend.write_outputs(result, tmp_path)
    saved = json.loads((tmp_path / "long_term_trend_v2.json").read_text())
    text = (tmp_path / "long_term_trend_v2.md").read_text(encoding="utf-8")
    assert "pooled_site_demeaned" not in json.dumps(saved)
    assert "Trends are reported separately by airport; no regional pooled rank test is reported." in text
    assert "≤1-SM extension (not a replication of Gray's dense-fog analysis)" in text
    assert "They exclude first-stage estimation uncertainty" in text
    assert saved["diagnostics"]["logistic_regression"]["first_stage_refit"] is False
    assert saved["diagnostics"]["logistic_regression"]["uncertainty_scope"] == "fixed_diagnostic_predictions"
    assert "uncertainty_scope" not in saved["gray_replication"]


def test_era5_missing_site_winter_or_channel_never_falls_back(tmp_path, monkeypatch):
    monkeypatch.setattr(trend, "load_sources", lambda *a, **k: pytest.fail("archive fallback"))
    with pytest.raises(FileNotFoundError):
        trend._weather("location_6", "era5", [2000], root=tmp_path)
    folder = tmp_path / "datasets" / "weather_era5_trend"
    folder.mkdir(parents=True)
    weather, _ = full_winter()
    path = folder / "location_6_weather_era5_trend.csv"
    weather.iloc[:-1].to_csv(path, index=False)
    with pytest.raises(ValueError, match="winter 2001"):
        trend._weather("location_6", "era5", [2000], root=tmp_path)
    weather.loc[0, trend.WEATHER_COLS[-1]] = np.nan
    weather.to_csv(path, index=False)
    with pytest.raises(ValueError, match="nonfinite"):
        trend._weather("location_6", "era5", [2000], root=tmp_path)


def test_era5_provenance_and_weather_change_do_not_change_labels(tmp_path):
    folder = tmp_path / "datasets" / "weather_era5_trend"
    folder.mkdir(parents=True)
    weather, visibility = full_winter()
    weather["temperature_2m"] = 5
    path = folder / "location_6_weather_era5_trend.csv"
    weather.to_csv(path, index=False)
    era5, provenance = trend._weather("location_6", "era5", [2000], root=tmp_path)
    assert provenance["source"] == "era5" and provenance["request_winters"] == [2001]
    assert len(provenance["sha256"]) == 64
    other = weather.assign(temperature_2m=4)
    a, _ = trend.daily_table(era5, visibility, "location_6", start_year=2000, end_year=2000)
    b, _ = trend.daily_table(other, visibility, "location_6", start_year=2000, end_year=2000)
    pd.testing.assert_series_equal(a.fog_day, b.fog_day)
    assert not a.optimal.any() and b.optimal.all()


def test_write_valid_winters_uses_labels_only_and_january_year(tmp_path, monkeypatch):
    _, visibility = full_winter()
    monkeypatch.setattr(trend, "SITES", {"location_6": "Madera"})
    monkeypatch.setattr(trend, "_visibility", lambda *a: (visibility, None))
    monkeypatch.setattr(trend, "_weather", lambda *a, **k: pytest.fail("weather loaded"))
    monkeypatch.setattr(trend, "capture_environment", lambda: pytest.fail("environment not needed for label export"))
    monkeypatch.setattr(trend, "diagnostic_predictions", lambda *a, **k: pytest.fail("fit attempted"))
    path = tmp_path / "winters.csv"
    trend.main(["--start-year", "2000", "--end-year", "2000", "--write-valid-winters", str(path)])
    assert pd.read_csv(path).to_dict("records") == [{"site": "location_6", "winter": 2001}]


def test_inference_only_uses_saved_panels_without_loading_or_fitting(tmp_path, monkeypatch):
    panel = synthetic_panel()
    panel["observed_all_days"] = panel["observed"] = 3
    panel["expected"], panel["n_days"] = 2.0, 100
    result = {"seed": 11, "coverage": [{"unchanged": True}], "nox_availability": {},
              "gray_replication": {"winters": [], "trends": {}},
              "diagnostics": {"logistic_regression": {"winters": panel.to_dict("records"), "folds": [{"unchanged": True}],
                                "attribution": {"status": "not_available"}}}}
    (tmp_path / "long_term_trend_v2.json").write_text(json.dumps(result))
    monkeypatch.setattr(trend, "capture_environment", lambda: {"sentinel": "analysis"})
    monkeypatch.setattr(trend, "_visibility", lambda *a: pytest.fail("labels loaded"))
    monkeypatch.setattr(trend, "_weather", lambda *a, **k: pytest.fail("weather loaded"))
    monkeypatch.setattr(trend, "diagnostic_predictions", lambda *a, **k: pytest.fail("fit attempted"))
    saved = {}
    monkeypatch.setattr(trend, "write_outputs", lambda result, out: saved.update(result))
    trend.main(["--inference-only", "--out-dir", str(tmp_path), "--n-boot", "2"])
    assert saved["diagnostics"]["logistic_regression"]["winters"] == result["diagnostics"]["logistic_regression"]["winters"]
    assert saved["diagnostics"]["logistic_regression"]["folds"] == [{"unchanged": True}]
    assert saved["coverage"] == [{"unchanged": True}]
    assert saved["analysis_environment"] == {"sentinel": "analysis"}
    assert saved["diagnostics"]["logistic_regression"]["panel"]["coefficients"]["year"]["estimate"] == pytest.approx(.012)


def test_nondefault_visibility_requires_isolated_output_and_missing_sidecar_fails(tmp_path):
    with pytest.raises(SystemExit) as exc:
        trend.main(["--visibility-mode", "routine_only"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit):
        trend.main(["--visibility-mode", "one_per_hour", "--out-dir", str(trend.OUT_DIR)])
    with pytest.raises(FileNotFoundError):
        trend._visibility("location_6", "routine_only", root=tmp_path)


def test_cadence_run_refits_endpoints_on_identical_paired_support(tmp_path, monkeypatch):
    weather, minimum = full_winter()
    selected = minimum.copy()
    minimum.loc[[0, 24], "category"] = 4
    selected.loc[0, "category"] = np.nan
    selected["report_type"], selected["obs_minute"] = "FM15", 55
    monkeypatch.setattr(trend, "SITES", {"location_6": "Madera"})
    monkeypatch.setattr(trend, "_visibility", lambda site, mode: (minimum if mode == "minimum" else selected, None))
    monkeypatch.setattr(trend, "_weather", lambda *a, **k: (weather, {"source": "era5"}))
    monkeypatch.setattr(trend, "measured_nox", lambda site: (pd.DataFrame(columns=["site", "winter", "nox_ppb"]), {"status": "not_available"}))
    monkeypatch.setattr(trend, "capture_environment", lambda: {"sentinel": True})
    calls = []
    def predictions(daily, X, family, **kwargs):
        calls.append((family, daily.copy()))
        return np.full(len(daily), .2 if daily.fog_day.sum() else .3), [{"status": "ok"}]
    monkeypatch.setattr(trend, "diagnostic_predictions", predictions)
    saved = {}
    monkeypatch.setattr(trend, "write_outputs", lambda result, out: saved.update(result))
    trend.main(["--start-year", "2000", "--end-year", "2000", "--visibility-mode", "routine_only",
                "--out-dir", str(tmp_path), "--n-boot", "2"])
    assert len(calls) == 6
    for (_, a), (_, b) in zip(calls[2:4], calls[4:6]):
        pd.testing.assert_frame_equal(a[["site", "date", "winter", "valid", "known_hours"]],
                                      b[["site", "date", "winter", "valid", "known_hours"]])
        assert a.fog_day.sum() == 1 and b.fog_day.sum() == 0
    paired = saved["cadence_sensitivity"]["paired_control"]
    for family in ("logistic_regression", "ebm"):
        a, b = paired["minimum"]["diagnostics"][family], paired["selected"]["diagnostics"][family]
        assert a["folds"] == b["folds"]
        assert a["winters"][0]["n_days"] == b["winters"][0]["n_days"]
        assert a["winters"][0]["expected"] != b["winters"][0]["expected"]
    assert saved["visibility_mode"] == "routine_only"
    assert saved["analysis_environment"] == {"sentinel": True}
    audit = saved["cadence_sensitivity"]["observation_process"]["location_6"]
    assert audit["site_year_distributions"][0]["report_type"] == {"FM15": 1463}


def test_attribution_scales_supported_pairs_interval_and_status(monkeypatch):
    monkeypatch.setattr(trend, "winter_bootstrap_indices", lambda years, rng: np.array([1, 2, 3, 4, 5, 5]))
    fit = trend.fit_panel(_sparse_panel(), ("nox_ppb",), steps={}, n_boot=2)
    fit["coefficients"]["year"] = fit["coefficients"]["nox_ppb"]
    monkeypatch.setattr(trend, "fit_panel", lambda *a, **k: {**fit})
    panel = _sparse_panel().assign(dpd_c=3, fog_day_residual=1)
    nox = panel[["site", "winter", "nox_ppb"]]
    result = trend.attribution(panel.drop(columns="nox_ppb"), nox, n_boot=2)
    for comparison in result["gray_comparison"].values():
        coef = comparison["coefficients"]["nox_ppb"]
        scaled = comparison["fog_days_per_10ppb"]
        assert scaled["estimate"] == 10 * coef["estimate"]
        assert scaled["inference_status"] == coef["inference_status"]
        assert scaled["inference_reason"] == "unit_leverage"
        assert scaled["ci95"] is None
        assert scaled["pairs_winter"]["ci95"] == pytest.approx([10 * x for x in coef["pairs_winter"]["ci95"]])
        assert scaled["pairs_winter"]["valid_draws"] == coef["pairs_winter"]["valid_draws"]


def test_null_primary_interval_is_reported_with_exploratory_secondary():
    coefficient = {"estimate": 2, "ci95": None, "p": None,
                   "pairs_winter": {"ci95": [1, 3], "p": .2}}
    fit = {"method": "dependent_wild_bootstrap", "block_length": 2,
           "coefficients": {"nox_ppb": coefficient}}
    text = "\n".join(trend._coefficient_lines(fit, "near_saturation"))
    assert "nox_ppb: 2" in text
    assert "exploratory; assumes independent winters" in text
    coefficient["pairs_winter"]["ci95"] = None
    assert "uncertainty unavailable" in "\n".join(trend._coefficient_lines(fit))


def test_archive_weather_dispatch_is_explicit(monkeypatch):
    weather, visibility = full_winter()
    calls = []
    def load(site):
        calls.append(site)
        return weather, visibility
    monkeypatch.setattr(trend, "load_sources", load)
    actual, provenance = trend._weather("location_6", "archive_best_match", [2000])
    assert actual is weather
    assert calls == ["location_6"]
    assert provenance["source"] == "archive_best_match"


def test_cadence_differences_include_panel_attribution_and_share_trends():
    def endpoint(value):
        return {"diagnostics": {"lr": {
            "panel": {"coefficients": {"year": {"estimate": value, "ci95": None}}},
            "attribution": {"models": {"nox_only": {"coefficients": {"nox_ppb": {"estimate": value}}}}}}},
            "gray_replication": {"trends": {"fog_share_of_optimal": {"a": {"slope": value}}}}}
    result = trend._cadence_differences({"minimum": endpoint(1), "selected": endpoint(3)})
    assert result["diagnostics"]["lr"]["panel"]["coefficients"]["year"]["delta_estimate"] == 2
    assert result["diagnostics"]["lr"]["attribution"]["models"]["nox_only"]["coefficients"]["nox_ppb"]["delta_estimate"] == 2
    assert result["gray_replication"]["trends"]["fog_share_of_optimal"]["a"]["delta_estimate"] == 2
