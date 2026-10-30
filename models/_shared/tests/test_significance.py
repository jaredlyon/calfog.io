"""Synthetic two-site tests: no training, downloads, or external data."""
import json

import numpy as np
import pandas as pd
import pytest

from models._shared import significance as sig
from models._shared.evaluation import rps, safe_ap


def _predictions():
    rows_a, rows_b = [], []
    for site, hours, scores in (("small", (0, 1), (.9, .1)),
                                 ("large", tuple(range(6)), (.2, .8))):
        for day in range(4):
            for hour in hours:
                c = 0 if hour % 2 == 0 else 17
                p = scores[hour % 2]
                common = {"site": site, "task": "nowcast",
                          "issue_time": pd.Timestamp("2024-01-01") + pd.Timedelta(days=day, hours=hour),
                          "fold": 1 if site == "small" else 2, "c": c,
                          "dpd_c": 3., "wind_kmh": 6.}
                pa, pb = np.zeros(18), np.zeros(18)
                pa[[0, 17]], pb[[0, 17]] = [p, 1 - p], [.5, .5]
                rows_a.append({**common, "pmf": pa})
                rows_b.append({**common, "pmf": pb})
    return pd.DataFrame(rows_a), pd.DataFrame(rows_b)


def test_equal_airport_estimand_is_not_row_pooled():
    a, b = _predictions()
    result = sig.pooled_delta(a, b, "ap", 40, 3)
    expected = []
    for site in ("small", "large"):
        aa, bb = a[a.site == site], b[b.site == site]
        delta = safe_ap(aa.c <= 4, sig.pmf_to_fog(sig._pmf(aa))) - safe_ap(bb.c <= 4, sig.pmf_to_fog(sig._pmf(bb)))
        expected.append(delta)
        assert result["per_airport"][site]["estimate"] == pytest.approx(delta)
    assert result["estimate"] == pytest.approx(np.mean(expected))
    assert result["estimate"] != pytest.approx(np.average(expected, weights=[8, 24]))
    assert (result["n_rows"], result["n_days"], result["valid_draws"]) == (32, 4, 40)


def test_joint_calendar_day_counts_are_shared_across_airports(monkeypatch):
    a, b = _predictions()
    calls = []
    original = sig._weighted_boot_metrics
    def spy(y, p, codes, counts):
        calls.append((codes.copy(), counts.copy()))
        return original(y, p, codes, counts)
    monkeypatch.setattr(sig, "_weighted_boot_metrics", spy)
    sig.pooled_delta(a, b, "ap", 30, 31)
    assert len(calls) == 4  # each airport, each arm
    for codes, counts in calls:
        np.testing.assert_array_equal(counts, calls[0][1])
        np.testing.assert_array_equal(counts.sum(axis=1), np.full(30, 4))
        # Every hour of a date has one multiplicity; the same date has that
        # multiplicity at both sites, even though the fold and row counts differ.
        hours_per_day = len(codes) // 4
        np.testing.assert_array_equal(codes, np.repeat(np.arange(4), hours_per_day))
        np.testing.assert_array_equal(counts[:, codes[::hours_per_day]], calls[0][1])
    assert np.unique(calls[0][1], axis=0).shape[0] > 1


def test_exact_weighted_ap_matches_physical_joint_day_draws():
    a, b = _predictions()
    # Vary rankings by day so this also detects accidental row bootstrapping.
    for i in a.index:
        p = (.85 if a.at[i, "c"] == 0 else .15) if a.at[i, "issue_time"].day % 2 else .5
        a.at[i, "pmf"] = np.array([p] + [0.] * 16 + [1 - p])
    codes, counts = sig._joint_day_counts(a, 25, 4)
    draws = []
    for count in counts:
        deltas = []
        for site in sorted(a.site.unique()):
            ix = np.flatnonzero(a.site.to_numpy() == site)
            sampled = np.repeat(ix, count[codes[ix]])
            y = a.c.to_numpy()[sampled] <= 4
            deltas.append(safe_ap(y, sig.pmf_to_fog(sig._pmf(a.iloc[sampled])))
                          - safe_ap(y, sig.pmf_to_fog(sig._pmf(b.iloc[sampled]))))
        draws.append(np.mean(deltas))
    result = sig.pooled_delta(a, b, "ap", 25, 4)
    expected = sig._summary(result["estimate"], draws)
    np.testing.assert_allclose(result["ci95"], expected["ci95"], atol=1e-12)
    np.testing.assert_allclose(result["ci_bonf"], expected["ci_bonf"], atol=1e-12)
    assert result["p_two_sided"] == expected["p_two_sided"]


def test_holm_matches_hand_calculation():
    # Sorted p: .001,.01,.03,.04; multipliers: 4,3,2,1;
    # cumulative maxima: .004,.03,.06,.06, then restore input order.
    assert sig.holm_adjust([.03, .001, .04, .01]) == pytest.approx([.06, .004, .06, .03])
    assert sig.holm_adjust([.8, .9]) == [1., 1.]


def test_equivalence_strictly_flips_at_point_zero_two():
    assert sig.rules_out_gain_0_02({"ci_bonf": [-.1, np.nextafter(.02, 0)]})
    assert not sig.rules_out_gain_0_02({"ci_bonf": [-.1, .02]})
    assert not sig.rules_out_gain_0_02({"ci_bonf": [-.1, .021]})
    assert not sig.rules_out_gain_0_02({"ci_bonf": None})


def test_intervals_and_centered_two_sided_p_follow_study_convention():
    draws = np.array([-.4, -.1, .1, .2, .7])
    result = sig._summary(.1, draws)
    for key, level in (("ci95", .95), ("ci_bonf", 1 - .05 / 6)):
        radius = np.quantile(np.abs(draws - .1), level)
        np.testing.assert_allclose(result[key], [.1 - radius, .1 + radius])
    assert result["p_two_sided"] == pytest.approx(5 / 6)


def test_threshold_brier_mean_equals_rps_and_haze_ranges():
    rng = np.random.default_rng(7)
    pmf = rng.dirichlet(np.ones(18), 60)
    c = rng.integers(0, 18, 60)
    assert abs(sig.threshold_brier(c, pmf).mean() - rps(c, pmf)) < 1e-12
    a, b = _predictions()
    a["pmf"] = list(rng.dirichlet(np.ones(18), len(a)))
    b["pmf"] = list(rng.dirichlet(np.ones(18), len(b)))
    result = sig.haze_decomposition(a, b, 20, 9)
    thresholds = np.array([row["estimate"] for row in result["thresholds"]])
    assert len(thresholds) == 17
    assert thresholds.mean() == pytest.approx(sig.pooled_delta(a, b, "rps", 20, 9)["estimate"], abs=1e-12)
    assert result["fog_0_4"]["estimate"] == pytest.approx(thresholds[:5].mean())
    assert result["haze_5_15"]["estimate"] == pytest.approx(thresholds[5:16].mean())


def test_near_saturation_strict_thresholds_and_missing_values():
    frame = pd.DataFrame({"dpd_c": [4.19, 4.2, 4.1, 4.1, np.nan, 3., -1.],
                          "wind_kmh": [7.19, 1., 7.2, 7.21, 1., np.nan, 0.]})
    np.testing.assert_array_equal(sig.near_saturation(frame), [True, False, False, False, False, False, True])


def test_alignment_is_paired_and_mask_uses_input_a_rows():
    a, b = _predictions()
    original = sig.pooled_delta(a, b, "rps", 20, 5)
    shuffled = sig.pooled_delta(a, b.sample(frac=1, random_state=5), "rps", 20, 5)
    assert original == shuffled
    a.loc[:1, "dpd_c"] = 4.2
    filtered = sig.pooled_delta(a, b, "rps", 20, 5, sig.near_saturation)
    assert filtered["n_rows"] == len(a) - 2
    b.loc[0, "c"] = 1
    with pytest.raises(ValueError, match="disagree on c"):
        sig.pooled_delta(a, b, "ap", 20, 5)


def test_zero_positive_draws_do_not_unpair_airports():
    a, b = _predictions()
    for frame in (a, b):
        frame.loc[(frame.site == "small") & (frame.issue_time.dt.day > 1), "c"] = 17
    result = sig.pooled_delta(a, b, "ap", 80, 2)
    codes, counts = sig._joint_day_counts(a, 80, 2)
    assert 0 < result["valid_draws"] < 80
    assert result["valid_draws"] == np.count_nonzero(counts[:, 0])


def test_loading_saved_predictions_and_empty_arm(tmp_path, monkeypatch):
    monkeypatch.setattr(sig, "_family_dirs", lambda: {"example": tmp_path})
    directory = tmp_path / "example_predictions_v2"
    directory.mkdir()
    a, _ = _predictions()
    frame = a[a.site == "small"]
    np.savez_compressed(directory / "small__nowcast__main.npz",
                        issue_time=frame.issue_time.to_numpy(dtype="datetime64[ns]").astype("int64"),
                        fold=frame.fold.to_numpy(dtype="int8"), c=frame.c.to_numpy(dtype="int8"),
                        pmf=sig._pmf(frame),
                        dpd_c=frame.dpd_c.to_numpy(dtype="float32"), wind_kmh=frame.wind_kmh.to_numpy(dtype="float32"))
    loaded = sig.load_predictions("example", "main")
    assert loaded.site.unique().tolist() == ["small"]
    np.testing.assert_array_equal(loaded.issue_time, frame.issue_time)
    assert sig.pooled_delta(loaded, loaded, "ap", 10, 2)["estimate"] == 0
    assert sig.load_predictions("example", "missing").empty


def test_analysis_outputs_and_holm_families(tmp_path, monkeypatch):
    roster = {name: tmp_path for name in sig.PRIMARY_FAMILIES}
    monkeypatch.setattr(sig, "_family_dirs", lambda: roster)
    a, b = _predictions()
    def load(family, arm):
        frame = a if arm.endswith("with") or arm == "main" else b
        return pd.concat([frame, frame.assign(task="lead_time")], ignore_index=True)
    monkeypatch.setattr(sig, "load_predictions", load)
    monkeypatch.setattr(sig, "ROOT", tmp_path)
    sig.write_analysis(12, 3)
    directory = tmp_path / "models" / "comparison_v2"
    aqi = json.loads((directory / "aqi_tests_v2.json").read_text(encoding="utf-8"))
    family = json.loads((directory / "family_significance_v2.json").read_text(encoding="utf-8"))
    primary = [row for row in aqi["tests"] if row["primary"]]
    assert len(primary) == 6
    for rows in (primary, [row["near_saturation"] for row in primary]):
        expected = sig.holm_adjust([row["ap"]["p_two_sided"] for row in rows])
        assert [row["ap"]["p_holm"] for row in rows] == expected
    assert len(family["tests"]) == 12  # four comparators, six pairs, two tasks
    for task in sig.TASKS:
        rows = [row for row in family["tests"] if row["task"] == task]
        for metric in ("ap", "rps"):
            assert [row[metric]["p_holm"] for row in rows] == sig.holm_adjust([row[metric]["p_two_sided"] for row in rows])
    aqi_md = (directory / "aqi_tests_v2.md").read_text(encoding="utf-8")
    assert "5–15" in aqi_md and "α/6" in aqi_md and "near-saturation" in aqi_md
    assert "| Family A" in (directory / "family_significance_v2.md").read_text(encoding="utf-8")


def test_seed_sensitivity_keeps_replicate_zero_primary_and_lr_single_fit(tmp_path, monkeypatch):
    roster = {name: tmp_path / name for name in sig.PRIMARY_FAMILIES}
    for family in ("xgboost", "ebm"):
        for r in (1, 2):
            (roster[family] / f"{family}_predictions_v2" / "replicates" / f"r{r}").mkdir(parents=True)
    monkeypatch.setattr(sig, "_family_dirs", lambda: roster)
    a, b = _predictions()
    calls = []

    def load(family, arm, replicate=0):
        calls.append((family, replicate))
        frame = b
        if arm.endswith("with") and not arm.endswith("without") and replicate:
            # a replicate refit reorders the informative arm's scores (AP is rank-based)
            fog = np.clip(np.array([p[0] for p in a.pmf]) + np.random.default_rng(replicate).normal(0, .3, len(a)),
                          .01, .99)
            frame = a.assign(pmf=[np.r_[q, np.zeros(16), 1 - q] for q in fog])
        elif arm.endswith("with") and not arm.endswith("without"):
            frame = a
        return pd.concat([frame, frame.assign(task="lead_time")], ignore_index=True)

    monkeypatch.setattr(sig, "load_predictions", load)
    result = sig.seed_sensitivity(12, 3)
    assert result["replicate_ids"] == [0, 1, 2]
    rows = {(row["family"], row["task"], row["stratum"]): row for row in result["tests"]}
    lr = rows["logistic_regression", "nowcast", "all"]
    assert lr["training_seed"] == "deterministic fit"
    assert lr["seed_sd"] == 0 and lr["holm_agree"] == lr["rules_out_agree"] == 3
    assert {r for f, r in calls if f == "logistic_regression"} == {0}  # LR fit loaded once, never as a replicate
    xgb = rows["xgboost", "nowcast", "all"]
    a2, b2 = (load("xgboost", arm) for arm in ("nox_with", "nox_without"))
    r0 = sig.pooled_delta(a2[a2.task == "nowcast"], b2[b2.task == "nowcast"], "ap", 12, 3)["estimate"]
    assert xgb["estimate_r0"] == r0 and xgb["seed_min"] <= r0 <= xgb["seed_max"]
    assert xgb["seed_min"] < xgb["seed_max"] and xgb["seed_sd"] > 0 and xgb["replicates"] == 3
    (roster["ebm"] / "ebm_predictions_v2" / "replicates" / "r2").rmdir()
    with pytest.raises(ValueError, match="ebm: replicates"):
        sig.seed_sensitivity(12, 3)
