"""Fast, synthetic contracts for the full five-airport evaluation.

These tests deliberately do not load the source CSVs or fit any model.  They
exercise the shared leakage-safe pipeline and the model-independent full-run
evaluation helpers.
"""

import numpy as np
import pandas as pd

from models._shared.evaluation import (block_bootstrap, pmf_to_fog, rps, nll, median_category,
                                       ordinal_summary, ordinal_bootstrap, paired_rps_delta_bootstrap)
from models._shared.splits import matched_cohort, rolling_origin_folds
from models._shared.labels import audit_causal_features
from models._shared.data import utc_to_pst_naive
from models._shared.windows import Prepared
from models._shared.visibility import N_CATEGORIES


def test_ci_ordering():
    """A whole-day bootstrap must return CIs containing its point estimates."""
    # Four observations per day make it possible to catch an accidental IID-row
    # implementation while keeping this test very small and deterministic.
    issue_time = pd.date_range("2024-01-01", periods=12 * 4, freq="6h")
    y = np.tile([0, 0, 1, 0], 12)
    score = np.random.default_rng(4).random(len(y))
    meta = pd.DataFrame({"issue_time": issue_time, "fold": np.repeat([1, 2, 3], 16)})

    result = block_bootstrap(y, score, meta, n_boot=128, seed=17, block="day")

    assert result["ap_ci"][0] <= result["ap"] <= result["ap_ci"][1]
    assert result["roc_auc_ci"][0] <= result["roc_auc"] <= result["roc_auc_ci"][1]
    assert result["bootstrap_unit"] == "day"
    assert result["valid_ap_resamples"] > 0
    # Independently repeat complete days physically; do not trust the unit label.
    from sklearn.metrics import average_precision_score, roc_auc_score
    rng = np.random.default_rng(17)
    weights = [rng.multinomial(4, np.full(4, .25), size=128) for _ in range(3)]
    ap_draws, auc_draws = [], []
    for draw in range(128):
        rows = np.concatenate([np.repeat(np.arange(f*16,(f+1)*16),np.repeat(weights[f][draw],4)) for f in range(3)])
        ap_draws.append(average_precision_score(y[rows],score[rows]))
        auc_draws.append(roc_auc_score(y[rows],score[rows]))
    for name, values in (("ap",ap_draws),("roc_auc",auc_draws)):
        radius = np.quantile(np.abs(np.asarray(values)-result[name]),.95)
        np.testing.assert_allclose(result[name+"_ci"],[max(0.,result[name]-radius),min(1.,result[name]+radius)],rtol=0,atol=1e-12)



def test_rolling_origin_purge():
    """Expanding folds are ordered, nested, disjoint, and purged by 24 hours."""
    time_col = "issue_time"
    frame = pd.DataFrame(
        {
            time_col: pd.date_range("2024-01-01", periods=120 * 24, freq="h"),
            "row_id": np.arange(120 * 24),
        }
    )
    folds = rolling_origin_folds(frame, time_col=time_col, purge="24h")

    assert len(folds) >= 3
    previous_train = set()
    test_timestamps = []
    for fold in folds:
        train, validation, test = fold["train"], fold["val"], fold["test"]
        assert not train.empty and not validation.empty and not test.empty
        assert train[time_col].is_monotonic_increasing
        assert validation[time_col].is_monotonic_increasing
        assert test[time_col].is_monotonic_increasing

        train_times = set(train[time_col])
        validation_times = set(validation[time_col])
        current_test_times = set(test[time_col])
        assert previous_train <= train_times  # expanding rather than moving window
        previous_train = train_times
        assert train_times.isdisjoint(validation_times)
        assert train_times.isdisjoint(current_test_times)
        assert validation_times.isdisjoint(current_test_times)

        # No training observation is within the prohibited 24 hours before any
        # validation/test issue; likewise the validation/test boundary is purged.
        purge = pd.Timedelta("24h")
        assert validation[time_col].min() - train[time_col].max() >= purge
        assert test[time_col].min() - validation[time_col].max() >= purge
        for boundary in (fold["validation_start"], fold["test_start"]):
            forbidden = frame.loc[
                frame[time_col].between(boundary - purge, boundary, inclusive="left"),
                time_col,
            ]
            assigned_before_boundary = pd.concat([train, validation]).loc[
                lambda x: x[time_col] < boundary, time_col
            ]
            assert set(forbidden).isdisjoint(set(assigned_before_boundary))

        test_timestamps.extend(test[time_col].tolist())

    assert len(test_timestamps) == len(set(test_timestamps))


def test_matched_cohort_identity():
    """AQI removal changes columns, never the per-site observational units."""
    keys = pd.MultiIndex.from_product(
        [["location_6", "location_7"], pd.date_range("2024-02-01", periods=4, freq="h")],
        names=["site", "issue_time"],
    ).to_frame(index=False)
    with_aqi = keys.assign(temperature=np.arange(8), pm2_5=np.arange(8) + 10)
    # Deliberately reverse and omit different edge observations.  The helper
    # must select the intersection and put both arms in exactly the same order.
    without_aqi = keys.iloc[1:-1].iloc[::-1].assign(temperature=np.arange(6))

    arms = matched_cohort(with_aqi, without_aqi)
    key_cols = ["site", "issue_time"]
    pd.testing.assert_frame_equal(
        arms.with_aqi[key_cols].reset_index(drop=True),
        arms.without_aqi[key_cols].reset_index(drop=True),
    )
    for site in arms.with_aqi["site"].unique():
        left = arms.with_aqi.loc[arms.with_aqi["site"] == site, key_cols]
        right = arms.without_aqi.loc[arms.without_aqi["site"] == site, key_cols]
        pd.testing.assert_frame_equal(left.reset_index(drop=True), right.reset_index(drop=True))
    assert "pm2_5" in arms.with_aqi and "pm2_5" not in arms.without_aqi
    pd.testing.assert_frame_equal(arms.with_aqi[key_cols].reset_index(drop=True), keys.iloc[1:-1].reset_index(drop=True))


def test_causal_features():
    """At an 18:00 lead issue, all usable features are available by 18:00."""
    issue = pd.to_datetime(["2024-03-01 18:00", "2024-03-02 18:00"])
    frame = pd.DataFrame(
        {
            "issue_time": issue,
            # Exact-cutoff availability is causal; earlier observations are too.
            "weather_available_at": issue - pd.to_timedelta([0, 1], unit="h"),
            "aqi_available_at": issue - pd.to_timedelta([2, 3], unit="h"),
            # The label is in the next morning and must not become the cutoff.
            "verification_end": issue.normalize() + pd.Timedelta(days=1, hours=10),
        }
    )
    audit = audit_causal_features(
        frame,
        {"weather": "weather_available_at", "aqi": "aqi_available_at"},
        task="lead",
        label_time_col="verification_end",
    )
    assert audit.ok

    future = frame.copy()
    future.loc[1, "aqi_available_at"] = future.loc[1, "issue_time"] + pd.Timedelta(minutes=1)
    violation = audit_causal_features(
        future,
        {"weather": "weather_available_at", "aqi": "aqi_available_at"},
        task="lead",
        label_time_col="verification_end",
    )
    assert not violation.ok
    assert violation.violations[["row_index", "feature"]].to_dict("records") == [
        {"row_index": 1, "feature": "aqi"}
    ]


def test_noaa_utc_to_fixed_pst_alignment():
    utc=pd.Series(pd.to_datetime(["2024-01-15 14:00Z","2024-07-15 14:00Z"]))
    local=utc_to_pst_naive(utc)
    assert local.tolist()==[pd.Timestamp("2024-01-15 06:00"),pd.Timestamp("2024-07-15 06:00")]
    assert local.dt.tz is None


def test_hand_computed_ordinal_scores_and_censoring():
    pmf = np.zeros((2, N_CATEGORIES))
    pmf[0, 1:3] = [.75, .25]
    pmf[1, [4, 5]] = [.25, .75]
    categories = np.array([1, 5])
    np.testing.assert_allclose(pmf_to_fog(pmf), [1., .25])
    np.testing.assert_allclose(rps(categories, pmf), .0625 / 17)
    np.testing.assert_allclose(nll(categories, pmf), -np.log(.75))
    np.testing.assert_array_equal(median_category(pmf), [1, 5])
    summary = ordinal_summary(categories, pmf)
    assert summary['within_one'] == 1.
    assert summary['median_category_mae'] == 0.
    assert summary['median_meters_mae_log10'] == 0.
    assert summary['median_meters_n'] == 2
    edge = np.eye(N_CATEGORIES)[[0, 17, 0, 4]]
    summary = ordinal_summary([0, 17, 1, 5], edge)
    assert summary['median_meters_n'] == 1
    np.testing.assert_allclose(summary['median_meters_mae_log10'], np.log10(1.25))
    empty = ordinal_summary([1, 17], edge[:2])
    assert empty['median_meters_n'] == 0 and empty['median_meters_mae_log10'] is None
    np.testing.assert_allclose(nll([17], pmf[:1]), -np.log(1e-9))
    tie = np.zeros((1, N_CATEGORIES)); tie[0, [2, 8]] = .5
    assert median_category(tie).tolist() == [2]


def test_prepared_binary_properties_derive_from_categories():
    categories = np.array([0, 4, 5, 17])
    X = np.zeros((4, 24, 1), dtype=np.float32)
    prep = Prepared(X, categories, X, categories.copy(), X, categories.copy(), {}, ['x'])
    for name in ('ytr', 'yv', 'yt'):
        np.testing.assert_array_equal(getattr(prep, name), [1, 1, 0, 0])
    prep.ctr[1] = 5
    np.testing.assert_array_equal(prep.ytr, [1, 0, 0, 0])
    import pytest
    with pytest.raises(AttributeError):
        prep.ytr = np.zeros(4)


def test_ordinal_bootstrap_physically_repeats_paired_days():
    categories = np.array([1, 5, 4, 8, 2, 6, 3, 17])
    pmf = .5*np.eye(N_CATEGORIES)[categories] + .5/N_CATEGORIES
    other = np.full_like(pmf, 1/N_CATEGORIES)
    meta = pd.DataFrame({'issue_time': pd.date_range('2024-01-01', periods=8, freq='12h'),
                         'fold': [1]*4+[2]*4})
    summary = ordinal_bootstrap(categories, pmf, meta, n_boot=32, seed=17)
    point, bounds, valid = paired_rps_delta_bootstrap(categories, pmf, other, meta, n_boot=32, seed=17)
    rng = np.random.default_rng(17)
    weights = [rng.multinomial(2, [.5, .5], size=32) for _ in range(2)]
    scores, deltas = [], []
    for draw in range(32):
        rows = np.concatenate([np.repeat(np.arange(f*4, (f+1)*4), np.repeat(weights[f][draw], 2)) for f in range(2)])
        scores.append(rps(categories[rows], pmf[rows]))
        deltas.append(scores[-1]-rps(categories[rows], other[rows]))
    radius = np.quantile(np.abs(np.asarray(scores)-summary['rps']), .95)
    np.testing.assert_allclose(summary['rps_ci'], [max(0., summary['rps']-radius), min(1., summary['rps']+radius)])
    radius = np.quantile(np.abs(np.asarray(deltas)-point), .95)
    np.testing.assert_allclose(bounds, [max(-1., point-radius), min(1., point+radius)])
    assert valid == 32 and point < 0


def test_climatology_predicts_training_category_frequencies():
    from models._shared.families import climatology
    X = np.zeros((4, 24, 1), dtype=np.float32)
    prep = Prepared(X, np.array([2, 2, 5, 17]), X[:2], np.array([0, 0]),
                    X[:1], np.array([0]), {}, ['x'])
    output = climatology.fit_predict(prep, cnn_cfg=None, seed=17)
    expected = np.zeros(N_CATEGORIES); expected[[2, 5, 17]] = [.5, .25, .25]
    np.testing.assert_allclose(output.pmf_val, np.tile(expected, (2, 1)))
    np.testing.assert_allclose(output.pmf_test, expected[None])
