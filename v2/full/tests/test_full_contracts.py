"""Fast, synthetic contracts for the full five-airport evaluation.

These tests deliberately do not load the source CSVs or fit any model.  They
exercise the shared leakage-safe pipeline and the model-independent full-run
evaluation helpers.
"""

import numpy as np
import pandas as pd

from v2.full.full_utils import block_bootstrap, matched_cohort, rolling_origin_folds, utc_to_pacific_naive
from v2.pipeline import audit_causal_features


def test_ci_ordering():
    """A whole-day bootstrap must return CIs containing its point estimates."""
    # Four observations per day make it possible to catch an accidental IID-row
    # implementation while keeping this test very small and deterministic.
    issue_time = pd.date_range("2024-01-01", periods=12 * 4, freq="6h")
    y = np.tile([0, 0, 1, 0], 12)
    score = np.clip(0.12 + 0.70 * y + np.linspace(-0.08, 0.08, len(y)), 0, 1)
    meta = pd.DataFrame({"issue_time": issue_time, "fold": np.repeat([1, 2, 3], 16)})

    result = block_bootstrap(y, score, meta, n_boot=128, seed=17, block="day")

    assert result["ap_ci"][0] <= result["ap"] <= result["ap_ci"][1]
    assert result["roc_auc_ci"][0] <= result["roc_auc"] <= result["roc_auc_ci"][1]
    assert result["bootstrap_unit"] == "day"
    assert result["valid_ap_resamples"] > 0


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


def test_noaa_utc_to_pacific_alignment():
    utc=pd.Series(pd.to_datetime(["2024-01-15 14:00Z","2024-07-15 14:00Z"]))
    local=utc_to_pacific_naive(utc)
    assert local.tolist()==[pd.Timestamp("2024-01-15 06:00"),pd.Timestamp("2024-07-15 07:00")]
    assert local.dt.tz is None
