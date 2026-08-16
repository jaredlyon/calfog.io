"""The serving log must never claim features from after issue time."""
import pytest

from v2.shadow.shadow_common import build_log_record


def _record(feature_end_time_utc):
    return build_log_record(
        site="location_6",
        task="nowcast",
        issue_time_utc="2026-01-10T12:00:00Z",
        target_window_utc={
            "start": "2026-01-10T12:00:00Z",
            "end": "2026-01-10T12:00:00Z",
        },
        model="xgboost",
        prob=0.25,
        feed_source="historical_forecast_best_match",
        feature_end_time_utc=feature_end_time_utc,
        costgrid_decisions={"1": False, "5": False},
        created_utc="2026-01-10T12:00:01Z",
    )


def test_latency_guard_accepts_feature_window_ending_at_issue_time():
    record = _record("2026-01-10T12:00:00Z")
    assert record["feature_end_time_utc"] == record["issue_time_utc"]


def test_latency_guard_rejects_post_issue_features():
    with pytest.raises((AssertionError, ValueError), match=r"feature.*issue|issue.*feature"):
        _record("2026-01-10T12:00:01Z")
