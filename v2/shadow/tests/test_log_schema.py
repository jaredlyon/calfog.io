"""Prediction records are bounded, audit-friendly, and append-only JSONL."""
import json

import pytest

from v2.shadow.shadow_common import LOG_FIELDS, build_log_record
from v2.shadow import shadow_predict


def _record(prob=0.37, *, model="temporal_cnn"):
    return build_log_record(
        site="location_7",
        task="lead_time",
        issue_time_utc="2026-02-04T02:00:00Z",
        target_window_utc={
            "start": "2026-02-04T08:00:00Z",
            "end": "2026-02-04T17:00:00Z",
        },
        model=model,
        prob=prob,
        feed_source="historical_forecast_best_match",
        feature_end_time_utc="2026-02-04T02:00:00Z",
        costgrid_decisions={"1": True, "3": False, "5": False},
        created_utc="2026-02-04T02:00:03Z",
    )


def test_log_schema_and_jsonl_append(tmp_path, monkeypatch):
    # Keep both JSONL and SQLite writes isolated from the real private log.
    monkeypatch.setattr(shadow_predict, "SHADOW", tmp_path)
    path = tmp_path / "predictions.jsonl"
    first = _record()
    second = _record(0.82, model="xgboost")

    shadow_predict.append_prediction(first)
    shadow_predict.append_prediction(second)
    rows = [json.loads(line) for line in path.read_text().splitlines()]

    assert rows == [first, second]  # append, do not silently overwrite
    for row in rows:
        assert set(LOG_FIELDS) <= set(row)
        assert 0.0 <= row["prob"] <= 1.0
        assert row["site"].startswith("location_")
        assert row["task"] in {"nowcast", "lead_time"}
        assert row["target_window_utc"]["start"] <= row["target_window_utc"]["end"]
        assert row["feature_end_time_utc"] <= row["issue_time_utc"]
        assert isinstance(row["costgrid_decisions"], dict)
        assert row["created_utc"].endswith("Z")


@pytest.mark.parametrize("bad_probability", [-0.001, 1.001, float("nan")])
def test_log_schema_rejects_invalid_probability(bad_probability):
    with pytest.raises(ValueError, match="prob"):
        _record(bad_probability)
