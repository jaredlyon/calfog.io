"""Scoring joins by prediction identity rather than accidental row order."""
import pandas as pd
import pytest

from v2.shadow.shadow_score import join_and_score


def test_scorer_join_produces_correct_average_precision():
    # Positives rank first and third: AP = (1/1 + 2/3) / 2 = 5/6.
    predictions = pd.DataFrame(
        {
            "site": ["location_8"] * 4,
            "task": ["nowcast"] * 4,
            "model": ["xgboost"] * 4,
            "issue_time_utc": [
                "2026-01-01T01:00:00Z",
                "2026-01-01T02:00:00Z",
                "2026-01-01T03:00:00Z",
                "2026-01-01T04:00:00Z",
            ],
            "prob": [0.9, 0.8, 0.7, 0.1],
        }
    )
    # Deliberately shuffled, plus an outcome with no prediction.
    outcomes = pd.DataFrame(
        {
            "site": ["location_8"] * 5,
            "task": ["nowcast"] * 5,
            "model": ["xgboost"] * 5,
            "issue_time_utc": [
                "2026-01-01T04:00:00Z",
                "2026-01-01T02:00:00Z",
                "2026-01-01T01:00:00Z",
                "2026-01-01T05:00:00Z",
                "2026-01-01T03:00:00Z",
            ],
            "label": [0, 0, 1, 1, 1],
        }
    )

    result = join_and_score(predictions, outcomes)

    assert result["live_ap"] == pytest.approx(5 / 6)
