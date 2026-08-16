"""Causal-window and chronological-purge checks on final serving/eval artifacts."""
from __future__ import annotations

from typing import Any

import pandas as pd

from conftest import RETRAIN, SITES, TASKS, load_json


def _find_values(value: Any, wanted: set[str]) -> list[Any]:
    values: list[Any] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key in wanted:
                values.append(child)
            values.extend(_find_values(child, wanted))
    elif isinstance(value, list):
        for child in value:
            values.extend(_find_values(child, wanted))
    return values


def _window_hours(spec: dict[str, Any]) -> Any:
    for key in ("window_hours", "lookback_hours", "sequence_length_hours"):
        if key in spec:
            return spec[key]
    window = spec.get("window")
    if isinstance(window, (int, float)):
        return window
    if isinstance(window, dict):
        for key in ("hours", "lookback_hours", "length_hours"):
            if key in window:
                return window[key]
    return None


def _timestamp(value: Any, *, where: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise AssertionError(f"{where} is not a timestamp: {value!r}") from exc
    assert not pd.isna(timestamp), f"{where} is missing"
    return timestamp


def test_causal_features(metrics):
    """Serving uses 24 trailing hours and labels/evaluation remain after cutoff."""
    per_site = metrics.get("per_site")
    assert isinstance(per_site, dict), "metrics_retrain.json: missing per_site object"
    purge = pd.Timedelta(hours=24)

    for site in SITES:
        spec_path = RETRAIN / "serving" / site / "feature_spec.json"
        metadata_path = RETRAIN / "serving" / site / "metadata.json"
        spec = load_json(spec_path)
        metadata = load_json(metadata_path)
        assert isinstance(spec, dict), f"{spec_path}: expected a JSON object"
        assert isinstance(metadata, dict), f"{metadata_path}: expected a JSON object"

        hours = _window_hours(spec)
        assert isinstance(hours, (int, float)) and not isinstance(hours, bool), (
            f"{site}: feature spec must state its trailing window length"
        )
        assert float(hours) == 24.0, f"{site}: serving window is {hours}, expected 24 hours"

        cutoff_claims = _find_values(
            metadata,
            {"all_feature_times_lte_issue", "features_end_at_or_before_issue"},
        )
        assert cutoff_claims and all(value is True for value in cutoff_claims), (
            f"{site}: metadata must prove all feature timestamps are <= issue time"
        )
        lead_claims = _find_values(
            metadata,
            {"lead_label_after_issue", "lead_target_after_issue"},
        )
        assert lead_claims and all(value is True for value in lead_claims), (
            f"{site}: metadata must state that the lead label window follows issue time"
        )

        site_metrics = per_site.get(site)
        assert isinstance(site_metrics, dict), f"metrics_retrain.json: missing {site}"
        audits = site_metrics.get("fold_audit")
        assert isinstance(audits, dict), f"{site}: missing fold_audit"
        for task in TASKS:
            folds = audits.get(task)
            assert isinstance(folds, list) and len(folds) == 3, (
                f"{site}.{task}: expected three real rolling-origin fold audits"
            )
            previous_train_end = None
            previous_test_start = None
            for index, fold in enumerate(folds, 1):
                assert isinstance(fold, dict), f"{site}.{task}.fold{index}: invalid audit"
                train_end = _timestamp(
                    fold.get("train_end"), where=f"{site}.{task}.fold{index}.train_end"
                )
                val_start = _timestamp(
                    fold.get("val_start"), where=f"{site}.{task}.fold{index}.val_start"
                )
                val_end = _timestamp(
                    fold.get("val_end"), where=f"{site}.{task}.fold{index}.val_end"
                )
                test_start = _timestamp(
                    fold.get("test_start"), where=f"{site}.{task}.fold{index}.test_start"
                )
                assert val_start - train_end >= purge, (
                    f"{site}.{task}.fold{index}: train/validation purge is under 24h"
                )
                assert test_start - val_end >= purge, (
                    f"{site}.{task}.fold{index}: validation/test purge is under 24h"
                )
                assert train_end < val_start <= val_end < test_start, (
                    f"{site}.{task}.fold{index}: fold timestamps are not causal"
                )
                if previous_train_end is not None:
                    assert train_end > previous_train_end, (
                        f"{site}.{task}: training origins do not expand forward"
                    )
                    assert test_start > previous_test_start, (
                        f"{site}.{task}: test folds are not ordered forward"
                    )
                previous_train_end = train_end
                previous_test_start = test_start
