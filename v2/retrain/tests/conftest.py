"""Shared loaders for the retrain artifact verification suite.

These tests intentionally do not synthesize replacement outputs.  A missing or
malformed final artifact is a test failure, so a green run means the files that
would actually be served were inspected.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

RETRAIN = Path(__file__).resolve().parents[1]
SITES = tuple(f"location_{number}" for number in range(6, 11))
TASKS = ("nowcast", "lead_time")
MODELS = ("climatology", "xgboost", "random_forest", "temporal_cnn")
BANNED_FEATURE = "soil_temperature_0_to_7cm"


def load_json(path: Path) -> Any:
    assert path.is_file(), f"required final artifact does not exist: {path}"
    assert path.stat().st_size > 0, f"required final artifact is empty: {path}"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        pytest.fail(f"could not read valid JSON from {path}: {exc}")


def feature_names(spec: Any, *, where: str) -> list[str]:
    """Read a feature list while tolerating the usual descriptive key names."""
    if isinstance(spec, list):
        values = spec
    elif isinstance(spec, dict):
        values = None
        for key in ("features", "feature_names", "approved_features", "approved"):
            candidate = spec.get(key)
            if isinstance(candidate, list):
                values = candidate
                break
        assert values is not None, f"{where}: no feature list"
    else:
        pytest.fail(f"{where}: feature specification must be an object or list")
    assert values, f"{where}: feature list is empty"
    assert all(isinstance(value, str) and value.strip() for value in values), (
        f"{where}: every feature must be a non-empty string"
    )
    assert len(values) == len(set(values)), f"{where}: duplicate feature names"
    return values


@pytest.fixture(scope="session")
def production_contract() -> dict[str, Any]:
    value = load_json(RETRAIN / "production_feature_contract.json")
    assert isinstance(value, dict), "production_feature_contract.json must contain an object"
    return value


@pytest.fixture(scope="session")
def approved_features(production_contract: dict[str, Any]) -> list[str]:
    return feature_names(production_contract, where="production_feature_contract.json")


@pytest.fixture(scope="session")
def metrics() -> dict[str, Any]:
    value = load_json(RETRAIN / "metrics_retrain.json")
    assert isinstance(value, dict), "metrics_retrain.json must contain an object"
    return value
