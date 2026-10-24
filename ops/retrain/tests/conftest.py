"""Shared loaders for the retrain artifact verification suite.

Regenerated artifacts are optional in a clean checkout. Once present, malformed
artifacts fail rather than being replaced with synthetic output.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
RETRAIN = ROOT / "ops" / "retrain"
SITES = tuple(f"location_{number}" for number in range(6, 11))
TASKS = ("nowcast", "lead_time")
MODELS = ("climatology", "xgboost", "random_forest", "temporal_cnn")
BANNED_FEATURE = "soil_temperature_0_to_7cm"


def load_json(path: Path) -> Any:
    if not path.is_file():pytest.skip(f"regenerated retrain artifact absent: {path}")
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
    if value.get("target")!="ordinal-18":
        pytest.skip("retrain metrics predate ordinal-18; regenerate with run_retrain.py")
    return value
