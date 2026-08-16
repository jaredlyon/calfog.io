"""Production/serving schema parity checks over the final JSON artifacts."""
from __future__ import annotations

from typing import Any

from conftest import BANNED_FEATURE, RETRAIN, SITES, feature_names, load_json


def _site_nodes(value: Any, site: str) -> list[Any]:
    found: list[Any] = []
    if isinstance(value, dict):
        if site in value:
            found.append(value[site])
        for child in value.values():
            found.extend(_site_nodes(child, site))
    elif isinstance(value, list):
        for child in value:
            if isinstance(child, dict) and child.get("site") == site:
                found.append(child)
            found.extend(_site_nodes(child, site))
    return found


def _has_positive_non_null_evidence(value: Any) -> bool:
    """Recognize count-, fraction-, or boolean-style non-null probe evidence."""
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = key.lower().replace("-", "_")
            if "non_null" in normalized or "nonnull" in normalized:
                if child is True or (
                    isinstance(child, (int, float))
                    and not isinstance(child, bool)
                    and child > 0
                ):
                    return True
            if _has_positive_non_null_evidence(child):
                return True
    elif isinstance(value, list):
        return any(_has_positive_non_null_evidence(child) for child in value)
    return False


def test_feature_parity(production_contract, approved_features):
    """Every real serving spec is an ordered subset of the live-approved set."""
    assert BANNED_FEATURE not in approved_features, (
        f"banned null live channel appears in production contract: {BANNED_FEATURE}"
    )
    approved_positions = {name: position for position, name in enumerate(approved_features)}

    for site in SITES:
        spec_path = RETRAIN / "serving" / site / "feature_spec.json"
        spec = load_json(spec_path)
        serving = feature_names(spec, where=str(spec_path))
        assert BANNED_FEATURE not in serving, f"{site}: banned channel appears in serving spec"

        unknown = [name for name in serving if name not in approved_positions]
        assert not unknown, f"{site}: serving features not approved by live probe: {unknown}"
        positions = [approved_positions[name] for name in serving]
        assert positions == sorted(positions), (
            f"{site}: serving feature order differs from production contract"
        )

        evidence = _site_nodes(production_contract, site)
        assert evidence, f"{site}: production contract has no per-site probe evidence"
        assert any(_has_positive_non_null_evidence(node) for node in evidence), (
            f"{site}: no positive non-null probe evidence in production contract"
        )
