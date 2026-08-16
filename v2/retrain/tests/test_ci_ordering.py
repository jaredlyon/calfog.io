"""Metric shape and confidence-interval checks over metrics_retrain.json."""
from __future__ import annotations

import math

from conftest import MODELS, SITES, TASKS


def _finite_number(value, *, where: str) -> float:
    assert isinstance(value, (int, float)) and not isinstance(value, bool), (
        f"{where} must be numeric"
    )
    assert math.isfinite(value), f"{where} must be finite"
    return float(value)


def test_ci_ordering(metrics):
    """All 5 x 2 x 4 final AP intervals are ordered and contain their AP."""
    per_site = metrics.get("per_site")
    assert isinstance(per_site, dict), "metrics_retrain.json: missing per_site object"

    for site in SITES:
        assert site in per_site, f"metrics_retrain.json: missing {site}"
        site_metrics = per_site[site]
        assert isinstance(site_metrics, dict), f"{site}: metrics must be an object"
        tasks = site_metrics.get("tasks")
        assert isinstance(tasks, dict), f"{site}: missing tasks object"
        for task in TASKS:
            assert task in tasks, f"{site}: missing task {task}"
            task_metrics = tasks[task]
            assert isinstance(task_metrics, dict), f"{site}.{task}: must be an object"
            for model in MODELS:
                assert model in task_metrics, f"{site}.{task}: missing model {model}"
                result = task_metrics[model]
                assert isinstance(result, dict), f"{site}.{task}.{model}: must be an object"
                prefix = f"{site}.{task}.{model}"

                ap = _finite_number(result.get("ap"), where=f"{prefix}.ap")
                assert 0.0 <= ap <= 1.0, f"{prefix}.ap is outside [0, 1]"
                interval = result.get("ap_ci")
                assert isinstance(interval, list) and len(interval) == 2, (
                    f"{prefix}.ap_ci must be a two-element JSON array"
                )
                low = _finite_number(interval[0], where=f"{prefix}.ap_ci[0]")
                high = _finite_number(interval[1], where=f"{prefix}.ap_ci[1]")
                assert 0.0 <= low <= ap <= high <= 1.0, (
                    f"{prefix}: expected 0 <= CI low <= AP <= CI high <= 1; "
                    f"got {interval} around {ap}"
                )

                for metric in ("roc_auc", "brier", "ece"):
                    value = _finite_number(result.get(metric), where=f"{prefix}.{metric}")
                    assert 0.0 <= value <= 1.0, f"{prefix}.{metric} is outside [0, 1]"
                n_test = result.get("n_test")
                pos_test = result.get("pos_test")
                assert isinstance(n_test, int) and not isinstance(n_test, bool) and n_test > 0, (
                    f"{prefix}.n_test must be a positive integer"
                )
                assert isinstance(pos_test, int) and not isinstance(pos_test, bool), (
                    f"{prefix}.pos_test must be an integer"
                )
                assert 0 <= pos_test <= n_test, f"{prefix}: invalid test positive count"
