from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import numpy as np

@dataclass
class FamilyOutput:
    pmf_val: np.ndarray
    pmf_test: np.ndarray
    importance: dict[str, float] = field(default_factory=dict)
    info: dict = field(default_factory=dict)
    state: object | None = None

def seed_for(*parts: object) -> int:
    return int(sha256('|'.join(map(str, parts)).encode()).hexdigest()[:8], 16)

from . import climatology, logistic_regression, random_forest, xgboost, xgboost_focal_loss, temporal_cnn, ensemble
from . import persistence, analog, xgboost_ordinal, xgboost_aft, ebm, xgboost_pooled

FAMILIES: dict[str, "module"] = {module.__name__.rsplit('.', 1)[-1]: module for module in
            (climatology, logistic_regression, random_forest, xgboost, xgboost_focal_loss, temporal_cnn, ensemble,
             persistence, analog, xgboost_ordinal, xgboost_aft, ebm, xgboost_pooled)}
