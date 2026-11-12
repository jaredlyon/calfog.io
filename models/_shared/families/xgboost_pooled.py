"""Pooled airport XGBoost with causal own- and neighbor-visibility lags."""

import numpy as np

from ..data import SITES
from ..windows import tree_feature_names, tree_view
from . import FamilyOutput
from .xgboost import fit_model

NEEDS_CNN_CFG = False
POOLED = True
VISIBILITY_FEATURES = [
    "lag_category", "nbr_min_cat", "nbr_mean_cat", "nbr_fog_count", "nbr_reporting",
]


def _features(X, meta):
    sites = meta.site.to_numpy()[:, None] == np.asarray(tuple(SITES))[None, :]
    visibility = meta[VISIBILITY_FEATURES].to_numpy(dtype=np.float32, na_value=np.nan)
    return np.concatenate([tree_view(X), sites, visibility], axis=1)


def fit_predict(prep, *, cnn_cfg: dict | None, seed: int) -> FamilyOutput:
    Xtr, Xv, Xt = (
        _features(X, prep.metas[split])
        for X, split in ((prep.Xtr, "train"), (prep.Xv, "val"), (prep.Xt, "test"))
    )
    model = fit_model(Xtr, prep.ctr, Xv, prep.cv, seed=seed)
    names = tree_feature_names(prep.cols) + ["site_" + s for s in SITES] + VISIBILITY_FEATURES
    importance = dict(zip(names, (100 * model.feature_importances_).tolist()))
    return FamilyOutput(
        model.predict_proba(Xv), model.predict_proba(Xt), importance,
        {"best_iteration": model.best_iteration},
    )
