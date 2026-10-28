"""Multiclass softmax focal XGBoost (gamma=1), stopped on validation NLL."""

import numpy as np
import xgboost as xgb
from scipy.special import logsumexp

from models._shared.families import FamilyOutput
from models._shared.families.xgboost import DEVICE, BoosterPMF
from models._shared.visibility import N_CATEGORIES
from models._shared.windows import Prepared, tree_feature_names, tree_view

NEEDS_CNN_CFG = False


def focal_loss_objective(preds, dtrain):
    """Exact gradient and floored diagonal Hessian of -(1-p_y) log(p_y)."""
    margins = np.asarray(preds, dtype=np.float64).reshape(-1, N_CATEGORIES)
    labels = dtrain.get_label().astype(int)
    rows = np.arange(len(labels))
    logp = margins - logsumexp(margins, axis=1, keepdims=True)
    probabilities = np.exp(logp)
    log_target = logp[rows, labels]
    target = probabilities[rows, labels]
    difference = probabilities.copy()
    difference[rows, labels] -= 1
    factor = 1 - target - target * log_target
    grad = factor[:, None] * difference
    hess = (
        factor[:, None] * probabilities * (1 - probabilities)
        + (target * (2 + log_target))[:, None] * difference**2
    )
    return grad, np.maximum(hess, 1e-6)


def mlogloss(preds, dtrain):
    """Custom objectives supply raw margins, not softmax probabilities."""
    margins = np.asarray(preds, dtype=np.float64).reshape(-1, N_CATEGORIES)
    labels = dtrain.get_label().astype(int)
    loss = logsumexp(margins, axis=1) - margins[np.arange(len(labels)), labels]
    return "mlogloss", float(loss.mean())


def fit_model(Xtr, ctr, Xv, cv, *, seed: int):
    params = {
        "objective": "multi:softprob", "num_class": N_CATEGORIES,
        "max_depth": 5, "learning_rate": 0.05, "min_child_weight": 3,
        "gamma": 0.1, "subsample": 0.7, "colsample_bytree": 0.7,
        "colsample_bylevel": 0.7, "reg_alpha": 1.0, "reg_lambda": 2.0,
        "tree_method": "hist", "device": DEVICE, "nthread": 8, "seed": seed,
        "disable_default_eval_metric": 1,
    }
    booster = xgb.train(
        params, xgb.DMatrix(Xtr, label=ctr), num_boost_round=10000,
        obj=focal_loss_objective, custom_metric=mlogloss,
        evals=[(xgb.DMatrix(Xv, label=cv), "val")],
        early_stopping_rounds=1000, maximize=False, verbose_eval=False,
    )
    return BoosterPMF(booster, Xtr.shape[1])


def fit_predict(prep: Prepared, *, cnn_cfg: dict | None, seed: int) -> FamilyOutput:
    Xtr, Xv, Xt = (tree_view(X) for X in (prep.Xtr, prep.Xv, prep.Xt))
    model = fit_model(Xtr, prep.ctr, Xv, prep.cv, seed=seed)
    importance = dict(zip(tree_feature_names(prep.cols), (100 * model.feature_importances_).tolist()))
    return FamilyOutput(
        model.predict_proba(Xv), model.predict_proba(Xt), importance,
        {"best_iteration": model.best_iteration, "gamma": 1.0},
    )
