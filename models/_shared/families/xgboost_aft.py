"""Interval-censored visibility with validation-selected AFT distributions."""

import numpy as np
import xgboost as xgb
from scipy.special import expit, ndtr

from ..evaluation import rps
from ..visibility import CATEGORY_LOWER_M, CATEGORY_UPPER_M
from ..windows import tree_view, tree_feature_names
from . import FamilyOutput
from .xgboost import BASE_PARAMS

NEEDS_CNN_CFG = False
DISTRIBUTIONS = ("normal", "logistic")
SCALES = (0.5, 1.0, 1.5)


def median_to_pmf(median, distribution, scale):
    # Boundaries include the open left and right tails of the visibility scale. The predicted median can
    # overflow to inf (or underflow to 0); clip it so inf - inf never yields NaN at the open top category.
    bounds = np.r_[CATEGORY_LOWER_M, np.inf]
    log_median = np.log(np.clip(np.asarray(median, dtype=float), 1e-3, 1e12))
    with np.errstate(divide="ignore"):
        z = (np.log(bounds)[None, :] - log_median[:, None]) / scale
    cdf = ndtr(z) if distribution == "normal" else expit(z)
    return np.diff(cdf, axis=1)


class AFTBoosterPMF:
    def __init__(self, booster, n_features, distribution, scale):
        self.booster = booster
        self.distribution = distribution
        self.scale = scale
        self.best_iteration = int(booster.best_iteration)
        gain = booster.get_score(importance_type="gain")
        values = np.array([gain.get(f"f{i}", 0.0) for i in range(n_features)])
        self.feature_importances_ = values / (values.sum() or 1)

    def predict_proba(self, X):
        median = self.booster.predict(
            xgb.DMatrix(X), iteration_range=(0, self.best_iteration + 1),
        )
        return median_to_pmf(median, self.distribution, self.scale)


def _interval_data(X, categories):
    data = xgb.DMatrix(X)
    data.set_float_info("label_lower_bound", np.asarray(CATEGORY_LOWER_M)[categories])
    data.set_float_info("label_upper_bound", np.asarray(CATEGORY_UPPER_M)[categories])
    return data


def fit_model(Xtr, ctr, Xv, cv, *, seed: int):
    train, val = _interval_data(Xtr, ctr), _interval_data(Xv, cv)
    best, best_score = None, np.inf
    for distribution in DISTRIBUTIONS:
        for scale in SCALES:
            booster = xgb.train(
                {**BASE_PARAMS, "seed": seed, "objective": "survival:aft",
                 "eval_metric": "aft-nloglik", "aft_loss_distribution": distribution,
                 "aft_loss_distribution_scale": scale},
                train, num_boost_round=1000, evals=[(val, "val")],
                early_stopping_rounds=40, verbose_eval=False,
            )
            model = AFTBoosterPMF(booster, Xtr.shape[1], distribution, scale)
            score = rps(cv, model.predict_proba(Xv))
            if score < best_score:
                best, best_score = model, score
    return best


def fit_predict(prep, *, cnn_cfg: dict | None, seed: int) -> FamilyOutput:
    Xtr, Xv, Xt = (tree_view(X) for X in (prep.Xtr, prep.Xv, prep.Xt))
    model = fit_model(Xtr, prep.ctr, Xv, prep.cv, seed=seed)
    importance = dict(zip(tree_feature_names(prep.cols), (100 * model.feature_importances_).tolist()))
    return FamilyOutput(
        model.predict_proba(Xv), model.predict_proba(Xt), importance,
        {"distribution": model.distribution, "scale": model.scale, "best_iteration": model.best_iteration},
    )
