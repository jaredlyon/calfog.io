"""Seventeen cumulative binary models, monotonized into a category PMF."""

import numpy as np
import xgboost as xgb

from ..visibility import N_CATEGORIES
from ..windows import tree_view, tree_feature_names
from . import FamilyOutput
from .xgboost import BASE_PARAMS

NEEDS_CNN_CFG = False


class OrdinalBoosterPMF:
    def __init__(self, thresholds, n_features):
        self.thresholds = thresholds
        gain = np.zeros(n_features)
        for model in thresholds:
            if not isinstance(model, float):
                scores = model.get_score(importance_type="gain")
                gain += [scores.get(f"f{i}", 0.0) for i in range(n_features)]
        gain /= N_CATEGORIES - 1
        self.feature_importances_ = gain / (gain.sum() or 1)

    def predict_cumulative(self, X):
        data = xgb.DMatrix(X)
        probabilities = np.column_stack([
            np.full(len(X), model) if isinstance(model, float) else model.predict(
                data, iteration_range=(0, int(model.best_iteration) + 1),
            )
            for model in self.thresholds
        ])
        return np.clip(np.maximum.accumulate(probabilities, axis=1), 0, 1)

    def predict_proba(self, X):
        cdf = self.predict_cumulative(X)
        return np.diff(cdf, axis=1, prepend=0, append=1)


def fit_model(Xtr, ctr, Xv, cv, *, seed: int):
    thresholds = []
    train, val = xgb.DMatrix(Xtr), xgb.DMatrix(Xv)
    for threshold in range(N_CATEGORIES - 1):
        labels = (ctr <= threshold).astype(int)
        if np.all(labels == labels[0]):
            thresholds.append(float(labels[0]))
            continue
        train.set_label(labels)
        val.set_label((cv <= threshold).astype(int))
        thresholds.append(xgb.train(
            {**BASE_PARAMS, "seed": seed, "objective": "binary:logistic", "eval_metric": "logloss"},
            train, num_boost_round=1000, evals=[(val, "val")],
            early_stopping_rounds=40, verbose_eval=False,
        ))
    return OrdinalBoosterPMF(thresholds, Xtr.shape[1])


def fit_predict(prep, *, cnn_cfg: dict | None, seed: int) -> FamilyOutput:
    Xtr, Xv, Xt = (tree_view(X) for X in (prep.Xtr, prep.Xv, prep.Xt))
    model = fit_model(Xtr, prep.ctr, Xv, prep.cv, seed=seed)
    importance = dict(zip(tree_feature_names(prep.cols), (100 * model.feature_importances_).tolist()))
    return FamilyOutput(model.predict_proba(Xv), model.predict_proba(Xt), importance)
