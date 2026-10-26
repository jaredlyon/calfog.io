"""Native multiclass XGBoost, including categories absent from training."""

import numpy as np
import xgboost as xgb
from scipy.special import softmax

from ..visibility import N_CATEGORIES
from ..windows import tree_view, tree_feature_names
from . import FamilyOutput

NEEDS_CNN_CFG = False

BASE_PARAMS = {
    "max_depth": 5, "learning_rate": 0.04, "min_child_weight": 3,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_lambda": 2.0, "reg_alpha": 0.2,
    "nthread": 8, "tree_method": "hist", "device": "cpu",
}


class BoosterPMF:
    def __init__(self, booster, n_features):
        self.booster = booster
        self.best_iteration = int(booster.best_iteration)
        gain = booster.get_score(importance_type="gain")
        values = np.array([gain.get(f"f{i}", 0.0) for i in range(n_features)])
        self.feature_importances_ = values / (values.sum() or 1)

    def predict_proba(self, X):
        margins = self.booster.predict(
            xgb.DMatrix(X), output_margin=True,
            iteration_range=(0, self.best_iteration + 1),
        )
        return softmax(np.asarray(margins).reshape(-1, N_CATEGORIES), axis=1)


def fit_model(Xtr, ctr, Xv, cv, *, seed: int):
    params = {
        "objective": "multi:softprob", "num_class": N_CATEGORIES,
        **BASE_PARAMS, "seed": seed,
        "eval_metric": "mlogloss",
    }
    booster = xgb.train(
        params, xgb.DMatrix(Xtr, label=ctr), num_boost_round=1000,
        evals=[(xgb.DMatrix(Xv, label=cv), "val")],
        early_stopping_rounds=40, verbose_eval=False,
    )
    return BoosterPMF(booster, Xtr.shape[1])


def fit_predict(prep, *, cnn_cfg: dict | None, seed: int) -> FamilyOutput:
    Xtr, Xv, Xt = (tree_view(X) for X in (prep.Xtr, prep.Xv, prep.Xt))
    model = fit_model(Xtr, prep.ctr, Xv, prep.cv, seed=seed)
    importance = dict(zip(tree_feature_names(prep.cols), (100 * model.feature_importances_).tolist()))
    return FamilyOutput(
        model.predict_proba(Xv), model.predict_proba(Xt), importance,
        {"best_iteration": model.best_iteration},
    )
