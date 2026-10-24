"""Multiclass forest with a fixed visibility-category probability support."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ..visibility import N_CATEGORIES
from ..windows import tree_view, tree_feature_names
from . import FamilyOutput

NEEDS_CNN_CFG = False


class CategoryForest(RandomForestClassifier):
    def predict_proba(self, X):
        probabilities = super().predict_proba(X)
        pmf = np.zeros((len(probabilities), N_CATEGORIES))
        pmf[:, self.classes_.astype(int)] = probabilities
        return pmf


def fit_model(Xtr, ctr, Xv, cv, *, seed: int):
    model = CategoryForest(
        n_estimators=240, max_depth=16, min_samples_leaf=3, max_features="sqrt",
        class_weight="balanced_subsample", random_state=seed, n_jobs=-1,
    )
    return model.fit(Xtr, ctr)


def fit_predict(prep, *, cnn_cfg: dict | None, seed: int) -> FamilyOutput:
    Xtr, Xv, Xt = (tree_view(X) for X in (prep.Xtr, prep.Xv, prep.Xt))
    model = fit_model(Xtr, prep.ctr, Xv, prep.cv, seed=seed)
    values = model.feature_importances_
    importance = dict(zip(tree_feature_names(prep.cols), (100 * values / (values.sum() or 1)).tolist()))
    return FamilyOutput(model.predict_proba(Xv), model.predict_proba(Xt), importance, {})
