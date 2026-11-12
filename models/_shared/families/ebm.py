"""Weather-only EBM with an explicit 18-category probability roster.

Interpret-core 0.7.8 can fit multiclass interactions, but cannot visualize pairs
in global explanations. Keep interactions off for fully interpretable mains:
https://github.com/interpretml/interpret/blob/v0.7.8/python/interpret-core/interpret/glassbox/_ebm/_ebm.py#L1244-L1246
API: https://interpret.ml/docs/python/api/ExplainableBoostingClassifier.html
"""

import numpy as np

from ..visibility import N_CATEGORIES
from ..windows import Prepared, tree_feature_names, tree_view
from . import FamilyOutput

NEEDS_CNN_CFG = False
MAX_TRAIN_ROWS = 100_000


def fit_predict(prep: Prepared, *, cnn_cfg: dict | None, seed: int) -> FamilyOutput:
    from interpret.glassbox import ExplainableBoostingClassifier

    Xtr, ctr = prep.Xtr, prep.ctr
    # ponytail: cap for runtime; raise if time allows
    if len(ctr) > MAX_TRAIN_ROWS:
        rows = np.random.default_rng(seed).choice(len(ctr), MAX_TRAIN_ROWS, replace=False)
        Xtr, ctr = Xtr[rows], ctr[rows]
    names = tree_feature_names(prep.cols)
    model = ExplainableBoostingClassifier(
        feature_names=names, outer_bags=2, interactions=0,
        random_state=seed, n_jobs=-1,
    ).fit(tree_view(Xtr), ctr)

    def predict(X):
        pmf = np.zeros((len(X), N_CATEGORIES), dtype=float)
        pmf[:, np.asarray(model.classes_, dtype=int)] = model.predict_proba(tree_view(X))
        return pmf

    values = np.asarray(model.term_importances(), dtype=float)
    importance = dict(zip(names, (100 * values / (values.sum() or 1)).tolist()))
    return FamilyOutput(
        pmf_val=predict(prep.Xv), pmf_test=predict(prep.Xt), importance=importance,
        info={
            "n_train_used": len(ctr), "interactions": 0,
            "top_terms": sorted(importance.items(), key=lambda item: -item[1])[:15],
        },
    )
