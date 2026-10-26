"""Analog category ensembles from latest and mean weather features."""

import numpy as np
from sklearn.neighbors import NearestNeighbors

from models._shared.evaluation import rps
from models._shared.families import FamilyOutput
from models._shared.visibility import N_CATEGORIES
from models._shared.windows import Prepared, tree_view

NEEDS_CNN_CFG = False
K_GRID = (25, 50, 100, 200)


def fit_predict(prep: Prepared, *, cnn_cfg: dict | None, seed: int) -> FamilyOutput:
    features = [tree_view(X)[:, :2 * len(prep.cols)] for X in (prep.Xtr, prep.Xv, prep.Xt)]
    counts = np.bincount(prep.ctr, minlength=N_CATEGORIES) + 0.5
    prior = counts / counts.sum()
    candidates = sorted({min(k, len(prep.ctr)) for k in K_GRID})
    neighbors = NearestNeighbors(n_neighbors=max(candidates), metric="euclidean").fit(features[0])
    neighbor_categories = [
        prep.ctr[neighbors.kneighbors(X, return_distance=False)]
        if len(X) else np.empty((0, max(candidates)), dtype=int)
        for X in features[1:]
    ]

    def predict(categories, k):
        histogram = np.zeros((len(categories), N_CATEGORIES))
        np.add.at(histogram, (np.arange(len(categories))[:, None], categories[:, :k]), 1)
        return (histogram + prior) / (k + 1)

    scores = [(rps(prep.cv, pmf := predict(neighbor_categories[0], k)), k, pmf) for k in candidates]
    _, k, pmf_val = min(scores, key=lambda candidate: candidate[0])
    return FamilyOutput(
        pmf_val=pmf_val,
        pmf_test=predict(neighbor_categories[1], k),
        importance={},
        info={"k": k},
    )
