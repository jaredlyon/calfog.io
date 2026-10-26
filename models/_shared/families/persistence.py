"""Markov category baseline using the latest causally available visibility."""

import numpy as np

from models._shared.evaluation import rps
from models._shared.families import FamilyOutput
from models._shared.visibility import N_CATEGORIES
from models._shared.windows import Prepared

NEEDS_CNN_CFG = False
ALPHAS = (1, 10, 100)


def fit_predict(prep: Prepared, *, cnn_cfg: dict | None, seed: int) -> FamilyOutput:
    counts = np.bincount(prep.ctr, minlength=N_CATEGORIES) + 0.5
    prior = counts / counts.sum()
    lags = {
        split: prep.metas[split]["lag_category"].to_numpy(dtype=float, na_value=np.nan)
        for split in ("train", "val", "test")
    }
    transitions = np.zeros((N_CATEGORIES, N_CATEGORIES))
    known_train = np.isfinite(lags["train"])
    np.add.at(
        transitions,
        (lags["train"][known_train].astype(int), prep.ctr[known_train]),
        1,
    )

    def predict(split, alpha):
        lag = lags[split]
        known = np.isfinite(lag)
        pmf = np.broadcast_to(prior, (len(lag), N_CATEGORIES)).copy()
        rows = transitions[lag[known].astype(int)]
        pmf[known] = (rows + alpha * prior) / (rows.sum(axis=1, keepdims=True) + alpha)
        return pmf

    candidates = [(rps(prep.cv, pmf := predict("val", alpha)), alpha, pmf) for alpha in ALPHAS]
    _, alpha, pmf_val = min(candidates, key=lambda candidate: candidate[0])
    known_test = np.isfinite(lags["test"])
    return FamilyOutput(
        pmf_val=pmf_val,
        pmf_test=predict("test", alpha),
        importance={},
        info={
            "alpha": alpha,
            "lag_coverage_test": float(known_test.mean()) if len(known_test) else None,
            "persistence_exact_test": float((lags["test"][known_test] == prep.ct[known_test]).mean())
            if known_test.any() else None,
        },
    )
