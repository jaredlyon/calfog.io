"""Fixed-support forest, with validation-only temperature/prior calibration."""

import numpy as np
try:
    from scipy.optimize import minimize
except ImportError:
    minimize = None
from sklearn.ensemble import RandomForestClassifier

from ..visibility import N_CATEGORIES
from ..windows import tree_view, tree_feature_names
from . import FamilyOutput

NEEDS_CNN_CFG = False


def _golden_minimum(loss, lower, upper):
    ratio = (np.sqrt(5) - 1) / 2
    left, right = upper - ratio * (upper - lower), lower + ratio * (upper - lower)
    fleft, fright = loss(left), loss(right)
    for _ in range(32):
        if fleft < fright:
            upper, right, fright = right, left, fleft
            left = upper - ratio * (upper - lower)
            fleft = loss(left)
        else:
            lower, left, fleft = left, right, fright
            right = lower + ratio * (upper - lower)
            fright = loss(right)
    return left if fleft < fright else right


class CategoryForest(RandomForestClassifier):
    def predict_proba_raw(self, X):
        probabilities = super().predict_proba(X)
        pmf = np.zeros((len(probabilities), N_CATEGORIES))
        pmf[:, self.classes_.astype(int)] = probabilities
        return pmf

    def _calibrated(self, raw, temperature, prior_weight):
        logits = np.log(np.clip(raw, 1e-6, 1)) / temperature
        scaled = np.exp(logits - logits.max(axis=1, keepdims=True))
        scaled /= scaled.sum(axis=1, keepdims=True)
        return (1 - prior_weight) * scaled + prior_weight * self.prior_

    def predict_proba(self, X):
        raw = self.predict_proba_raw(X)
        if not self.calibrated_:
            return raw
        return self._calibrated(raw, self.temperature_, self.prior_weight_)

    def calibrate(self, Xv, cv):
        self.temperature_, self.prior_weight_ = 1.0, 0.0
        self.calibrated_ = bool(len(cv))
        if not len(cv):
            return self
        raw = self.predict_proba_raw(Xv)
        rows = np.arange(len(cv))

        def loss(parameters):
            pmf = self._calibrated(raw, np.exp(parameters[0]), parameters[1])
            return float(-np.log(np.clip(pmf[rows, cv], 1e-15, 1)).mean())

        # Start inside and on the bounds; retain the best grid point if optimization fails.
        candidates = [(np.log(t), w) for t in (0.05, 0.25, 1, 4, 20) for w in (0, 0.25, 0.5)]
        best = min(candidates, key=loss)
        bounds = [(np.log(0.05), np.log(20)), (0, 0.5)]
        if minimize is not None:
            result = minimize(loss, best, method="L-BFGS-B", bounds=bounds)
            if np.isfinite(result.fun) and result.fun < loss(best):
                best = result.x
        else:
            best = list(best)
            for _ in range(4):
                for axis, (lower, upper) in enumerate(bounds):
                    def coordinate(value):
                        trial = best.copy()
                        trial[axis] = value
                        return loss(trial)

                    candidate = _golden_minimum(coordinate, lower, upper)
                    if coordinate(candidate) < loss(best):
                        best[axis] = candidate
        self.temperature_, self.prior_weight_ = float(np.exp(best[0])), float(best[1])
        return self


def fit_model(Xtr, ctr, Xv, cv, *, seed: int):
    model = CategoryForest(
        n_estimators=240, max_depth=16, min_samples_leaf=3, max_features="sqrt",
        class_weight="balanced_subsample", random_state=seed, n_jobs=-1,
    )
    model.fit(Xtr, ctr)
    counts = np.bincount(ctr, minlength=N_CATEGORIES) + 0.5
    model.prior_ = counts / counts.sum()
    return model.calibrate(Xv, cv)


def fit_predict(prep, *, cnn_cfg: dict | None, seed: int) -> FamilyOutput:
    Xtr, Xv, Xt = (tree_view(X) for X in (prep.Xtr, prep.Xv, prep.Xt))
    model = fit_model(Xtr, prep.ctr, Xv, prep.cv, seed=seed)
    values = model.feature_importances_
    importance = dict(zip(tree_feature_names(prep.cols), (100 * values / (values.sum() or 1)).tolist()))
    return FamilyOutput(
        model.predict_proba(Xv), model.predict_proba(Xt), importance,
        {"temperature": model.temperature_, "prior_weight": model.prior_weight_},
    )
