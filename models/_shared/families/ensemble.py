"""Chronological LR -> RF -> XGB -> CNN staircase with a fixed PMF blend.

Five contiguous blocks generate strictly earlier-prefix upstream predictions.
Each stage appends 17 cumulative probabilities, never in-sample features.
RF stacking features stay raw; only its final blend contribution is calibrated.
OOF XGB early stopping uses the prefix's final 15%, not the predicted block.
Final refits use available OOF rows; final XGB/CNN use outer validation.
"""

import numpy as np
import torch

from . import FamilyOutput, seed_for
from .logistic_regression import fit_model as fit_lr
from .random_forest import fit_model as fit_rf
from .xgboost import fit_model as fit_xgb
from .temporal_cnn import predict_cnn, train_cnn
from ..visibility import FOG_MAX_CATEGORY, N_CATEGORIES
from ..windows import tree_feature_names, tree_view

NEEDS_CNN_CFG = True
N_BLOCKS = 5
BLEND_WEIGHTS = (0.1, 0.2, 0.3, 0.4)


def _cnn_inputs(windows, probabilities):
    channels = np.broadcast_to(
        probabilities[:, None, :],
        (len(windows), windows.shape[1], probabilities.shape[1]),
    )
    return np.concatenate((windows, channels), axis=2).astype(np.float32)


def _importance(models, names):
    values = np.zeros(len(names), dtype=float)
    stage_values = (np.abs(models[0].coef_), models[1].feature_importances_,
                    models[2].feature_importances_)
    for weight, stage in zip((1 / 6, 2 / 6, 3 / 6), stage_values):
        original = np.asarray(stage, dtype=float)[: len(names)]
        if original.sum() > 0:
            values += weight * original / original.sum()
    total = values.sum()
    return dict(zip(names, (100 * values / total).tolist())) if total else {}


def _require_categories(categories, stage, block):
    distinct = np.unique(categories).size
    fog = int(np.count_nonzero(categories <= FOG_MAX_CATEGORY))
    nonfog = int(np.count_nonzero(categories > FOG_MAX_CATEGORY))
    if distinct < 2 or not fog or not nonfog:
        from ..study import NotEstimable

        raise NotEstimable(f"{stage} block {block}: categories={distinct}, fog={fog}, nonfog={nonfog}; "
                           "requires at least two categories and both fog/non-fog rows")


def fit_predict(prep, *, cnn_cfg: dict | None, seed: int) -> FamilyOutput:
    if cnn_cfg is None:
        raise ValueError("ensemble requires cnn_cfg")
    if len(prep.ctr) < 2 * N_BLOCKS:
        raise ValueError("ensemble requires at least ten chronological training rows")

    features = tree_view(prep.Xtr)
    n_features = features.shape[1]
    n_cumulative = N_CATEGORIES - 1
    train = np.full((len(features), n_features + 3 * n_cumulative), np.nan, dtype=np.float32)
    train[:, :n_features] = features
    val = np.empty((len(prep.Xv), train.shape[1]), dtype=np.float32)
    test = np.empty((len(prep.Xt), train.shape[1]), dtype=np.float32)
    val[:, :n_features] = tree_view(prep.Xv)
    test[:, :n_features] = tree_view(prep.Xt)
    blocks = np.array_split(np.arange(len(train)), N_BLOCKS)
    bounds = np.r_[0, np.cumsum([len(block) for block in blocks])]
    fitters = (fit_lr, fit_rf, fit_xgb)
    names = ("logistic_regression", "random_forest", "xgboost")

    # Fail before any fits, including all-one-side later prefixes.
    for stage, name in enumerate(names):
        for block in range(stage + 1, N_BLOCKS):
            categories = prep.ctr[bounds[stage]:bounds[block]]
            if stage == 2:
                categories = categories[:max(1, int(len(categories) * 0.85))]
            _require_categories(categories, name, f"{block + 1} OOF")
        _require_categories(prep.ctr[bounds[stage]:], name, f"{stage + 1}..{N_BLOCKS} refit")
    _require_categories(prep.ctr[bounds[3]:], "temporal_cnn", f"4..{N_BLOCKS} refit")

    for stage, (fit_model, name) in enumerate(zip(fitters, names)):
        width = n_features + stage * n_cumulative
        for block in range(stage + 1, N_BLOCKS):
            start, stop = bounds[stage], bounds[block]
            Xtr, ctr = train[start:stop, :width], prep.ctr[start:stop]
            if stage == 2:
                cut = max(1, int(len(Xtr) * 0.85))
                Xtr, Xv = Xtr[:cut], Xtr[cut:]
                ctr, cv = ctr[:cut], ctr[cut:]
            else:
                Xv, cv = Xtr[:0], ctr[:0]
            model = fit_model(Xtr, ctr, Xv, cv, seed=seed_for(seed, name, "oof", block))
            rows = slice(bounds[block], bounds[block + 1])
            predict = model.predict_proba_raw if stage == 1 else model.predict_proba
            pmf = predict(train[rows, :width])
            train[rows, width:width + n_cumulative] = pmf.cumsum(axis=1)[:, :-1]

    models, pmf_val, pmf_test = [], [], []
    for stage, (fit_model, name) in enumerate(zip(fitters, names)):
        width = n_features + stage * n_cumulative
        start = bounds[stage]
        model = fit_model(train[start:, :width], prep.ctr[start:], val[:, :width], prep.cv,
                          seed=seed_for(seed, name, "refit"))
        models.append(model)
        pv, pt = model.predict_proba(val[:, :width]), model.predict_proba(test[:, :width])
        pmf_val.append(pv)
        pmf_test.append(pt)
        if stage == 1:
            pv, pt = model.predict_proba_raw(val[:, :width]), model.predict_proba_raw(test[:, :width])
        val[:, width:width + n_cumulative] = pv.cumsum(axis=1)[:, :-1]
        test[:, width:width + n_cumulative] = pt.cumsum(axis=1)[:, :-1]

    start = bounds[3]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cnn, cnn_info = train_cnn(
        _cnn_inputs(prep.Xtr[start:], train[start:, n_features:]), prep.ctr[start:],
        _cnn_inputs(prep.Xv, val[:, n_features:]), prep.cv, cnn_cfg,
        device=device, seed=seed_for(seed, "temporal_cnn", "refit"),
    )
    pmf_val.append(predict_cnn(cnn, _cnn_inputs(prep.Xv, val[:, n_features:]), device))
    pmf_test.append(predict_cnn(cnn, _cnn_inputs(prep.Xt, test[:, n_features:]), device))
    return FamilyOutput(
        pmf_val=sum(weight * pmf for weight, pmf in zip(BLEND_WEIGHTS, pmf_val)),
        pmf_test=sum(weight * pmf for weight, pmf in zip(BLEND_WEIGHTS, pmf_test)),
        importance=_importance(models, tree_feature_names(prep.cols)),
        info={"inner_blocks": N_BLOCKS, "cnn_train_rows": int(len(prep.ctr) - start),
              **cnn_info, "blend_weights": list(BLEND_WEIGHTS)},
        state={key: value.detach().cpu().clone() for key, value in cnn.state_dict().items()},
    )
