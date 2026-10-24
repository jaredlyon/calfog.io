"""CPU proportional-odds regression with training-only standardization."""

import numpy as np
import torch
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from torch.nn import functional as F

from models._shared.families import FamilyOutput
from models._shared.visibility import N_CATEGORIES
from models._shared.windows import Prepared, tree_feature_names, tree_view

NEEDS_CNN_CFG = False
MAX_ITER = 100
L2 = 1e-4


class OrderedLogit:
    def __init__(self, scaler, coef, cutpoints):
        self.scaler = scaler
        self.coef_ = coef
        self.cutpoints_ = cutpoints

    def predict_proba(self, X):
        score = self.scaler.transform(X) @ self.coef_
        cumulative = expit(self.cutpoints_[None, :] - score[:, None])
        return np.diff(
            np.column_stack((np.zeros(len(X)), cumulative, np.ones(len(X)))), axis=1
        )


def fit_model(Xtr, ctr, Xv, cv, *, seed: int):
    scaler = StandardScaler().fit(Xtr)
    X = torch.as_tensor(scaler.transform(Xtr), dtype=torch.float64, device="cpu")
    c = torch.as_tensor(ctr, dtype=torch.long, device="cpu")
    counts = np.bincount(ctr, minlength=N_CATEGORIES) + 0.5
    cumulative = np.cumsum(counts)[:-1] / counts.sum()
    initial = np.log(cumulative / (1 - cumulative))
    raw = np.r_[initial[0], np.log(np.expm1(np.diff(initial)))]
    cuts = torch.nn.Parameter(torch.tensor(raw, dtype=torch.float64, device="cpu"))
    coef = torch.nn.Parameter(torch.zeros(X.shape[1], dtype=torch.float64, device="cpu"))
    optimizer = torch.optim.LBFGS(
        [coef, cuts], max_iter=MAX_ITER, line_search_fn="strong_wolfe"
    )
    low, high = c == 0, c == N_CATEGORIES - 1
    middle = ~(low | high)

    def closure():
        optimizer.zero_grad()
        theta = torch.cat((cuts[:1], cuts[0] + torch.cumsum(F.softplus(cuts[1:]) + 1e-6, 0)))
        score = X @ coef
        logp = torch.empty(len(c), dtype=torch.float64, device="cpu")
        logp[low] = F.logsigmoid(theta[0] - score[low])
        logp[high] = F.logsigmoid(score[high] - theta[-1])
        upper = theta[c[middle]] - score[middle]
        lower = theta[c[middle] - 1] - score[middle]
        # Stable log(sigmoid(upper) - sigmoid(lower)), including saturated tails.
        logp[middle] = F.logsigmoid(upper) + F.logsigmoid(-lower) + torch.log(-torch.expm1(lower - upper))
        loss = -logp.mean() + L2 * coef.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        theta = torch.cat((cuts[:1], cuts[0] + torch.cumsum(F.softplus(cuts[1:]) + 1e-6, 0)))
    return OrderedLogit(scaler, coef.detach().numpy().copy(), theta.numpy().copy())


def fit_predict(prep: Prepared, *, cnn_cfg: dict | None, seed: int) -> FamilyOutput:
    Xtr, Xv, Xt = (tree_view(X) for X in (prep.Xtr, prep.Xv, prep.Xt))
    model = fit_model(Xtr, prep.ctr, Xv, prep.cv, seed=seed)
    values = np.abs(model.coef_)
    importance = dict(zip(tree_feature_names(prep.cols), (100 * values / (values.sum() or 1)).tolist()))
    return FamilyOutput(
        pmf_val=model.predict_proba(Xv),
        pmf_test=model.predict_proba(Xt),
        importance=importance,
    )
