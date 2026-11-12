"""CPU proportional-odds regression with training-only standardization."""

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from torch.nn import functional as F

from models._shared.families import FamilyOutput
from models._shared.visibility import N_CATEGORIES
from models._shared.windows import Prepared, tree_feature_names, tree_view

NEEDS_CNN_CFG = False
MAX_ITER = 4000
L2 = 1e-4


class OrderedLogit:
    def __init__(self, scaler, coef, cutpoints):
        self.scaler = scaler
        self.coef_ = coef
        self.cutpoints_ = cutpoints
        self.optimization_ = None

    def predict_proba(self, X):
        score = self.scaler.transform(X) @ self.coef_
        cumulative = expit(self.cutpoints_[None, :] - score[:, None])
        return np.diff(
            np.column_stack((np.zeros(len(X)), cumulative, np.ones(len(X)))), axis=1
        )


def fit_model(Xtr, ctr, Xv, cv, *, seed: int):
    scaler = StandardScaler().fit(Xtr)
    X = torch.as_tensor(scaler.transform(Xtr), dtype=torch.float64, device="cpu")
    c = torch.as_tensor(ctr if ctr.flags.writeable else ctr.copy(), dtype=torch.long, device="cpu")
    counts = np.bincount(ctr, minlength=N_CATEGORIES) + 0.5
    cumulative = np.cumsum(counts)[:-1] / counts.sum()
    initial = np.log(cumulative / (1 - cumulative))
    n_features = X.shape[1]
    initial_params = np.r_[np.zeros(n_features), initial[0], np.diff(initial) + 1e-6]
    lower = np.r_[np.full(n_features + 1, -np.inf), np.full(N_CATEGORIES - 2, 1e-6)]
    upper = np.full(len(initial_params), np.inf)
    low, high = c == 0, c == N_CATEGORIES - 1
    middle = ~(low | high)

    def objective(values):
        params = torch.tensor(values, dtype=torch.float64, device="cpu", requires_grad=True)
        coef, cuts = params[:n_features], params[n_features:]
        theta = torch.cat((cuts[:1], cuts[0] + torch.cumsum(cuts[1:], 0)))
        score = X @ coef
        logp = torch.empty(len(c), dtype=torch.float64, device="cpu")
        logp[low] = F.logsigmoid(theta[0] - score[low])
        logp[high] = F.logsigmoid(score[high] - theta[-1])
        upper = theta[c[middle]] - score[middle]
        lower = theta[c[middle] - 1] - score[middle]
        # Stable log(sigmoid(upper) - sigmoid(lower)), including saturated tails.
        logp[middle] = F.logsigmoid(upper) + F.logsigmoid(-lower) + torch.log(-torch.expm1(lower - upper))
        loss = -logp.mean() + L2 * coef.square().sum()
        gradient, = torch.autograd.grad(loss, params)
        return float(loss.detach()), gradient.detach().numpy()

    options = {"maxiter": MAX_ITER, "maxfun": 20000, "maxls": 50, "maxcor": 100,
               "gtol": 1e-7, "ftol": 0.0}
    result = minimize(objective, initial_params, method="L-BFGS-B", jac=True,
                      bounds=list(zip(lower, upper)), options=options)
    value, gradient = objective(result.x)
    projected = result.x - np.clip(result.x - gradient, lower, upper)
    cuts = result.x[n_features:]
    theta = np.r_[cuts[0], cuts[0] + np.cumsum(cuts[1:])]
    accepted = bool(
        np.isfinite(value) and np.isfinite(result.x).all() and np.isfinite(gradient).all()
        and np.all(result.x >= lower) and np.all(result.x <= upper)
        and np.all(np.diff(theta) > 0) and np.max(np.abs(projected)) <= options["gtol"]
    )
    optimization = {
        "method": "L-BFGS-B", **options, "nit": int(result.nit), "nfev": int(result.nfev),
        "objective": value, "grad_inf_norm": float(np.max(np.abs(gradient))),
        "projected_grad_inf_norm": float(np.max(np.abs(projected))),
        "status": int(result.status), "message": str(result.message), "converged": accepted,
    }
    if not accepted:
        raise RuntimeError(f"ordered-logit optimization rejected: {optimization}")
    model = OrderedLogit(scaler, result.x[:n_features].copy(), theta)
    model.optimization_ = optimization
    return model


def fit_predict(prep: Prepared, *, cnn_cfg: dict | None, seed: int) -> FamilyOutput:
    Xtr, Xv, Xt = (tree_view(X) for X in (prep.Xtr, prep.Xv, prep.Xt))
    model = fit_model(Xtr, prep.ctr, Xv, prep.cv, seed=seed)
    values = np.abs(model.coef_)
    importance = dict(zip(tree_feature_names(prep.cols), (100 * values / (values.sum() or 1)).tolist()))
    return FamilyOutput(
        pmf_val=model.predict_proba(Xv),
        pmf_test=model.predict_proba(Xt),
        importance=importance,
        info={"optimization": model.optimization_},
    )
