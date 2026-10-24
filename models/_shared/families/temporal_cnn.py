"""Temporal features with a proportional-odds visibility distribution."""
from __future__ import annotations

import os
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from ..data import SEED
from ..evaluation import nll, pmf_to_fog, safe_ap
from ..visibility import FOG_MAX_CATEGORY, N_CATEGORIES
from . import FamilyOutput

NEEDS_CNN_CFG = True


def seed_all(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TemporalCNN(nn.Module):
    def __init__(self, n_features, cfg):
        super().__init__()
        channels, kernel = int(cfg["channels"]), int(cfg["kernel"])
        dropout = float(cfg["dropout"])
        self.net = nn.Sequential(
            nn.Conv1d(n_features, channels, kernel, padding=kernel // 2), nn.ReLU(),
            nn.BatchNorm1d(channels), nn.Dropout(dropout),
            nn.Conv1d(channels, channels * 2, kernel, padding=kernel // 2), nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(channels * 2, 1))
        self.cutpoint_start = nn.Parameter(torch.tensor(-3.0))
        gap = 6.0 / (N_CATEGORIES - 2)
        self.cutpoint_gaps = nn.Parameter(torch.full((N_CATEGORIES - 2,), float(np.log(np.expm1(gap)))))

    def cutpoints(self):
        return torch.cat((self.cutpoint_start[None],
                          self.cutpoint_start + F.softplus(self.cutpoint_gaps).cumsum(0)))

    def log_probabilities(self, x):
        score = self.head(self.net(x.transpose(1, 2)))
        logits = self.cutpoints()[None, :] - score
        # sigmoid(b) - sigmoid(a) = sigmoid(b) * sigmoid(-a)
        # * (1 - exp(a-b)); the ordered gap is softplus(raw), so the
        # last factor's log is logsigmoid(raw), even for extreme scores.
        interior = (F.logsigmoid(logits[:, 1:]) + F.logsigmoid(-logits[:, :-1])
                    + F.logsigmoid(self.cutpoint_gaps)[None, :])
        return torch.cat((F.logsigmoid(logits[:, :1]), interior,
                          F.logsigmoid(-logits[:, -1:])), dim=1)

    def forward(self, x):
        return self.log_probabilities(x).exp()


def predict_cnn(m, X, device, batch=4096):
    m.eval()
    out = []
    with torch.no_grad():
        for (xb,) in DataLoader(TensorDataset(torch.as_tensor(X, dtype=torch.float32)), batch_size=batch):
            out.append(m(xb.to(device)).cpu().numpy())
    return np.concatenate(out) if out else np.empty((0, N_CATEGORIES), dtype=np.float32)


def balanced_subset(X, c, max_neg=80000, seed=SEED):
    """Cap only 10+ SM rows; inverse inclusion weights retain the full prior."""
    if max_neg < 1:
        raise ValueError("max_neg must retain at least one category-17 row")
    c = np.asarray(c, dtype=np.int64)
    other = np.flatnonzero(c != N_CATEGORIES - 1)
    top = np.flatnonzero(c == N_CATEGORIES - 1)
    kept = min(len(top), int(max_neg))
    selected_top = np.random.default_rng(seed).choice(top, size=kept, replace=False)
    ix = np.sort(np.r_[other, selected_top])
    weights = np.ones(len(ix), dtype=np.float32)
    if kept:
        weights[c[ix] == N_CATEGORIES - 1] = len(top) / kept
    return X[ix], c[ix], weights


def _ordinal_nll(log_pmf, categories):
    return -log_pmf.gather(1, categories[:, None]).squeeze(1)


def train_cnn(Xtr, ctr, Xv, cv, config, device="cuda", max_epochs=16, patience=4, seed=SEED, report=None):
    if not len(ctr) or not len(cv) or max_epochs < 1:
        raise ValueError("CNN training requires nonempty train/validation arrays and at least one epoch")
    seed_all(seed)
    Xs, cs, weights = balanced_subset(Xtr, ctr, config.get("max_neg", 80000), seed)
    m = TemporalCNN(Xtr.shape[-1], config).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"]))
    gen = torch.Generator().manual_seed(seed)
    dl = DataLoader(TensorDataset(torch.as_tensor(Xs, dtype=torch.float32), torch.from_numpy(cs),
                                 torch.from_numpy(weights)), batch_size=int(config["batch_size"]),
                    shuffle=True, generator=gen, num_workers=0)
    # Fixed population normalization, not a random per-minibatch weight ratio.
    weight_mean = float(weights.mean())
    best, best_ap, best_state, stale, epochs = float("inf"), 0.0, None, 0, 0
    for ep in range(max_epochs):
        m.train()
        for xb, cb, wb in dl:
            xb, cb, wb = xb.to(device), cb.to(device), wb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = (_ordinal_nll(m.log_probabilities(xb), cb) * wb).mean() / weight_mean
            loss.backward()
            opt.step()
        pv = predict_cnn(m, Xv, device)
        val_nll = nll(cv, pv)
        ap = safe_ap(np.asarray(cv) <= FOG_MAX_CATEGORY, pmf_to_fog(pv))
        epochs = ep + 1
        if report:
            report({"val_nll": val_nll, "val_ap": ap, "epoch": epochs})
        if val_nll < best - 1e-6:
            best, best_ap = val_nll, ap
            best_state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise ValueError("CNN validation NLL is not finite")
    m.load_state_dict(best_state)
    return m, {"cnn_best_val_nll": float(best), "cnn_best_val_ap": float(best_ap), "cnn_epochs": epochs}


def fit_predict(prep, *, cnn_cfg: dict | None, seed: int) -> FamilyOutput:
    if cnn_cfg is None:
        raise ValueError("temporal_cnn requires a tuned configuration")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, info = train_cnn(prep.Xtr, prep.ctr, prep.Xv, prep.cv, cnn_cfg,
                            device=device, max_epochs=6, patience=2, seed=seed)
    weights = model.net[0].weight.detach().abs().mean(dim=(0, 2)).cpu().numpy()
    importance = dict(zip(prep.cols, (100 * weights / weights.sum()).tolist())) if weights.sum() else {}
    output = FamilyOutput(predict_cnn(model, prep.Xv, device), predict_cnn(model, prep.Xt, device),
                          importance, info,
                          {key: value.detach().cpu().clone() for key, value in model.state_dict().items()})
    del model
    torch.cuda.empty_cache()
    return output
