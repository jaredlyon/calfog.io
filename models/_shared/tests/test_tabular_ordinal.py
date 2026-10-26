"""Synthetic ordinal recovery, absent categories, and focal derivatives."""

import numpy as np
import pytest
from scipy.special import logsumexp

from models._shared.evaluation import rps
from models._shared.families import logistic_regression, random_forest, xgboost_focal_loss
from models._shared.visibility import N_CATEGORIES


def test_ordered_logit_recovers_monotonic_categories_and_beats_uniform_rps():
    rng = np.random.default_rng(91)
    Xtr = rng.uniform(-3, 3, (240, 2))
    Xv = rng.uniform(-3, 3, (60, 2))
    edges = np.linspace(-3, 3, N_CATEGORIES + 1)[1:-1]
    ctr = np.digitize(Xtr[:, 0] + 0.08 * rng.normal(size=len(Xtr)), edges)
    cv = np.digitize(Xv[:, 0], edges)
    model = logistic_regression.fit_model(Xtr, ctr, Xv, cv, seed=42)
    pmf = model.predict_proba(Xv)
    uniform = np.full_like(pmf, 1 / N_CATEGORIES)
    assert rps(cv, pmf) < 0.5 * rps(cv, uniform)
    grid = np.column_stack((np.linspace(-3, 3, 50), np.zeros(50)))
    cdf = model.predict_proba(grid).cumsum(1)[:, :-1]
    assert np.all(np.diff(cdf, axis=0) <= 1e-12)


def test_forest_pads_absent_nonconsecutive_categories():
    X = np.linspace(-1, 1, 48)[:, None]
    c = np.where(X[:, 0] < 0, 2, 15)
    model = random_forest.fit_model(X, c, X[:8], c[:8], seed=42)
    pmf = model.predict_proba_raw(X)
    absent = np.setdiff1d(np.arange(N_CATEGORIES), [2, 15])
    assert pmf.shape == (len(X), N_CATEGORIES)
    np.testing.assert_array_equal(pmf[:, absent], 0)
    np.testing.assert_allclose(pmf.sum(1), 1)
    assert pmf[0, 2] > pmf[0, 15]
    assert pmf[-1, 15] > pmf[-1, 2]


def test_ordered_logit_keeps_full_support_when_categories_are_absent():
    X = np.linspace(-1, 1, 60)[:, None]
    c = np.where(X[:, 0] < 0, 3, 12)
    model = logistic_regression.fit_model(X, c, X[:10], c[:10], seed=42)
    pmf = model.predict_proba(np.zeros((2, 1)))
    assert pmf.shape == (2, N_CATEGORIES)
    assert np.isfinite(pmf).all()
    assert (pmf >= 0).all()
    np.testing.assert_allclose(pmf.sum(1), 1)
    assert np.all(np.diff(model.cutpoints_) > 0)
    absent = np.setdiff1d(np.arange(N_CATEGORIES), [3, 12])
    assert pmf[:, absent].sum(1).max() < 0.05


class CategoryLabels:
    def __init__(self, labels):
        self.labels = np.asarray(labels)

    def get_label(self):
        return self.labels


def test_focal_gradient_and_diagonal_hessian_match_finite_differences():
    rng = np.random.default_rng(13)
    margins = rng.normal(size=(3, N_CATEGORIES))
    labels = np.array([0, 8, 17])
    data = CategoryLabels(labels)
    grad, hess = xgboost_focal_loss.focal_loss_objective(margins, data)

    def loss(values):
        logp = values - logsumexp(values, axis=1, keepdims=True)
        target_logp = logp[np.arange(len(labels)), labels]
        return -np.sum((1 - np.exp(target_logp)) * target_logp)

    eps = 1e-5
    finite_grad = np.empty_like(grad)
    finite_hess = np.empty_like(hess)
    for row, column in np.ndindex(margins.shape):
        plus, minus = margins.copy(), margins.copy()
        plus[row, column] += eps
        minus[row, column] -= eps
        finite_grad[row, column] = (loss(plus) - loss(minus)) / (2 * eps)
        grad_plus, _ = xgboost_focal_loss.focal_loss_objective(plus, data)
        grad_minus, _ = xgboost_focal_loss.focal_loss_objective(minus, data)
        finite_hess[row, column] = (grad_plus[row, column] - grad_minus[row, column]) / (2 * eps)
    np.testing.assert_allclose(grad, finite_grad, atol=1e-9, rtol=1e-6)
    np.testing.assert_allclose(hess, np.maximum(finite_hess, 1e-6), atol=1e-9, rtol=1e-6)
    np.testing.assert_allclose(grad.sum(1), 0, atol=1e-12)


def test_focal_extreme_margins_are_finite_and_negative_curvature_is_floored():
    margins = np.zeros((2, N_CATEGORIES))
    margins[0, 0] = 1000
    margins[1, 0] = -1000
    grad, hess = xgboost_focal_loss.focal_loss_objective(margins, CategoryLabels([0, 0]))
    assert np.isfinite(grad).all()
    assert np.isfinite(hess).all()
    assert np.all(hess >= 1e-6)
    assert hess[1, 0] == pytest.approx(1e-6)
    name, loss = xgboost_focal_loss.mlogloss(margins, CategoryLabels([0, 0]))
    assert name == "mlogloss"
    assert np.isfinite(loss)
