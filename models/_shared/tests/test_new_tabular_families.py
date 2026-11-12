"""Tabular PMFs use training categories and validation-only early stopping."""

from types import SimpleNamespace
import json
import warnings

import numpy as np
import pytest
import torch
from scipy.special import expit

from models._shared.families import FamilyOutput
from models._shared.families import logistic_regression, random_forest, xgboost, xgboost_focal_loss
from models._shared.visibility import N_CATEGORIES
from models._shared.windows import Prepared, tree_feature_names, tree_view

FAMILIES = [logistic_regression, random_forest, xgboost, xgboost_focal_loss]


@pytest.fixture
def prep():
    rng = np.random.default_rng(42)

    def partition(n):
        categories = rng.choice([0, 4, 9, 17], size=n)
        X = rng.normal(0, 0.1, (n, 24, 2))
        X[:, :, 0] += categories[:, None] / 4
        return X.astype(np.float32), categories

    Xtr, ctr = partition(160)
    Xv, cv = partition(24)
    Xt, ct = partition(20)
    return Prepared(Xtr, ctr, Xv, cv, Xt, ct, {}, ["signal", "noise"])


@pytest.mark.parametrize("family", FAMILIES)
def test_predictions_satisfy_pmf_and_importance_contract(prep, family, monkeypatch):
    if family in (xgboost, xgboost_focal_loss):
        train = family.xgb.train

        def short_train(*args, **kwargs):
            kwargs["num_boost_round"] = 12
            kwargs["early_stopping_rounds"] = 3
            return train(*args, **kwargs)

        monkeypatch.setattr(family.xgb, "train", short_train)
    out = family.fit_predict(prep, cnn_cfg=None, seed=42)
    assert family.NEEDS_CNN_CFG is False
    assert isinstance(out, FamilyOutput)
    assert out.state is None
    for X, probabilities in [(prep.Xv, out.pmf_val), (prep.Xt, out.pmf_test)]:
        assert probabilities.shape == (len(X), N_CATEGORIES)
        assert np.isfinite(probabilities).all()
        assert ((probabilities >= 0) & (probabilities <= 1)).all()
        np.testing.assert_allclose(probabilities.sum(1), 1, atol=1e-6)
    assert set(out.importance) == set(tree_feature_names(prep.cols))
    assert sum(out.importance.values()) == pytest.approx(100)
    if family is logistic_regression:
        diagnostics = out.info["optimization"]
        assert diagnostics["converged"] is True
        assert diagnostics["projected_grad_inf_norm"] <= 1e-7
        assert set(diagnostics) == {
            "method", "maxiter", "maxfun", "maxls", "maxcor", "gtol", "ftol",
            "nit", "nfev", "objective", "grad_inf_norm", "projected_grad_inf_norm",
            "status", "message", "converged",
        }
        json.dumps(diagnostics, allow_nan=False)


def test_logistic_scaler_is_fitted_on_training_only(prep):
    Xtr = tree_view(prep.Xtr)
    model = logistic_regression.fit_model(Xtr, prep.ctr, tree_view(prep.Xv) + 1000, prep.cv, seed=42)
    np.testing.assert_allclose(model.scaler.mean_, Xtr.mean(axis=0, dtype=np.float64))
    assert model.scaler.n_samples_seen_ == len(prep.ctr)
    assert np.all(np.diff(model.cutpoints_) > 0)
    probabilities = model.predict_proba(Xtr)
    objective = -np.log(probabilities[np.arange(len(prep.ctr)), prep.ctr]).mean()
    objective += 1e-4 * np.square(model.coef_).sum()
    assert model.optimization_["objective"] == pytest.approx(objective, abs=1e-12)


@pytest.mark.parametrize("family", FAMILIES)
def test_fit_predict_never_reads_test_or_binary_labels(prep, family, monkeypatch):
    class GuardedPrepared(Prepared):
        def __getattribute__(self, name):
            if name in {"ct", "yt", "ytr", "yv"}:
                raise AssertionError("fitting must use training/validation categories only")
            return super().__getattribute__(name)

    guarded = GuardedPrepared(**vars(prep))
    calls = []
    n_features = len(tree_feature_names(prep.cols))

    def fit_spy(Xtr, ctr, Xv, cv, *, seed):
        calls.append(seed)
        np.testing.assert_array_equal(ctr, prep.ctr)
        np.testing.assert_array_equal(cv, prep.cv)
        np.testing.assert_array_equal(Xtr, tree_view(prep.Xtr))
        np.testing.assert_array_equal(Xv, tree_view(prep.Xv))
        return SimpleNamespace(
            coef_=np.ones(n_features), feature_importances_=np.full(n_features, 1 / n_features),
            best_iteration=7, temperature_=1.0, prior_weight_=0.0,
            optimization_={"converged": True, "projected_grad_inf_norm": 0.0},
            predict_proba=lambda X: np.full((len(X), N_CATEGORIES), 1 / N_CATEGORIES),
        )

    monkeypatch.setattr(family, "fit_model", fit_spy)
    family.fit_predict(guarded, cnn_cfg=None, seed=42)
    assert calls == [42]


@pytest.mark.parametrize("family", [xgboost, xgboost_focal_loss])
def test_booster_validation_labels_and_best_iteration(prep, family, monkeypatch):
    calls = []

    class Booster:
        best_iteration = 7

        def predict(self, data, *, output_margin, iteration_range):
            assert output_margin is True
            assert iteration_range == (0, 8)
            assert data.get_label().size == 0
            margins = np.zeros((data.num_row(), N_CATEGORIES))
            margins[:, 4] = 1
            return margins

        def __getitem__(self, selected):
            assert selected == slice(None, 8)
            return SimpleNamespace(get_score=lambda **kw: {"f0": 2.0, "f1": 1.0})

        def get_score(self, **kwargs):
            raise AssertionError("importance must exclude post-best trees")

    def train_spy(params, dtrain, **kwargs):
        calls.append(kwargs)
        np.testing.assert_array_equal(dtrain.get_label(), prep.ctr)
        np.testing.assert_allclose(dtrain.get_data().toarray(), tree_view(prep.Xtr))
        assert params["objective"] == "multi:softprob"
        assert params["num_class"] == N_CATEGORIES
        assert params["device"] == xgboost.DEVICE
        assert params["seed"] == 42
        assert len(kwargs["evals"]) == 1
        dval, name = kwargs["evals"][0]
        assert name == "val"
        np.testing.assert_array_equal(dval.get_label(), prep.cv)
        np.testing.assert_allclose(dval.get_data().toarray(), tree_view(prep.Xv))
        if family is xgboost:
            assert params["eval_metric"] == "mlogloss"
            assert kwargs["early_stopping_rounds"] == 40
        else:
            assert kwargs["obj"] is family.focal_loss_objective
            metric_name, loss = kwargs["custom_metric"](np.zeros((len(prep.cv), N_CATEGORIES)), dval)
            assert metric_name == "mlogloss"
            assert loss == pytest.approx(np.log(N_CATEGORIES))
            assert kwargs["maximize"] is False
            assert kwargs["early_stopping_rounds"] == 1000
        return Booster()

    monkeypatch.setattr(family.xgb, "train", train_spy)
    out = family.fit_predict(prep, cnn_cfg=None, seed=42)
    assert len(calls) == 1
    assert out.pmf_test[0, 4] == pytest.approx(np.e / (np.e + N_CATEGORIES - 1))
    names = tree_feature_names(prep.cols)
    assert out.importance[names[0]] == pytest.approx(200 / 3)
    assert out.importance[names[1]] == pytest.approx(100 / 3)
    assert out.info["best_iteration"] == 7


def test_ordered_logit_objective_and_gradient_match_independent_likelihood(monkeypatch):
    X = np.random.default_rng(7).normal(size=(36, 2))
    categories = np.tile(np.arange(N_CATEGORIES), 2)

    class CheckedObjective(Exception):
        pass

    def inspect(objective, initial, *, method, jac, bounds, options):
        assert method == "L-BFGS-B" and jac is True
        assert options == {"maxiter": 4000, "maxfun": 20000, "maxls": 50,
                           "maxcor": 100, "gtol": 1e-7, "ftol": 0.0}
        assert bounds[:3] == [(-np.inf, np.inf)] * 3
        assert bounds[3:] == [(1e-6, np.inf)] * 16
        counts = np.bincount(categories, minlength=N_CATEGORIES) + 0.5
        cumulative = counts.cumsum()[:-1] / counts.sum()
        cuts = np.log(cumulative / (1 - cumulative))
        np.testing.assert_array_equal(initial, np.r_[0., 0., cuts[0], np.diff(cuts) + 1e-6])
        values = np.r_[0.2, -0.1, -2., np.full(16, 0.3)]
        value, gradient = objective(values)
        theta = np.r_[values[2], values[2] + np.cumsum(values[3:])]
        raw_gaps = np.log(np.expm1(values[3:] - 1e-6))
        old_theta = np.r_[values[2], values[2] + np.cumsum(np.logaddexp(0., raw_gaps) + 1e-6)]
        np.testing.assert_allclose(theta, old_theta, atol=1e-14, rtol=0)
        standardized = (X - X.mean(0)) / X.std(0)
        cdf = expit(theta[None, :] - (standardized @ values[:2])[:, None])
        pmf = np.diff(cdf, axis=1, prepend=0., append=1.)
        expected = -np.log(pmf[np.arange(len(X)), categories]).mean() + 1e-4 * (values[:2] ** 2).sum()
        assert value == pytest.approx(expected, abs=1e-14)
        numerical = np.empty_like(values)
        for j in range(len(values)):
            step = np.zeros_like(values)
            step[j] = 1e-6
            numerical[j] = (objective(values + step)[0] - objective(values - step)[0]) / 2e-6
        np.testing.assert_allclose(gradient, numerical, atol=1e-6, rtol=0)
        raise CheckedObjective

    monkeypatch.setattr(logistic_regression, "minimize", inspect)
    with pytest.raises(CheckedObjective):
        logistic_regression.fit_model(X, categories, X[:0], categories[:0], seed=1)


def test_ordered_logit_rejects_exhausted_one_iteration_budget(monkeypatch):
    rng = np.random.default_rng(4)
    X = rng.normal(size=(100, 3))
    X[:, 1] = X[:, 0] + 0.001 * X[:, 1]
    categories = np.digitize(X[:, 0], np.linspace(-2, 2, 17))
    monkeypatch.setattr(logistic_regression, "MAX_ITER", 1)
    with pytest.raises(RuntimeError, match="optimization rejected"):
        logistic_regression.fit_model(X, categories, X[:0], categories[:0], seed=1)


def test_ordered_logit_rejects_solver_success_with_large_gradient(monkeypatch):
    def unchanged(objective, initial, **kwargs):
        return SimpleNamespace(x=initial, nit=0, nfev=1, status=0, message="success")

    monkeypatch.setattr(logistic_regression, "minimize", unchanged)
    X = np.linspace(-1, 1, 40)[:, None]
    categories = np.where(X[:, 0] < 0, 2, 15)
    with pytest.raises(RuntimeError, match="optimization rejected"):
        logistic_regression.fit_model(X, categories, X[:0], categories[:0], seed=1)


def test_ordered_logit_acceptance_uses_projected_gradient_at_active_bounds(monkeypatch):
    def at_bounds(objective, initial, **kwargs):
        values = initial.copy()
        values[2:] = 1e-6
        return SimpleNamespace(x=values, nit=1, nfev=2, status=0, message="active bounds")

    def bound_gradient(loss, params):
        gradient = torch.zeros_like(params)
        gradient[2:] = 1.
        return (gradient,)

    monkeypatch.setattr(logistic_regression, "minimize", at_bounds)
    monkeypatch.setattr(logistic_regression.torch.autograd, "grad", bound_gradient)
    X, categories = np.zeros((36, 1)), np.tile(np.arange(18), 2)
    model = logistic_regression.fit_model(X, categories, X[:0], categories[:0], seed=1)
    assert model.optimization_["grad_inf_norm"] == 1.
    assert model.optimization_["projected_grad_inf_norm"] == 0.
    assert model.optimization_["converged"] is True


@pytest.mark.parametrize("category", [0, 17])
def test_ordered_logit_accepts_all_bottom_and_all_top_categories(category):
    X, categories = np.linspace(-1, 1, 24)[:, None], np.full(24, category)
    model = logistic_regression.fit_model(X, categories, X[:0], categories[:0], seed=1)
    assert model.optimization_["converged"] is True
    assert model.optimization_["projected_grad_inf_norm"] <= 1e-7
    pmf = model.predict_proba(X)
    assert np.isfinite(pmf).all() and (pmf >= 0).all()
    np.testing.assert_allclose(pmf.sum(1), 1.)
    assert pmf[:, category].min() > 0.99


def test_ordered_logit_readonly_categories_and_validation_labels_leave_fit_unchanged():
    X = np.linspace(-1, 1, 60)[:, None]
    categories = np.where(X[:, 0] < 0, 3, 12)
    readonly = categories.copy()
    readonly.flags.writeable = False
    baseline = logistic_regression.fit_model(X, categories, X[:10], categories[:10], seed=1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = logistic_regression.fit_model(X, readonly, X[:10] + 1000, 17 - categories[:10], seed=9)
    assert not any("not writable" in str(item.message) for item in caught)
    assert not readonly.flags.writeable and categories.flags.writeable
    np.testing.assert_array_equal(readonly, categories)
    np.testing.assert_array_equal(model.predict_proba(X), baseline.predict_proba(X))
    assert model.optimization_ == baseline.optimization_


@pytest.mark.parametrize("best_iteration", [0, 7])
@pytest.mark.parametrize("gains,expected", [
    ({"f0": 6., "f1": 2.}, [0.75, 0.25, 0.]), ({}, [0., 0., 0.]),
])
def test_booster_importance_uses_only_selected_rounds(best_iteration, gains, expected):
    class Booster:
        def __init__(self):
            self.best_iteration = best_iteration

        def __getitem__(self, selected):
            assert selected.start is None and selected.step is None
            assert selected.stop == best_iteration + 1
            return SimpleNamespace(get_score=lambda **kwargs: gains)

        def get_score(self, **kwargs):
            raise AssertionError("full booster contains a post-best-only f2")

    model = xgboost.BoosterPMF(Booster(), 3)
    np.testing.assert_array_equal(model.feature_importances_, expected)
