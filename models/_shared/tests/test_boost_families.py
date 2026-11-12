"""Small synthetic tests for ordinal/AFT probabilities and forest calibration."""

import math
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from models._shared.evaluation import rps
from models._shared.families import random_forest, xgboost_aft, xgboost_ordinal
from models._shared.families.xgboost import BASE_PARAMS
from models._shared.visibility import N_CATEGORIES

FROZEN_AFT_BOUNDS = np.r_[
    np.array([0., .25, .5, .75, 1., 1.25, 1.5, 1.75, 2., 2.5, 3., 4., 5., 6., 7., 8., 9., 10.]) * 1609.344,
    np.inf,
]


class _Booster:
    def __init__(self, best_iteration, gains, predict):
        self.best_iteration = best_iteration
        self.gains = gains
        self.predict = predict

    def __getitem__(self, selected):
        assert selected.start is None and selected.step is None
        assert selected.stop == self.best_iteration + 1

        def get_score(*, importance_type):
            assert importance_type == "gain"
            return self.gains

        return SimpleNamespace(get_score=get_score)

    def get_score(self, **kwargs):
        raise AssertionError("importance must exclude post-best trees")


def test_ordinal_monotonizes_crossing_thresholds_and_keeps_constant_tails(monkeypatch):
    Xtr = np.arange(4, dtype=np.float32)[:, None]
    ctr = np.array([2, 5, 2, 5])
    Xv, cv = Xtr[:2], np.array([2, 5])
    probabilities = iter((0.8, 0.2, 0.7))
    calls = []

    def train(params, data, **kwargs):
        assert all(params[key] == value for key, value in BASE_PARAMS.items())
        assert params["objective"] == "binary:logistic"
        assert params["eval_metric"] == "logloss"
        assert kwargs["early_stopping_rounds"] == 40
        np.testing.assert_array_equal(data.get_label(), [1, 0, 1, 0])
        np.testing.assert_array_equal(kwargs["evals"][0][0].get_label(), [1, 0])
        probability = next(probabilities)
        calls.append(probability)
        def predict(data, *, iteration_range):
            assert iteration_range == (0, 3)
            return np.full(data.num_row(), probability)

        return _Booster(2, {"f0": 4.0}, predict)

    monkeypatch.setattr(xgboost_ordinal.xgb, "train", train)
    model = xgboost_ordinal.fit_model(Xtr, ctr, Xv, cv, seed=7)
    cumulative = model.predict_cumulative(Xv)
    assert calls == [0.8, 0.2, 0.7]
    assert np.all(np.diff(cumulative, axis=1) >= 0)
    np.testing.assert_allclose(cumulative[0], [0, 0, 0.8, 0.8, 0.8] + [1] * 12)
    pmf = model.predict_proba(Xv)
    np.testing.assert_allclose(pmf[:, 2], 0.8)
    np.testing.assert_allclose(pmf[:, 5], 0.2)
    assert np.all(pmf >= 0)
    np.testing.assert_allclose(pmf.sum(axis=1), 1)
    np.testing.assert_allclose(model.feature_importances_, [1])


def test_ordinal_single_class_thresholds_need_no_boosters(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("single-class thresholds must be constant")

    monkeypatch.setattr(xgboost_ordinal.xgb, "train", forbidden)
    X = np.zeros((3, 1), dtype=np.float32)
    model = xgboost_ordinal.fit_model(X, np.full(3, 7), X[:1], np.array([7]), seed=3)
    expected = np.zeros((1, N_CATEGORIES))
    expected[:, 7] = 1
    np.testing.assert_array_equal(model.predict_proba(X[:1]), expected)
    np.testing.assert_array_equal(model.feature_importances_, [0])


def test_aft_normal_interval_mass_matches_hand_calculation():
    median, scale = 1800.0, 0.5

    def cdf(bound):
        if bound == 0:
            return 0.0
        if math.isinf(bound):
            return 1.0
        return 0.5 * (1 + math.erf(math.log(bound / median) / (scale * math.sqrt(2))))

    expected = np.array([cdf(upper) - cdf(lower)
                         for lower, upper in zip(FROZEN_AFT_BOUNDS[:-1], FROZEN_AFT_BOUNDS[1:])])
    actual = xgboost_aft.median_to_pmf(np.array([median]), "normal", scale)[0]
    np.testing.assert_allclose(actual, expected, atol=2e-16)
    assert np.all(actual >= 0)
    assert actual.sum() == pytest.approx(1)
    assert actual[0] > 0 and actual[-1] > 0


@pytest.mark.parametrize("distribution", ["normal", "logistic"])
def test_aft_extreme_predicted_medians_give_valid_pmf(distribution):
    # xgboost can return an overflowed (inf) or zero median; neither may produce NaN mass
    pmf = xgboost_aft.median_to_pmf(np.array([np.inf, 0.0, 1e30]), distribution, 1.0)
    assert np.isfinite(pmf).all()
    np.testing.assert_allclose(pmf.sum(axis=1), 1)
    assert pmf[0, -1] > 0.99 and pmf[1, 0] > 0.99  # far above / below the scale land in the end bins


def test_aft_grid_uses_validation_rps_and_interval_bounds(monkeypatch):
    Xtr = np.arange(4, dtype=np.float32)[:, None]
    ctr = np.array([0, 1, 16, 17])
    Xv = np.array([[10.0], [11.0]], dtype=np.float32)
    calls = []

    def train(params, data, **kwargs):
        assert all(params[key] == value for key, value in BASE_PARAMS.items())
        assert params["objective"] == "survival:aft"
        assert params["eval_metric"] == "aft-nloglik"
        assert kwargs["early_stopping_rounds"] == 40
        np.testing.assert_allclose(data.get_float_info("label_lower_bound"),
                                   FROZEN_AFT_BOUNDS[:-1][ctr], rtol=1e-6)
        np.testing.assert_allclose(data.get_float_info("label_upper_bound"),
                                   FROZEN_AFT_BOUNDS[1:][ctr], rtol=1e-6)
        val = kwargs["evals"][0][0]
        assert val.num_row() == len(Xv)
        np.testing.assert_allclose(val.get_float_info("label_lower_bound"),
                                   FROZEN_AFT_BOUNDS[:-1][cv], rtol=1e-6)
        np.testing.assert_allclose(val.get_float_info("label_upper_bound"),
                                   FROZEN_AFT_BOUNDS[1:][cv], rtol=1e-6)
        distribution, scale = params["aft_loss_distribution"], params["aft_loss_distribution_scale"]
        calls.append((distribution, scale))
        median = 1e8 if (distribution, scale) == ("logistic", 1.0) else 1.0
        def predict(data, *, iteration_range):
            assert iteration_range == (0, 7)
            return np.full(data.num_row(), median)

        return _Booster(6, {"f0": 2.0}, predict)

    monkeypatch.setattr(xgboost_aft.xgb, "train", train)
    cv = np.full(2, 17)
    selected = xgboost_aft.fit_model(Xtr, ctr, Xv, cv, seed=7)
    assert calls == [(d, s) for d in ("normal", "logistic") for s in (0.5, 1.0, 1.5)]
    assert (selected.distribution, selected.scale, selected.best_iteration) == ("logistic", 1.0, 6)
    assert rps(cv, selected.predict_proba(Xv)) < 1e-6
    calls.clear()
    cv = np.zeros(2, dtype=int)
    selected_low = xgboost_aft.fit_model(Xtr, ctr, Xv, cv, seed=7)
    assert (selected_low.distribution, selected_low.scale) == ("normal", 0.5)


def test_aft_frozen_latent_bounds_are_independent_of_report_support(monkeypatch):
    from models._shared import visibility

    np.testing.assert_array_equal(xgboost_aft.AFT_LATENT_BOUNDS, FROZEN_AFT_BOUNDS)
    X, categories = np.zeros((18, 1), dtype=np.float32), np.arange(18)
    before = xgboost_aft.median_to_pmf(np.array([1609.344]), "logistic", 1.)
    monkeypatch.setattr(visibility, "CATEGORY_LOWER_M", (0.,) * 18)
    monkeypatch.setattr(visibility, "CATEGORY_UPPER_M", (np.inf,) * 18)
    data = xgboost_aft._interval_data(X, categories)
    np.testing.assert_array_equal(data.get_float_info("label_lower_bound"),
                                  FROZEN_AFT_BOUNDS[:-1].astype(np.float32))
    np.testing.assert_array_equal(data.get_float_info("label_upper_bound"),
                                  FROZEN_AFT_BOUNDS[1:].astype(np.float32))
    after = xgboost_aft.median_to_pmf(np.array([1609.344]), "logistic", 1.)
    np.testing.assert_array_equal(after, before)
    assert after[0, :5].sum() == pytest.approx(5 / 9)
    assert FROZEN_AFT_BOUNDS[2] > FROZEN_AFT_BOUNDS[1]


@pytest.mark.parametrize("best_iteration", [0, 6])
@pytest.mark.parametrize("gains,expected", [
    ({"f0": 6., "f1": 2.}, [0.75, 0.25, 0.]), ({}, [0., 0., 0.]),
])
def test_aft_importance_uses_selected_rounds(best_iteration, gains, expected):
    booster = _Booster(best_iteration, gains, None)
    model = xgboost_aft.AFTBoosterPMF(booster, 3, "normal", .5)
    np.testing.assert_array_equal(model.feature_importances_, expected)


@pytest.mark.parametrize("best_iteration", [0, 6])
@pytest.mark.parametrize("empty", [False, True])
def test_ordinal_importance_uses_selected_rounds_and_ignores_constants(best_iteration, empty):
    first = _Booster(best_iteration, {} if empty else {"f0": 6., "f1": 2.}, None)
    second = _Booster(best_iteration, {} if empty else {"f1": 2.}, None)
    model = xgboost_ordinal.OrdinalBoosterPMF([first, second] + [1.] * 15, 3)
    np.testing.assert_allclose(model.feature_importances_, [0., 0., 0.] if empty else [.6, .4, 0.])


@pytest.fixture
def small_forest(monkeypatch):
    original_fit = random_forest.CategoryForest.fit

    def fit(self, X, y):
        self.set_params(n_estimators=12, n_jobs=1)
        return original_fit(self, X, y)

    monkeypatch.setattr(random_forest.CategoryForest, "fit", fit)
    X = np.r_[np.full(36, -1.0), np.full(36, 1.0)][:, None]
    categories = np.where(X[:, 0] < 0, 2, 15)
    return X, categories


def _nll(categories, pmf):
    return -np.log(np.clip(pmf[np.arange(len(categories)), categories], 1e-15, 1)).mean()


@pytest.mark.parametrize("use_scipy", [True, False])
def test_forest_calibration_lowers_validation_nll_and_survives_pickle(small_forest, monkeypatch, use_scipy):
    if not use_scipy:
        monkeypatch.setattr(random_forest, "minimize", None)
    Xtr, ctr = small_forest
    Xv = np.tile([[-1.0], [1.0]], (15, 1))
    cv = np.where(Xv[:, 0] < 0, 2, 15)
    cv[::3] = 17 - cv[::3]
    model = random_forest.fit_model(Xtr, ctr, Xv, cv, seed=2)
    raw, calibrated = model.predict_proba_raw(Xv), model.predict_proba(Xv)
    clipped = np.clip(raw, 1e-6, 1)
    clipped /= clipped.sum(axis=1, keepdims=True)
    # beat clipping alone (T=1, w=0), so only a fitted temperature/prior weight can pass
    assert _nll(cv, calibrated) < 0.7 * _nll(cv, clipped)
    assert 0.05 <= model.temperature_ <= 20
    assert 0 <= model.prior_weight_ <= 0.5
    expected_prior = (np.bincount(ctr, minlength=N_CATEGORIES) + 0.5) / (len(ctr) + 9)
    np.testing.assert_allclose(model.prior_, expected_prior)
    scaled = np.clip(raw, 1e-6, 1) ** (1 / model.temperature_)
    scaled /= scaled.sum(axis=1, keepdims=True)
    expected = (1 - model.prior_weight_) * scaled + model.prior_weight_ * expected_prior
    np.testing.assert_allclose(calibrated, expected)
    assert np.all(calibrated >= 0)
    np.testing.assert_allclose(calibrated.sum(axis=1), 1)
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(restored.predict_proba(Xv), calibrated)


def test_forest_without_validation_keeps_exact_raw_probabilities(small_forest):
    Xtr, ctr = small_forest
    model = random_forest.fit_model(Xtr, ctr, Xtr[:0], ctr[:0], seed=2)
    assert model.temperature_ == 1
    assert model.prior_weight_ == 0
    np.testing.assert_array_equal(model.predict_proba(Xtr), model.predict_proba_raw(Xtr))
