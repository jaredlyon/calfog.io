"""Tabular PMFs use training categories and validation-only early stopping."""

from types import SimpleNamespace

import numpy as np
import pytest

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


def test_logistic_scaler_is_fitted_on_training_only(prep):
    Xtr = tree_view(prep.Xtr)
    model = logistic_regression.fit_model(Xtr, prep.ctr, tree_view(prep.Xv) + 1000, prep.cv, seed=42)
    np.testing.assert_allclose(model.scaler.mean_, Xtr.mean(axis=0, dtype=np.float64))
    assert model.scaler.n_samples_seen_ == len(prep.ctr)
    assert np.all(np.diff(model.cutpoints_) > 0)


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
            best_iteration=7,
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

        def get_score(self, *, importance_type):
            assert importance_type == "gain"
            return {"f0": 2.0, "f1": 1.0}

    def train_spy(params, dtrain, **kwargs):
        calls.append(kwargs)
        np.testing.assert_array_equal(dtrain.get_label(), prep.ctr)
        np.testing.assert_allclose(dtrain.get_data().toarray(), tree_view(prep.Xtr))
        assert params["objective"] == "multi:softprob"
        assert params["num_class"] == N_CATEGORIES
        assert params["device"] == "cpu"
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
