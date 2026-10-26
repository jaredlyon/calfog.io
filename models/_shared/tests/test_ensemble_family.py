import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from models._shared.families import FamilyOutput
from models._shared.families import ensemble
from models._shared.visibility import N_CATEGORIES
from models._shared.windows import Prepared, tree_feature_names, tree_view


def _windows(ids):
    rows = np.column_stack((ids, np.asarray(ids) % 2)).astype(np.float32)
    return np.repeat(rows[:, None, :], 24, axis=1)


def _categories(ids):
    return np.where(np.asarray(ids) % 2, 17, 2)


def _pmf(stage, ids):
    weights = np.arange(1, N_CATEGORIES + 1)[None, :] + stage + np.asarray(ids)[:, None] / 10000
    return weights / weights.sum(axis=1, keepdims=True)


class _ForbiddenTestLabels:
    def __array__(self, *args, **kwargs):
        raise AssertionError("ensemble fitting must not read test labels")

    def __len__(self):
        raise AssertionError("ensemble fitting must not read test labels")


@pytest.fixture
def prep():
    return Prepared(
        Xtr=_windows(np.arange(100)), ctr=_categories(np.arange(100)),
        Xv=_windows(np.arange(100, 110)), cv=_categories(np.arange(100, 110)),
        Xt=_windows(np.arange(110, 120)), ct=_ForbiddenTestLabels(),
        metas={}, cols=["weather_a", "weather_b"],
    )


@pytest.fixture
def stage_spies(monkeypatch, prep):
    n_features = tree_view(prep.Xtr).shape[1]
    n_cumulative = N_CATEGORIES - 1
    predictions, fits, cnn_calls = {}, [], []

    def check_upstream(X, stage, *, cnn=False):
        ids = X[:, 0, 0].astype(int) if cnn else X[:, 0].astype(int)
        upstream = X[:, 0, -3 * n_cumulative:] if cnn else X[:, n_features:]
        assert np.isfinite(X).all()
        assert upstream.shape[1] == stage * n_cumulative
        for previous in range(stage):
            expected = _pmf(previous, ids).cumsum(axis=1)[:, :-1]
            np.testing.assert_allclose(upstream[:, previous * n_cumulative:(previous + 1) * n_cumulative],
                                       expected, rtol=1e-6)
            for row in ids:
                fit_ids = predictions[previous, row]
                assert row not in fit_ids
                assert max(fit_ids) < row
        if cnn:
            np.testing.assert_array_equal(X[:, :, -3 * n_cumulative:],
                                          np.repeat(upstream[:, None, :], 24, axis=1))
        return ids

    def fitter(stage):
        def fit_model(Xtr, ctr, Xv, cv, *, seed):
            ids = check_upstream(Xtr, stage)
            np.testing.assert_array_equal(ctr, _categories(ids))
            val_ids = Xv[:, 0].astype(int)
            if len(Xv):
                check_upstream(Xv, stage)
            np.testing.assert_array_equal(cv, _categories(val_ids))
            assert set(ids).isdisjoint(val_ids)
            fits.append((stage, ids.copy(), val_ids.copy()))
            importance = np.zeros(Xtr.shape[1])
            importance[stage] = 1
            importance[n_features:] = 1000

            def predict_proba(X):
                row_ids = X[:, 0].astype(int)
                assert set(row_ids).isdisjoint(ids)
                assert max(ids) < min(row_ids)
                for row in row_ids:
                    predictions[stage, row] = set(ids)
                return _pmf(stage, row_ids)

            def calibrated(X):
                assert len(cv), "OOF RF stacking must call predict_proba_raw"
                predict_proba(X)
                return _pmf(stage + 4, X[:, 0].astype(int))

            return SimpleNamespace(
                predict_proba=calibrated if stage == 1 else predict_proba,
                predict_proba_raw=predict_proba, feature_importances_=importance,
                coef_=importance,
            )
        return fit_model

    for stage, name in enumerate(("fit_lr", "fit_rf", "fit_xgb")):
        monkeypatch.setattr(ensemble, name, fitter(stage))

    def train_cnn(Xtr, ctr, Xv, cv, config, *, device, seed):
        ids = check_upstream(Xtr, 3, cnn=True)
        np.testing.assert_array_equal(ctr, _categories(ids))
        check_upstream(Xv, 3, cnn=True)
        cnn_calls.append(ids.copy())
        assert Xtr.shape[1:] == (24, prep.Xtr.shape[2] + 3 * n_cumulative)
        assert device == "cpu"
        return (SimpleNamespace(state_dict=lambda: {"weight": torch.tensor([1.0])}),
                {"cnn_best_val_nll": 0.75, "cnn_best_val_ap": 0.5, "cnn_epochs": 4})

    def predict_cnn(model, X, device):
        check_upstream(X, 3, cnn=True)
        return _pmf(3, np.zeros(len(X)))

    monkeypatch.setattr(ensemble, "train_cnn", train_cnn)
    monkeypatch.setattr(ensemble, "predict_cnn", predict_cnn)
    monkeypatch.setattr(ensemble.torch.cuda, "is_available", lambda: False)
    return fits, cnn_calls


def test_staircase_never_trains_on_in_sample_upstream_predictions(prep, stage_spies):
    ensemble.fit_predict(prep, cnn_cfg={"channels": 4}, seed=17)
    fits, cnn_calls = stage_spies
    assert [sum(stage == call[0] for call in fits) for stage in range(3)] == [5, 4, 3]
    np.testing.assert_array_equal(cnn_calls[0], np.arange(60, 100))
    for stage in range(3):
        refit = [call for call in fits if call[0] == stage][-1]
        np.testing.assert_array_equal(refit[1], np.arange(20 * stage, 100))
    for _, train_ids, val_ids in [call for call in fits if call[0] == 2][:-1]:
        assert max(train_ids) < min(val_ids)
        assert max(val_ids) < 80
        assert len(val_ids) == int(np.ceil(0.15 * (len(train_ids) + len(val_ids))))
    np.testing.assert_array_equal(fits[-1][2], np.arange(100, 110))


def test_exact_pmf_blend_and_family_output(prep, stage_spies):
    result = ensemble.fit_predict(prep, cnn_cfg={"channels": 4}, seed=17)
    assert isinstance(result, FamilyOutput)
    assert ensemble.NEEDS_CNN_CFG is True
    for actual, ids in ((result.pmf_val, np.arange(100, 110)),
                        (result.pmf_test, np.arange(110, 120))):
        expected = sum(weight * _pmf(stage + 4 if stage == 1 else stage, ids)
                       for stage, weight in enumerate((0.1, 0.2, 0.3)))
        expected += 0.4 * _pmf(3, np.zeros(len(ids)))
        np.testing.assert_allclose(actual, expected, rtol=1e-7)
        assert actual.shape == (10, N_CATEGORIES)
        assert np.isfinite(actual).all()
        assert (actual >= 0).all()
        np.testing.assert_allclose(actual.sum(axis=1), 1)
    names = tree_feature_names(prep.cols)
    assert set(result.importance) == set(names)
    assert sum(result.importance.values()) == pytest.approx(100)
    assert result.importance[names[0]] == pytest.approx(100 / 6)
    assert result.importance[names[1]] == pytest.approx(200 / 6)
    assert result.importance[names[2]] == pytest.approx(300 / 6)
    assert all(result.importance[name] == 0 for name in names[3:])
    assert result.info["inner_blocks"] == 5
    assert result.info["cnn_train_rows"] == 40
    assert result.info["cnn_best_val_nll"] == 0.75
    assert result.info["blend_weights"] == [0.1, 0.2, 0.3, 0.4]
    json.dumps(result.info)
    assert torch.equal(result.state["weight"], torch.tensor([1.0]))


def test_ensemble_requires_cnn_configuration(prep):
    with pytest.raises(ValueError, match="cnn_cfg"):
        ensemble.fit_predict(prep, cnn_cfg=None, seed=17)


def test_ensemble_rejects_insufficient_staircase_rows(prep):
    prep.Xtr, prep.ctr = prep.Xtr[:9], prep.ctr[:9]
    with pytest.raises(ValueError, match="ten chronological training rows"):
        ensemble.fit_predict(prep, cnn_cfg={}, seed=17)


@pytest.mark.parametrize(
    "start,stop,label,message",
    [(0, 20, 17, "logistic_regression block 2 OOF"),
     (20, 40, 17, "random_forest block 3 OOF"),
     (20, 40, 2, "random_forest block 3 OOF"),
     (40, 57, 17, "xgboost block 4 OOF"),
     (60, 100, 17, "temporal_cnn block 4..5 refit")],
)
def test_single_category_staircase_fails_before_any_fit(prep, monkeypatch, start, stop, label, message):
    _assert_preflight_rejects(prep, monkeypatch, start, stop, label, message)


@pytest.mark.parametrize("categories", [(1, 4), (6, 17)])
def test_multiple_categories_on_one_fog_side_are_not_estimable(prep, monkeypatch, categories):
    _assert_preflight_rejects(prep, monkeypatch, 0, 20, np.tile(categories, 10),
                             "logistic_regression block 2 OOF")


def _assert_preflight_rejects(prep, monkeypatch, start, stop, label, message):
    from models._shared.study import NotEstimable

    prep.ctr[start:stop] = label
    calls = []

    def unexpected_fit(*args, **kwargs):
        calls.append(True)
        raise AssertionError("unestimable staircase must fail before fitting")

    for name in ("fit_lr", "fit_rf", "fit_xgb", "train_cnn"):
        monkeypatch.setattr(ensemble, name, unexpected_fit)
    with pytest.raises(NotEstimable, match=message):
        ensemble.fit_predict(prep, cnn_cfg={}, seed=17)
    assert calls == []
