"""Pooled predictors preserve airport identity, missing lags, and neighbor signal."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from models._shared.data import SITES
from models._shared.evaluation import nll, rps
from models._shared.families import xgboost, xgboost_pooled
from models._shared.visibility import N_CATEGORIES
from models._shared.windows import Prepared, tree_view


def test_extended_features_names_sites_and_missing_values(monkeypatch):
    site_order = ["location_10", "location_6", "location_9", "location_7", "location_8"]
    meta = pd.DataFrame({
        "site": site_order,
        "lag_category": pd.array([pd.NA, 2, 7, 11, 17], dtype="Int64"),
        "nbr_min_cat": [np.nan, 1, 4, 6, 12],
        "nbr_mean_cat": [np.nan, 3, 8, 10, 15],
        "nbr_fog_count": [0, 2, 1, 0, 0],
        "nbr_reporting": [0, 4, 3, 2, 1],
    })
    arrays = [np.arange(5 * 24 * 2, dtype=np.float32).reshape(5, 24, 2) + i
              for i in range(3)]
    metas = {split: meta.iloc[np.roll(np.arange(5), i)].reset_index(drop=True)
             for i, split in enumerate(("train", "val", "test"))}
    categories = np.arange(5)
    prep = Prepared(arrays[0], categories, arrays[1], categories + 1,
                    arrays[2], categories + 2, metas, ["temperature", "humidity"])
    names = [
        "latest_temperature", "latest_humidity", "mean_temperature", "mean_humidity",
        "std_temperature", "std_humidity", "min_temperature", "min_humidity",
        "max_temperature", "max_humidity", "site_location_6", "site_location_7",
        "site_location_8", "site_location_9", "site_location_10", "lag_category",
        "nbr_min_cat", "nbr_mean_cat", "nbr_fog_count", "nbr_reporting",
    ]
    expected = []
    for X, split in zip(arrays, ("train", "val", "test")):
        rows = metas[split]
        one_hot = np.array([[int(site == known) for known in SITES] for site in rows.site])
        lag_features = rows[[
            "lag_category", "nbr_min_cat", "nbr_mean_cat", "nbr_fog_count", "nbr_reporting",
        ]].to_numpy(dtype=np.float32, na_value=np.nan)
        expected.append(np.concatenate([tree_view(X), one_hot, lag_features], axis=1))
    predicted = []
    calls = []

    def predict(X):
        predicted.append(X)
        return np.full((len(X), N_CATEGORIES), 1 / N_CATEGORIES)

    weights = np.arange(1, len(names) + 1, dtype=float)
    weights /= weights.sum()

    def fit_spy(Xtr, ctr, Xv, cv, *, seed):
        calls.append(seed)
        np.testing.assert_allclose(Xtr, expected[0], equal_nan=True)
        np.testing.assert_allclose(Xv, expected[1], equal_nan=True)
        np.testing.assert_array_equal(ctr, prep.ctr)
        np.testing.assert_array_equal(cv, prep.cv)
        assert Xtr.dtype == np.float32
        assert Xv.dtype == np.float32
        return SimpleNamespace(
            predict_proba=predict, feature_importances_=weights, best_iteration=7,
        )

    monkeypatch.setattr(xgboost_pooled, "fit_model", fit_spy)
    out = xgboost_pooled.fit_predict(prep, cnn_cfg=None, seed=31)
    assert calls == [31]
    assert xgboost_pooled.POOLED is True
    assert xgboost_pooled.NEEDS_CNN_CFG is False
    assert list(out.importance) == names
    np.testing.assert_allclose(list(out.importance.values()), 100 * weights)
    assert sum(out.importance.values()) == pytest.approx(100)
    assert out.info["best_iteration"] == 7
    assert len(predicted) == 2
    for actual, wanted in zip(predicted, expected[1:]):
        np.testing.assert_allclose(actual, wanted, equal_nan=True)
        np.testing.assert_array_equal(np.isnan(actual), np.isnan(wanted))


def test_pooled_booster_uses_neighbor_signal(monkeypatch):
    train = xgboost.xgb.train

    def short_cpu_train(params, dtrain, **kwargs):
        kwargs["num_boost_round"] = 40
        kwargs["early_stopping_rounds"] = 8
        return train(dict(params, device="cpu", nthread=1), dtrain, **kwargs)

    monkeypatch.setattr(xgboost.xgb, "train", short_cpu_train)

    def partition(n):
        categories = np.tile([2, 14], n // 2)
        meta = pd.DataFrame({
            "site": np.resize(tuple(SITES), n),
            "lag_category": pd.array([pd.NA] * n, dtype="Int64"),
            "nbr_min_cat": categories.astype(float),
            "nbr_mean_cat": np.full(n, 14.0),
            "nbr_fog_count": np.zeros(n, dtype=int),
            "nbr_reporting": np.ones(n, dtype=int),
        })
        return np.zeros((n, 24, 1), dtype=np.float32), categories, meta

    Xtr, ctr, mtr = partition(200)
    Xv, cv, mv = partition(60)
    Xt, ct, mt = partition(60)
    prep = Prepared(Xtr, ctr, Xv, cv, Xt, ct,
                    {"train": mtr, "val": mv, "test": mt}, ["constant_weather"])
    out = xgboost_pooled.fit_predict(prep, cnn_cfg=None, seed=42)
    names = list(out.importance)
    keep = np.arange(len(names)) != names.index("nbr_min_cat")
    baseline = xgboost.fit_model(
        xgboost_pooled._features(Xtr, mtr)[:, keep], ctr,
        xgboost_pooled._features(Xv, mv)[:, keep], cv, seed=42,
    )
    baseline_pmf = baseline.predict_proba(xgboost_pooled._features(Xt, mt)[:, keep])
    assert nll(ct, out.pmf_test) < 0.6 * nll(ct, baseline_pmf)
    assert rps(ct, out.pmf_test) < 0.5 * rps(ct, baseline_pmf)
    np.testing.assert_array_equal(out.pmf_test.argmax(axis=1), ct)
    assert out.importance["nbr_min_cat"] > 0
    for probabilities in (out.pmf_val, out.pmf_test):
        assert probabilities.shape == (60, N_CATEGORIES)
        assert np.isfinite(probabilities).all()
        assert (probabilities >= 0).all()
        np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
