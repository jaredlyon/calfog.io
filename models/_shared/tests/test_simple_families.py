"""Synthetic Markov recovery and causal weather-analog behavior."""

from dataclasses import replace

import numpy as np
import pandas as pd

from models._shared.evaluation import rps
from models._shared.families import analog, persistence
from models._shared.visibility import N_CATEGORIES
from models._shared.windows import Prepared, tree_view


def _prior(categories):
    counts = np.bincount(categories, minlength=N_CATEGORIES) + 0.5
    return counts / counts.sum()


def _prep(ctr, cv, ct, *, lags=None, features=None):
    categories = [np.asarray(c, dtype=int) for c in (ctr, cv, ct)]
    if features is None:
        features = [np.zeros((len(c), 24, 2), dtype=np.float32) for c in categories]
    if lags is None:
        lags = [[pd.NA] * len(c) for c in categories]
    metas = {
        split: pd.DataFrame({"lag_category": pd.array(lag, dtype="Int64")})
        for split, lag in zip(("train", "val", "test"), lags)
    }
    return Prepared(
        features[0], categories[0], features[1], categories[1], features[2], categories[2],
        metas, ["signal", "constant"],
    )


def test_persistence_learns_diagonal_transitions_and_beats_prior():
    ctr = np.repeat(np.arange(N_CATEGORIES), 20)
    cv = np.tile(np.arange(N_CATEGORIES), 2)
    ct = np.array([2, 7, 14, 17])
    prep = _prep(ctr, cv, ct, lags=[ctr, cv, [2, 7, 14, pd.NA]])
    output = persistence.fit_predict(prep, cnn_cfg=None, seed=42)
    prior = _prior(ctr)
    expected = (20 * np.eye(N_CATEGORIES)[cv] + prior) / 21
    np.testing.assert_allclose(output.pmf_val, expected)
    np.testing.assert_allclose(output.pmf_test.sum(axis=1), 1)
    assert (output.pmf_test >= 0).all()
    assert rps(cv, output.pmf_val) < 0.1 * rps(cv, np.tile(prior, (len(cv), 1)))
    assert output.info == {"alpha": 1, "lag_coverage_test": 0.75, "persistence_exact_test": 1.0}
    assert output.importance == {}
    np.testing.assert_array_equal(output.pmf_test[-1], prior)


def test_persistence_missing_training_lags_do_not_contribute_transitions():
    prep = _prep(
        [2, 2, 14, 14, 14], [2, 14], [2, 14],
        lags=[[pd.NA] * 5, [2, pd.NA], [pd.NA, pd.NA]],
    )
    output = persistence.fit_predict(prep, cnn_cfg=None, seed=42)
    expected = np.tile(_prior(prep.ctr), (2, 1))
    np.testing.assert_allclose(output.pmf_val, expected)
    np.testing.assert_array_equal(output.pmf_test, expected)
    assert output.info["lag_coverage_test"] == 0
    assert output.info["persistence_exact_test"] is None


def test_persistence_selects_large_alpha_by_validation_not_test():
    ctr = np.tile([2, 14], 8)
    lag_val = np.tile([2, 14], 4)
    cv = np.tile([14, 2], 4)
    prep = _prep(ctr, cv, [2, 14], lags=[ctr, lag_val, [2, 14]])
    output = persistence.fit_predict(prep, cnn_cfg=None, seed=42)
    prior = _prior(ctr)
    expected = (8 * np.eye(N_CATEGORIES)[lag_val] + 100 * prior) / 108
    np.testing.assert_allclose(output.pmf_val, expected)
    assert output.info["alpha"] == 100
    for alpha in (1, 10):
        alternative = (8 * np.eye(N_CATEGORIES)[lag_val] + alpha * prior) / (8 + alpha)
        assert rps(cv, output.pmf_val) < rps(cv, alternative)
    changed_test = persistence.fit_predict(replace(prep, ct=np.array([14, 2])), cnn_cfg=None, seed=42)
    assert changed_test.info["alpha"] == 100
    np.testing.assert_array_equal(changed_test.pmf_test, output.pmf_test)
    assert output.info["persistence_exact_test"] == 1
    assert changed_test.info["persistence_exact_test"] == 0


def _analog_prep():
    signals = (np.arange(240), np.array([105, 134]), np.array([104, 135]))
    features = [
        np.repeat(np.column_stack((signal, np.zeros(len(signal))))[:, None, :], 24, axis=1)
        .astype(np.float32)
        for signal in signals
    ]
    categories = [np.where(signal < 120, 2, 14) for signal in signals]
    return _prep(*categories, features=features)


def test_analog_selects_smallest_good_neighborhood_and_recovers_categories():
    prep = _analog_prep()
    output = analog.fit_predict(prep, cnn_cfg=None, seed=42)
    prior = _prior(prep.ctr)
    assert output.info == {"k": 25}
    assert output.importance == {}
    expected = (25 * np.eye(N_CATEGORIES)[prep.ct] + prior) / 26
    np.testing.assert_allclose(output.pmf_test, expected)
    assert (output.pmf_test[np.arange(len(prep.ct)), prep.ct] > 0.98).all()
    assert rps(prep.cv, output.pmf_val) < 0.01 * rps(prep.cv, np.tile(prior, (len(prep.cv), 1)))
    np.testing.assert_allclose(output.pmf_test.sum(axis=1), 1)
    assert (output.pmf_test >= 0).all()
    changed_test = analog.fit_predict(replace(prep, ct=prep.ct[::-1]), cnn_cfg=None, seed=42)
    assert changed_test.info == output.info
    np.testing.assert_array_equal(changed_test.pmf_test, output.pmf_test)


def test_analog_selects_large_k_by_validation_not_test():
    original = _analog_prep()
    prep = replace(
        original, ctr=np.full(len(original.ctr), 14), cv=np.full(len(original.cv), 14),
        ct=np.full(len(original.ct), 2),
    )
    output = analog.fit_predict(prep, cnn_cfg=None, seed=42)
    assert output.info == {"k": 200}
    prior = _prior(prep.ctr)
    expected = (200 * np.eye(N_CATEGORIES)[prep.cv] + prior) / 201
    np.testing.assert_allclose(output.pmf_val, expected)
    for k in (25, 50, 100):
        alternative = (k * np.eye(N_CATEGORIES)[prep.cv] + prior) / (k + 1)
        assert rps(prep.cv, output.pmf_val) < rps(prep.cv, alternative)


def test_analog_ignores_std_min_max_blocks():
    prep = _analog_prep()
    changed = []
    for X in (prep.Xtr, prep.Xv, prep.Xt):
        perturbed = X.copy()
        # Opposite interior offsets preserve latest and mean exactly, not the other summaries.
        offsets = (np.arange(len(X), dtype=np.float32) + 1) * 512
        perturbed[:, 0, 0] += offsets
        perturbed[:, 1, 0] -= offsets
        np.testing.assert_array_equal(tree_view(X)[:, :4], tree_view(perturbed)[:, :4])
        assert not np.array_equal(tree_view(X)[:, 4:], tree_view(perturbed)[:, 4:])
        changed.append(perturbed)
    original = analog.fit_predict(prep, cnn_cfg=None, seed=42)
    perturbed = analog.fit_predict(
        replace(prep, Xtr=changed[0], Xv=changed[1], Xt=changed[2]), cnn_cfg=None, seed=42,
    )
    assert perturbed.info == original.info
    np.testing.assert_array_equal(perturbed.pmf_val, original.pmf_val)
    np.testing.assert_array_equal(perturbed.pmf_test, original.pmf_test)


def test_analog_clamps_and_deduplicates_neighbors_for_small_training_sets(monkeypatch):
    from sklearn.neighbors import NearestNeighbors

    queries = []

    class RecordedNeighbors(NearestNeighbors):
        def kneighbors(self, X, *args, **kwargs):
            queries.append((len(X), X.dtype, self.n_neighbors))
            return super().kneighbors(X, *args, **kwargs)

    monkeypatch.setattr(analog, "NearestNeighbors", RecordedNeighbors)
    original = _analog_prep()
    prep = replace(original, Xtr=original.Xtr[:12], ctr=original.ctr[:12])
    output = analog.fit_predict(prep, cnn_cfg=None, seed=42)
    prior = _prior(prep.ctr)
    expected = (np.bincount(prep.ctr, minlength=N_CATEGORIES) + prior) / 13
    np.testing.assert_allclose(output.pmf_test, np.tile(expected, (len(prep.ct), 1)))
    assert output.info == {"k": 12}
    assert queries == [(len(prep.cv), np.dtype("float32"), 12), (len(prep.ct), np.dtype("float32"), 12)]
