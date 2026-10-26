"""Small CPU-only EBM checks for category mapping, sampling, and explanations."""

import numpy as np
import pytest

pytest.importorskip("interpret")
from interpret import glassbox

from models._shared.families import ebm
from models._shared.visibility import N_CATEGORIES
from models._shared.windows import Prepared, tree_feature_names, tree_view


def prepared(categories=(1, 6, 17)):
    rng = np.random.default_rng(42)

    def partition(n):
        c = np.resize(np.asarray(categories), n)
        rng.shuffle(c)
        X = rng.normal(0, 0.05, (n, 24, 2))
        X[:, :, 0] += c[:, None] / 4
        return X.astype(np.float32), c

    Xtr, ctr = partition(96)
    Xv, cv = partition(18)
    Xt, ct = partition(18)
    return Prepared(Xtr, ctr, Xv, cv, Xt, ct, {}, ["signal", "noise"])


@pytest.fixture
def tiny_ebms(monkeypatch):
    classifier = glassbox.ExplainableBoostingClassifier
    fitted = []

    class TinyEBM(classifier):
        def fit(self, X, y, **kwargs):
            self.fit_X_ = X.copy()
            self.fit_y_ = y.copy()
            return super().fit(X, y, **kwargs)

    def short_model(**kwargs):
        assert kwargs["outer_bags"] == 2
        assert kwargs["interactions"] == 0
        assert kwargs["n_jobs"] == -1
        kwargs.update(
            max_rounds=16, early_stopping_rounds=0, learning_rate=0.2,
            smoothing_rounds=0, greedy_ratio=0, min_samples_leaf=2,
            max_bins=16, n_jobs=1,
        )
        model = TinyEBM(**kwargs)
        fitted.append(model)
        return model

    monkeypatch.setattr(glassbox, "ExplainableBoostingClassifier", short_model)
    return fitted


@pytest.mark.parametrize("categories", [(2,), (1, 17), (1, 6, 17)])
def test_category_padding_preserves_class_probabilities(categories, tiny_ebms):
    prep = prepared(categories)
    out = ebm.fit_predict(prep, cnn_cfg=None, seed=42)
    model = tiny_ebms[-1]
    assert ebm.NEEDS_CNN_CFG is False
    assert out.info["n_train_used"] == len(prep.ctr)
    assert out.info["interactions"] == 0
    for X, pmf in [(prep.Xv, out.pmf_val), (prep.Xt, out.pmf_test)]:
        assert pmf.shape == (len(X), N_CATEGORIES)
        assert np.isfinite(pmf).all()
        assert ((0 <= pmf) & (pmf <= 1)).all()
        np.testing.assert_allclose(pmf.sum(axis=1), 1)
        np.testing.assert_allclose(
            pmf[:, model.classes_.astype(int)], model.predict_proba(tree_view(X))
        )
        absent = np.setdiff1d(np.arange(N_CATEGORIES), model.classes_)
        assert not pmf[:, absent].any()
    if len(categories) > 1:
        # A misplaced class column or a constant-PMF fallback fails this check.
        high = out.pmf_test[prep.ct == max(categories), max(categories)].mean()
        low = out.pmf_test[prep.ct == min(categories), max(categories)].mean()
        assert high > low + 0.1


def test_subsample_cap_is_seeded_uniform_and_keeps_labels_aligned(monkeypatch, tiny_ebms):
    prep = prepared()
    assert ebm.MAX_TRAIN_ROWS == 100_000
    monkeypatch.setattr(ebm, "MAX_TRAIN_ROWS", 32)
    outputs = [ebm.fit_predict(prep, cnn_cfg=None, seed=seed) for seed in (7, 7, 8)]
    for seed, model, out in zip((7, 7, 8), tiny_ebms, outputs):
        rows = np.random.default_rng(seed).choice(len(prep.ctr), 32, replace=False)
        np.testing.assert_array_equal(model.fit_X_, tree_view(prep.Xtr[rows]))
        np.testing.assert_array_equal(model.fit_y_, prep.ctr[rows])
        assert out.info["n_train_used"] == len(model.fit_y_) == 32
        assert len(np.unique(rows)) == 32
    np.testing.assert_array_equal(tiny_ebms[0].fit_X_, tiny_ebms[1].fit_X_)
    np.testing.assert_allclose(outputs[0].pmf_test, outputs[1].pmf_test)
    assert not np.array_equal(tiny_ebms[0].fit_X_, tiny_ebms[2].fit_X_)


def test_importance_uses_named_normalized_terms(tiny_ebms):
    prep = prepared()
    out = ebm.fit_predict(prep, cnn_cfg=None, seed=42)
    model = tiny_ebms[-1]
    names = tree_feature_names(prep.cols)
    assert list(model.feature_names_in_) == names
    assert list(out.importance) == names
    values = model.term_importances()
    np.testing.assert_allclose(list(out.importance.values()), 100 * values / values.sum())
    assert sum(out.importance.values()) == pytest.approx(100)
    assert out.info["top_terms"] == sorted(out.importance.items(), key=lambda item: -item[1])[:15]
    assert len(out.info["top_terms"]) == min(15, len(names))


def test_fitting_reads_only_training_labels(tiny_ebms):
    class GuardedPrepared(Prepared):
        def __getattribute__(self, name):
            if name in {"cv", "ct", "ytr", "yv", "yt"}:
                raise AssertionError("EBM fitting must not read held-out targets")
            return super().__getattribute__(name)

    prep = prepared()
    guarded = GuardedPrepared(**vars(prep))
    ebm.fit_predict(guarded, cnn_cfg=None, seed=42)
    np.testing.assert_array_equal(tiny_ebms[-1].fit_X_, tree_view(prep.Xtr))
    np.testing.assert_array_equal(tiny_ebms[-1].fit_y_, prep.ctr)
