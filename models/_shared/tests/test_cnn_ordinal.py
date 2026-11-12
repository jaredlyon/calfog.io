from types import SimpleNamespace

import numpy as np
import pytest
import torch

from models._shared.families import temporal_cnn as cnn
from models._shared.visibility import N_CATEGORIES


CFG = {"channels": 2, "kernel": 3, "dropout": 0.0, "lr": 0.01,
       "weight_decay": 0.0, "batch_size": 8, "max_neg": 5}


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_ordinal_head_returns_valid_pmf_and_ordered_cumulative_probabilities():
    cnn.seed_all(5)
    model = cnn.TemporalCNN(2, CFG).cpu()
    X = np.random.default_rng(5).normal(size=(6, 4, 2)).astype(np.float32)
    pmf = cnn.predict_cnn(model, X, "cpu", batch=2)
    assert pmf.shape == (6, N_CATEGORIES)
    assert np.isfinite(pmf).all()
    assert (pmf >= 0).all()
    np.testing.assert_allclose(pmf.sum(axis=1), 1, atol=1e-7)
    assert (np.diff(pmf.cumsum(axis=1), axis=1) >= 0).all()
    assert (model.cutpoints().detach().diff() > 0).all()
    assert cnn.predict_cnn(model, X[:0], "cpu").shape == (0, N_CATEGORIES)
    # The PMF is specifically the difference of the proportional-odds CDF.
    with torch.no_grad():
        tensor = torch.from_numpy(X)
        score = model.head(model.net(tensor.transpose(1, 2)))
        expected = torch.sigmoid(model.cutpoints()[None, :] - score).numpy()
    np.testing.assert_allclose(pmf.cumsum(axis=1)[:, :-1], expected, atol=1e-7)


@pytest.mark.parametrize("score,category", [
    (-100.0, 8), (-30.0, 8), (30.0, 8), (100.0, 8),
    (30.0, 0), (100.0, 0), (-30.0, 17), (-100.0, 17),
])
def test_extreme_wrong_scores_keep_finite_recovery_gradients(score, category):
    model = cnn.TemporalCNN(1, CFG).eval()
    with torch.no_grad():
        model.head[-1].weight.zero_()
        model.head[-1].bias.fill_(score)
    log_pmf = model.log_probabilities(torch.zeros((1, 4, 1)))
    assert torch.isfinite(log_pmf).all()
    loss = cnn._ordinal_nll(log_pmf, torch.tensor([category])).mean()
    assert torch.isfinite(loss)
    loss.backward()
    gradient = model.head[-1].bias.grad.item()
    assert np.isfinite(gradient)
    # Positive score shifts mass upward; gradient descent must pull an
    # erroneously high score down and an erroneously low score up.
    assert gradient * score > 0
    assert abs(gradient) > 0.9


def test_top_subsampling_weight_recovers_no_signal_category_frequencies():
    categories = np.r_[np.arange(N_CATEGORIES - 1), np.full(83, N_CATEGORIES - 1)]
    X = np.zeros((len(categories), 4, 1), dtype=np.float32)
    Xs, cs, weights = cnn.balanced_subset(X, categories, max_neg=5, seed=7)
    assert len(cs) == 22
    assert (cs == 17).sum() == 5
    np.testing.assert_array_equal(cs[cs != 17], np.arange(17))
    np.testing.assert_allclose(weights[cs == 17], 83 / 5)
    assert weights.sum() == pytest.approx(len(categories))
    model = cnn.TemporalCNN(1, CFG).double().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    parameters = [model.cutpoint_start, model.cutpoint_gaps, model.head[-1].bias]
    for parameter in parameters:
        parameter.requires_grad_(True)
    optimizer = torch.optim.LBFGS(parameters, max_iter=100, line_search_fn="strong_wolfe",
                                 tolerance_grad=1e-9, tolerance_change=1e-12)
    tensor = torch.as_tensor(Xs, dtype=torch.float64)
    labels = torch.from_numpy(cs)
    w = torch.as_tensor(weights, dtype=torch.float64)

    def closure():
        optimizer.zero_grad()
        loss = (cnn._ordinal_nll(model.log_probabilities(tensor), labels) * w).sum() / w.sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        pmf = model(tensor).mean(0).numpy()
    expected = np.bincount(categories, minlength=N_CATEGORIES) / len(categories)
    np.testing.assert_allclose(pmf, expected, atol=0.003)
    # Without inverse-rate correction the selected sample would claim 5/22,
    # not 83/100, for the top category.
    assert abs(pmf[-1] - (cs == 17).mean()) > 0.5


def test_training_loss_weights_equal_full_population_on_constant_inputs(monkeypatch):
    categories = np.r_[np.arange(17), np.full(83, 17)]
    X = np.zeros((len(categories), 4, 1), dtype=np.float32)
    # A zero learning rate preserves the parameters for an exact gradient
    # comparison between the actual sampled training loop and the full data.
    monkeypatch.setattr(cnn.torch.optim, "AdamW",
                        lambda parameters, **kwargs: torch.optim.SGD(parameters, lr=0))
    config = {**CFG, "batch_size": len(categories)}
    model, _ = cnn.train_cnn(X, categories, X, categories, config, device="cpu",
                             max_epochs=1, seed=7)
    sampled = {name: parameter.grad.clone() for name, parameter in model.named_parameters()
               if name.startswith("cutpoint") or name == "head.2.bias"}
    model.zero_grad()
    model.train()
    cnn._ordinal_nll(model.log_probabilities(torch.from_numpy(X)), torch.from_numpy(categories)).mean().backward()
    for name, parameter in model.named_parameters():
        if name in sampled:
            torch.testing.assert_close(sampled[name], parameter.grad, atol=1e-5, rtol=1e-4)


def test_uncapped_subset_and_invalid_cap():
    X = np.zeros((4, 4, 1), dtype=np.float32)
    categories = np.array([0, 4, 12, 17])
    Xs, cs, weights = cnn.balanced_subset(X, categories, max_neg=10)
    np.testing.assert_array_equal(Xs, X)
    np.testing.assert_array_equal(cs, categories)
    np.testing.assert_array_equal(weights, np.ones(4))
    with pytest.raises(ValueError, match="max_neg"):
        cnn.balanced_subset(X, categories, max_neg=0)


def test_train_early_stops_on_nll_not_ap(monkeypatch):
    X = np.zeros((4, 4, 1), dtype=np.float32)
    ctr = np.array([0, 17, 0, 17])
    cv = np.array([0, 17])
    fog_probabilities = iter(([0.2, 0.1], [0.4, 0.5], [0.1, 0.9]))

    def predictions(model, features, device):
        p = np.asarray(next(fog_probabilities))
        pmf = np.zeros((2, N_CATEGORIES))
        pmf[:, 0], pmf[:, -1] = p, 1 - p
        return pmf

    monkeypatch.setattr(cnn, "predict_cnn", predictions)
    reports = []
    model, info = cnn.train_cnn(X, ctr, X[:2], cv, CFG, device="cpu", max_epochs=4,
                                patience=1, seed=7, report=reports.append)
    assert isinstance(model, cnn.TemporalCNN)
    assert info["cnn_epochs"] == 3
    assert info["cnn_best_val_nll"] == pytest.approx(-np.log([0.4, 0.5]).mean())
    assert reports[0]["val_ap"] > reports[1]["val_ap"]
    assert info["cnn_best_val_ap"] == reports[1]["val_ap"]
    assert {"val_nll", "val_ap", "epoch"} <= reports[0].keys()


def test_fit_predict_preserves_state_and_first_convolution_importance(monkeypatch):
    model = cnn.TemporalCNN(2, CFG).eval()
    with torch.no_grad():
        model.net[0].weight[:, 0, :] = 1
        model.net[0].weight[:, 1, :] = 3
    X = np.zeros((4, 4, 2), dtype=np.float32)
    c = np.array([0, 17, 1, 16])
    prep = SimpleNamespace(Xtr=X, ctr=c, Xv=X[:2], cv=c[:2], Xt=X[:1], cols=["a", "b"])

    def train(Xtr, ctr, Xv, cv, config, **kwargs):
        np.testing.assert_array_equal(ctr, c)
        return model, {"cnn_best_val_nll": 1.0, "cnn_best_val_ap": 0.5, "cnn_epochs": 2}

    monkeypatch.setattr(cnn, "train_cnn", train)
    monkeypatch.setattr(cnn.torch.cuda, "is_available", lambda: False)
    output = cnn.fit_predict(prep, cnn_cfg=CFG, seed=7)
    assert output.pmf_val.shape == (2, N_CATEGORIES)
    assert output.pmf_test.shape == (1, N_CATEGORIES)
    assert output.importance == {"a": 25.0, "b": 75.0}
    assert output.state.keys() == model.state_dict().keys()
    for key, value in output.state.items():
        assert value.device.type == "cpu"
        assert torch.equal(value, model.state_dict()[key])
        assert value.data_ptr() != model.state_dict()[key].data_ptr()
