import numpy as np
import pytest
import torch

from whobpyt.datatypes import Timeseries
from whobpyt.optimization.custom_cost_JR import CostsJR
from whobpyt.run import ModelFitting


def _stimulus(n_nodes, n_samples, steps_per_sample=10):
    u = np.zeros((n_nodes, steps_per_sample, n_samples))
    u[:, :, 10:15] = 1000
    return u


@pytest.mark.parametrize("use_fit_gains", [True, False])
def test_simulation_shapes(make_jr_model, toy_network, use_fit_gains):
    model = make_jr_model(use_fit_gains=use_fit_gains)
    fit = ModelFitting(model, CostsJR(model))
    n_samples = 40
    blank = Timeseries(np.zeros((toy_network["n_chans"], n_samples)), 0.001)
    with np.errstate(divide="ignore", invalid="ignore"):
        fit.evaluate(u=_stimulus(toy_network["n_nodes"], n_samples), empRec=blank,
                     TPperWindow=10, base_window_num=2)
    eeg = fit.lastRec["eeg"].npTS()
    assert eeg.shape == (toy_network["n_chans"], n_samples)
    assert np.all(np.isfinite(eeg))
    for state in ["E", "I", "P"]:
        assert fit.lastRec[state].npTS().shape == (toy_network["n_nodes"], n_samples)


def test_fixed_gains_are_not_learnable(make_jr_model, toy_network):
    n = toy_network["n_nodes"]
    fitted = make_jr_model(use_fit_gains=True).params_fitted["modelparameter"]
    fixed = make_jr_model(use_fit_gains=False).params_fitted["modelparameter"]
    n_gain = lambda ps: sum(1 for p in ps if tuple(p.shape) == (n, n))
    assert n_gain(fitted) == 3
    assert n_gain(fixed) == 0


@pytest.mark.parametrize("use_fit_gains", [True, False])
def test_training_updates_parameters(make_jr_model, toy_network, use_fit_gains):
    model = make_jr_model(use_fit_gains=use_fit_gains)
    a_before = model.params.a.value().item()
    fit = ModelFitting(model, CostsJR(model))
    n_samples = 40
    rng = np.random.default_rng(1)
    emp = Timeseries(rng.normal(size=(toy_network["n_chans"], n_samples)), 0.001)
    fit.train(u=_stimulus(toy_network["n_nodes"], n_samples), empRecs=[emp],
              num_epochs=2, TPperWindow=10)
    assert len(fit.trainingStats.loss) == 2
    assert np.all(np.isfinite(fit.trainingStats.loss))
    assert len(fit.trainingStats.fit_params["a"]) == 2
    assert model.params.a.value().item() != pytest.approx(a_before)


def test_save_and_reload(make_jr_model, toy_network, tmp_path):
    import pickle

    model = make_jr_model()
    fit = ModelFitting(model, CostsJR(model))
    path = tmp_path / "fit.pkl"
    fit.save(str(path))
    with open(path, "rb") as f:
        loaded = pickle.load(f)
    assert loaded.model.node_size == toy_network["n_nodes"]
