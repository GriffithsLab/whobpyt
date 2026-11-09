import numpy as np
import pytest
import torch

from whobpyt.datatypes import Parameter as par
from whobpyt.datatypes import Timeseries
from whobpyt.datatypes.abstract_neural_model import get_git_commit_hash


class TestParameter:
    def test_fixed_value(self):
        p = par(3.25)
        assert not p.fit_par
        assert p.value().item() == pytest.approx(3.25)
        assert p.npValue() == pytest.approx(3.25)

    def test_learnable_starts_near_prior(self):
        p = par(100, 100, 2, True)
        assert p.fit_par and p.fit_hyper
        # Starting value is drawn from N(100, 2^2)
        assert abs(p.npValue().item() - 100) < 20
        assert p.prior_precision.item() == pytest.approx(1 / 4)

    def test_aslog_is_positive(self):
        p = par(np.log(1.5), asLog=True)
        assert p.value().item() == pytest.approx(1.5)

    def test_lower_bound(self):
        p = par(np.log(1.5), asLog=True, lb=0.1)
        assert p.value().item() == pytest.approx(1.6)

    def test_array_value(self):
        p = par(np.ones((3, 1)))
        assert p.value().shape == (3, 1)


class TestTimeseries:
    def test_shapes(self):
        data = np.arange(20, dtype=float).reshape(2, 10)
        ts = Timeseries(data, step_size=0.001, modality="eeg")
        assert ts.numNodes == 2
        assert ts.length == 10
        assert ts.step_size == 0.001
        assert ts.modality == "eeg"
        assert torch.is_tensor(ts.pyTS())
        np.testing.assert_array_equal(ts.npTS(), data)
        np.testing.assert_array_equal(ts.npTimeByNodes(), data.T)

    def test_windowed_tensor(self):
        data = np.arange(20, dtype=float).reshape(2, 10)
        win = Timeseries(data, 0.001).windowedTensor(5)
        assert win.shape == (2, 2, 5)
        np.testing.assert_array_equal(win[1], data[:, 5:])


def test_git_commit_hash_is_none_or_sha():
    h = get_git_commit_hash()
    assert h is None or (len(h) == 40 and all(c in "0123456789abcdef" for c in h))


def test_get_fitted_names():
    from whobpyt.models.jansen_rit import JansenRitParams

    params = JansenRitParams(a=par(100, 100, 2, True), A=par(3.25))
    names = params.getFittedNames()
    assert "a" in names and "a_m" in names and "a_v_inv" in names
    assert "A" not in names
