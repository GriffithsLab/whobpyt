"""Shared fixtures for the WhoBPyT test suite."""

import matplotlib
import numpy as np
import pytest
import torch

matplotlib.use("agg")


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(0)
    torch.manual_seed(0)


@pytest.fixture
def toy_network():
    """A small random connectome, lead field and stimulus weights."""
    rng = np.random.default_rng(0)
    n_nodes, n_chans = 6, 4
    sc = rng.random((n_nodes, n_nodes))
    sc = (sc + sc.T) / 2
    np.fill_diagonal(sc, 0)
    sc = np.log1p(sc) / np.linalg.norm(np.log1p(sc))
    dist = rng.uniform(10, 60, size=(n_nodes, n_nodes))
    dist = (dist + dist.T) / 2
    np.fill_diagonal(dist, 0)
    lm = rng.normal(0, 0.1, size=(n_chans, n_nodes))
    ki = np.zeros((n_nodes, 1))
    ki[0] = 1
    return dict(n_nodes=n_nodes, n_chans=n_chans, sc=sc, dist=dist, lm=lm, ki=ki)


@pytest.fixture
def make_jr_model(toy_network):
    """Factory for a small Jansen-Rit model on the toy network."""
    from whobpyt.datatypes import Parameter as par
    from whobpyt.models.jansen_rit import JansenRitModel, JansenRitParams

    net = toy_network

    def _make(use_fit_gains=True, TRs_per_window=10):
        params = JansenRitParams(
            A=par(3.25), a=par(100, 100, 2, True), B=par(22), b=par(50, 50, 1, True),
            g=par(500, 500, 2, True), g_f=par(10), g_b=par(10),
            c1=par(135), c2=par(108), c3=par(33.75), c4=par(33.75),
            std_in=par(np.log(10), np.log(10), .1, True, True),
            vmax=par(5), v0=par(6), r=par(0.56), y0=par(-2),
            mu=par(np.log(1.5), np.log(1.5), .1, True, True, lb=0.1),
            k=par(5., lb=1), k0=par(0), cy0=par(50), ki=par(net["ki"]), lm=par(net["lm"]),
        )
        return JansenRitModel(
            params, node_size=net["n_nodes"], TRs_per_window=TRs_per_window,
            step_size=0.0001, output_size=net["n_chans"], tr=0.001,
            sc=net["sc"], lm=net["lm"], dist=net["dist"], use_fit_gains=use_fit_gains,
        )

    return _make
