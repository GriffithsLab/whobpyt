.. _cookbook:

Cookbook
========

Short recipes for common WhoBPyT tasks. For a complete, runnable walkthrough,
start with :ref:`ex-getting-started`; for full analyses of real data, see the
:ref:`examples gallery <examples-gallery>`.

.. contents:: Recipes
   :local:
   :depth: 1


Check your installation
-----------------------

.. code-block:: python

   import torch
   import whobpyt
   from whobpyt.models.jansen_rit import JansenRitModel

   print("PyTorch", torch.__version__, "| CUDA available:", torch.cuda.is_available())


Download the example datasets
-----------------------------

Each example has a fetcher in ``whobpyt.datasets.fetchers``. Data are
downloaded once to ``~/.whobpyt/data`` and reused after that. Each fetcher
returns the path of the folder it downloaded.

.. code-block:: python

   from whobpyt.datasets.fetchers import fetch_egtmseeg

   data_dir = fetch_egtmseeg()                          # default location
   data_dir = fetch_egtmseeg(dest_folder="/data/wbm")   # custom location
   data_dir = fetch_egtmseeg(redownload=True)           # force a fresh copy

The other fetchers are ``fetch_egmomi2023``, ``fetch_egmomi2025``,
``fetch_egismail2025`` and ``fetch_egismail2026``.


Make a parameter fixed or learnable
-----------------------------------

Model parameters are :class:`~whobpyt.datatypes.parameter.Parameter` objects:
``par(val, prior_mean=None, prior_std=None, fit_par=False, asLog=False, lb=0)``.

.. code-block:: python

   from whobpyt.datatypes import Parameter as par

   A  = par(3.25)                       # fixed at 3.25
   a  = par(100, 100, 2, True)          # learnable, Gaussian prior N(100, 2^2)
   mu = par(np.log(1.5), np.log(1.5), 0.1, True, True, lb=0.1)
                                        # learnable, stored as a log so it stays positive

When a parameter is learnable and has a prior, its starting value is a random
draw from the prior. Set ``np.random.seed`` and ``torch.manual_seed`` if you
need reproducible runs.

Pass the parameters to the model's parameter container, for example
``JansenRitParams(A=A, a=a, mu=mu, ...)``. Anything you don't pass keeps the
model's default value.


Choose which regions are stimulated
-----------------------------------

For the Jansen-Rit model, the ``ki`` parameter is a ``(regions, 1)`` array
that weights how strongly each region receives the external input:

.. code-block:: python

   ki0 = np.zeros((n_regions, 1))
   ki0[stimulated_region_indices] = 1
   params = JansenRitParams(..., ki=par(ki0))

The stimulus itself, ``u``, has shape
``(regions, steps_per_sample, n_samples)``, where
``steps_per_sample = int(tr / step_size)``:

.. code-block:: python

   u = np.zeros((n_regions, int(tr / step_size), n_samples))
   u[:, :, 80:120] = 1000    # pulse from sample 80 to 120


Prepare EEG data from MNE
-------------------------

WhoBPyT expects recordings as a
:class:`~whobpyt.datatypes.timeseries.Timeseries` of shape
``(channels, samples)``. From an :class:`mne.Evoked`:

.. code-block:: python

   import mne
   from whobpyt.datatypes import Timeseries

   evoked = mne.read_epochs("sub-01-epo.fif").average()
   t0, t1 = evoked.time_as_index([-0.1, 0.3])
   eeg = evoked.data[:, t0:t1]
   eeg = eeg / np.abs(eeg).max() * 4            # put on a convenient scale
   emp = Timeseries(eeg, step_size=1 / evoked.info["sfreq"])

The number of samples should be a multiple of the model's ``TRs_per_window``.


Train, evaluate, and simulate
-----------------------------

.. code-block:: python

   from whobpyt.optimization.custom_cost_JR import CostsJR
   from whobpyt.run import ModelFitting

   fit = ModelFitting(model, CostsJR(model))

   # Fit learnable parameters to one or more recordings
   fit.train(u=u, empRecs=[emp], num_epochs=50, TPperWindow=model.TRs_per_window,
             learningrate=0.05)

   # Run the trained model and compare it with the data
   fit.evaluate(u=u, empRec=emp, TPperWindow=model.TRs_per_window, base_window_num=20)

   sim_eeg = fit.lastRec["eeg"].npTS()      # (channels, samples)
   sim_P   = fit.lastRec["P"].npTS()        # pyramidal population, (regions, samples)

``base_window_num`` runs that many windows of resting activity before the
recording starts, so the model can settle into its steady state.


Track convergence
-----------------

.. code-block:: python

   import matplotlib.pyplot as plt

   plt.plot(fit.trainingStats.loss)                   # mean loss per epoch
   plt.plot(fit.trainingStats.fit_params["g"])        # a parameter per epoch

``fit.trainingStats.fit_params`` also stores each parameter's prior mean and
precision (for example ``"g_prior_mean"``) when those are learned.


Save and reload a fitted model
------------------------------

.. code-block:: python

   fit.save("sub-01_fit.pkl")

   import pickle
   with open("sub-01_fit.pkl", "rb") as f:
       fit = pickle.load(f)

The saved object contains the model, its learned parameters, and the
training history. Reloading it needs the same version of WhoBPyT.


Plot simulated EEG with MNE
---------------------------

Copy an empirical ``Evoked`` and replace its data with the simulation to use
MNE's plotting functions, such as topographic maps:

.. code-block:: python

   sim = evoked.copy()
   sim.data[:, t0:t1] = fit.lastRec["eeg"].npTS()
   sim.plot_joint(ts_args=dict(xlim=[-0.1, 0.3]))
