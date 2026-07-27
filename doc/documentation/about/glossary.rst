Glossary
========

.. glossary::
   :sorted:

   Whole-brain model (WBM)
      A dynamical systems model that simulates activity across many brain regions at once, using region-to-region coupling and neural state equations.

   Connectome-based neural mass model (CNMM)
      In this project, a neural mass model coupled by a structural connectivity matrix (``sc``), with optional distance-based delays (``dist``) and a measurement model.

   Neural mass model (NMM)
      A reduced biophysical model of population activity per region. In WhoBPyT, NMMs inherit from ``whobpyt.datatypes.AbstractNeuralModel``.

   Structural connectivity (SC)
      The inter-regional coupling matrix used for long-range interactions (for example ``sc`` in ``JansenRitModel``), often derived from diffusion MRI tractography.

   Distance matrix
      Pairwise inter-regional distances (``dist``), used to compute integer delay indices in delayed coupling terms.

   Lead field matrix
      A source-to-sensor projection matrix (``lm``) used to map simulated source activity to M/EEG channel space.

   State variables
      Dynamical variables integrated by the model at each step. For ``JansenRitModel`` these are ``E``, ``Ev``, ``I``, ``Iv``, ``P``, and ``Pv``.

   Output variables
      Simulated observables exposed by a model for fitting. For the current Jansen-Rit implementation, ``output_names = ["eeg"]``.

   Timeseries
      The ``whobpyt.datatypes.Timeseries`` container for empirical or simulated data, typically shaped as ``num_regions x ts_length``.

   Windowed fitting
      A training pattern where longer recordings are split into windows (``Timeseries.windowedTensor``), and gradients are backpropagated window-by-window.

   Objective function / loss
      A differentiable criterion minimized during fitting, implemented via ``AbstractLoss`` and concrete classes such as ``CostsTS``, ``CostsFC``, and ``CostsMean``.

   Prior loss
      A regularization term from ``AbstractLoss.prior_loss`` that penalizes deviation from parameter priors when hyperparameters are fit.

   Parameter object
      The ``whobpyt.datatypes.Parameter`` class storing a value, optional prior mean/precision, and flags controlling whether values and/or priors are trainable.

   Fitting paradigm
      A training workflow class (for example ``ModelFitting``, ``FittingBatch``, ``FittingFNGFPG``) that defines simulation, optimization, and bookkeeping behavior.

   FNG-FPG
      ``FittingFNGFPG`` ("Forward No Gradient, Forward Parallel Gradient"), a specialized approach for long simulations using a serial pass for initial conditions and a gradient-enabled blocked pass.

   Training statistics
      The ``TrainingStats`` record of losses, tracked parameters, and selected matrices (for example fitted connectivity/lead-field terms) across optimization.

   External input / stimulus
      The ``u`` or ``external`` input tensor supplied to forward simulation for resting-state (often zero) or evoked/stimulus-driven modeling.
