.. _implementation:

Algorithms and other implementation details
===========================================

WhoBPyT is organized around composable building blocks for differentiable whole-brain modeling:

* **Model classes** in ``whobpyt.models`` implement neural dynamics and forward simulation.
* **Datatypes** in ``whobpyt.datatypes`` define shared interfaces and containers (for example ``AbstractNeuralModel``, ``Parameter``, ``Timeseries``, ``TrainingStats``).
* **Objective functions** in ``whobpyt.optimization`` implement fitting criteria such as time-series error and FC-based losses.
* **Fitting paradigms** in ``whobpyt.run`` implement optimization workflows (for example ``ModelFitting``, ``FittingBatch``, ``FittingFNGFPG``).

The current mainline model implementation is the Jansen-Rit neural mass model
(``whobpyt/models/jansen_rit/jansen_rit.py``), with state variables
``E``, ``Ev``, ``I``, ``Iv``, ``P``, ``Pv`` and output ``eeg``.

Core simulation and training flow
---------------------------------

At a high level, model fitting uses gradient-based optimization over parameters
registered as trainable ``torch.nn.Parameter`` objects through ``AbstractNeuralModel.setModelParameters``.

The standard ``ModelFitting.train`` workflow is:

#. Initialize model state and (if used) delay history via ``createIC`` and ``createDelayIC``.
#. Convert empirical recordings into windows with ``Timeseries.windowedTensor``.
#. For each epoch and each window, call the model ``forward`` pass.
#. Compute loss via a selected objective function (and optional prior terms).
#. Backpropagate with ``loss.backward()`` and update parameters using Adam optimizers.
#. Store fit diagnostics in ``TrainingStats``.

Within ``JansenRitModel.forward``, numerical integration is implemented as an
explicit Euler update over ``steps_per_TR`` integration steps per sampled output
time point, with:

* local population interactions,
* long-range coupling weighted by structural connectivity (``sc``),
* optional distance-based delays (``dist``),
* optional Laplacian-like self-coupling terms,
* stochastic input noise and external drive.

Objective functions and priors
------------------------------

The optimization module provides reusable loss components, including:

* ``CostsTS``: time-series RMSE-style loss,
* ``CostsFC`` / ``CostsFixedFC``: FC similarity based on lower-triangle Pearson-correlation structure,
* ``CostsMean``: target mean-value matching for selected variables.

``AbstractLoss.prior_loss`` provides a prior-based penalty term when fitting
parameter hyperparameters (for example prior mean/precision in ``Parameter`` objects).

Alternative fitting paradigms
-----------------------------

Besides ``ModelFitting``, the repository includes:

* ``FittingBatch``: batched simulation/optimization workflow,
* ``FittingFNGFPG``: a serial-forward then blocked-gradient workflow for long-duration simulations.

These paradigms reuse the same model/loss abstractions, so objective functions
and model classes can be mixed as long as tensor shapes and expected keys align.

Data handling in practice
-------------------------

The repository includes both synthetic generators (for example ``datasets.generators.gen_cube``)
and dataset fetchers (``datasets.fetchers``) for example and reproduction workflows.
Common data objects across fitting APIs are:

* structural connectivity matrices,
* distance matrices for delays,
* lead-field matrices for source-to-sensor mapping,
* empirical ``Timeseries`` objects in region- or channel-space.

Relation to published work
--------------------------

WhoBPyT's implementation follows the same general project direction described in
the published and linked work: differentiable large-scale neural modeling in
PyTorch, fitted against empirical neuroimaging/neurophysiology targets.

In this repository, that relation is visible through:

* explicit dataset helpers and example paths named for these studies (for example
  ``fetch_MomiEtAlELife2023``, ``fetch_egmomi2025``, ``fetch_egismail2026``),
* archived/deprecated study-specific code under ``whobpyt/depr/*``,
* core reusable abstractions in ``datatypes``, ``models``, ``optimization``, and ``run`` used to support similar workflows across studies.

References:

* `Momi et al. (2023), eLife <https://elifesciences.org/articles/83232>`_
* `Momi et al. (2025), Nature Communications <https://www.nature.com/articles/s41467-025-58187-6>`_
* `Ismail et al. (2026), Nature Communications <https://www.nature.com/articles/s41467-026-71918-7>`_

