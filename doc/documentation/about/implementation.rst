.. _implementation:

Algorithms and other implementation details
===========================================

This page describes *how* WhoBPyT actually simulates and fits Whole Brain Models (WBMs), as
distinct from the "Code Architecture" page, which describes how the classes are organized.
In short: a WBM is implemented as a PyTorch ``nn.Module`` whose
``forward()`` method numerically integrates a set of coupled differential equations, and
"fitting" the model means backpropagating a loss computed from the simulated output through
that numerical integration in order to update the equations' parameters with gradient
descent. The sections below unpack each part of that pipeline: the numerical solver, the
connectivity/delay machinery, how parameters and priors are represented, the objective
functions, and the windowed training loop that ties it all together.


Simulation as a recurrent computation graph
--------------------------------------------

Every model in ``whobpyt.models`` (e.g. ``JansenRitModel``) subclasses
``AbstractNeuralModel``, which in turn subclasses ``torch.nn.Module``. Its ``forward()``
method advances the model's state variables (e.g. the current and voltage of the pyramidal,
excitatory, and inhibitory populations in the Jansen-Rit model) forward in time using a
fixed-step explicit numerical solver, currently the **forward Euler method**:

.. math::

   x_{t + \Delta t} = x_t + \Delta t \cdot f(x_t, \theta)

where :math:`f` is the model's system of ODEs/SDEs and :math:`\theta` is the set of model
parameters (see `Parameters and priors`_ below). Because integration is expressed with
standard PyTorch tensor operations, the entire sequence of Euler updates is a differentiable
computation graph: gradients of any scalar function of the simulated output (i.e. an
objective/loss function) can be backpropagated all the way through the numerical solver to
the model's parameters, using PyTorch's automatic differentiation (``autograd``). This is the
central mechanism that WhoBPyT exploits and the reason for implementing WBMs in PyTorch
rather than, say, NumPy or MATLAB: it turns classical parameter estimation for neural mass
models into a deep-learning-style optimization problem.

Simulation time is organized into four nested loops, matching the loops used in
``ModelFitting.train()``:

1. **Epochs** — full passes through the training dataset.
2. **Recordings** — one or more empirical time series (each represented as a
   ``Timeseries``/``Recording`` object) that make up a training dataset.
3. **Windows** — each recording is split into fixed-length windows of ``TPperWindow`` time
   points; one gradient update happens per window.
4. **Integration steps** — within a window, the solver takes many small time steps
   (``step_size``) for every simulated output sample (``steps_per_TR``), and only the state
   at the end of each ``steps_per_TR`` block is kept as one time point of simulated output.

This nesting is what allows the very small integration step required for numerical stability
(fractions of a millisecond) to coexist with the much coarser sampling rate of the empirical
neuroimaging data (e.g. TR of ~1s for fMRI, ~1ms for EEG).


Connectivity, gains, and delays
--------------------------------

Whole Brain Models couple many local neural mass "nodes" (one per brain region/ROI) through a
structural connectivity (SC) matrix. WhoBPyT treats the *strength* of these connections as
learnable in the same way as any other model parameter:

- A raw SC matrix (from tractography, for example) is supplied at model construction.
- If ``use_fit_gains=True``, an additional learnable gain matrix (or matrices, e.g.
  ``w_p2e``, ``w_p2i``, ``w_p2p`` for pyramidal-to-excitatory, pyramidal-to-inhibitory, and
  pyramidal-to-pyramidal connections in the Jansen-Rit model) is exponentiated,
  elementwise-multiplied with the SC matrix, and normalized, so that the *effective*
  connectivity used at each forward pass is always non-negative and can be optimized
  by gradient descent without leaving that constraint.
- If ``use_laplacian=True``, a graph-Laplacian term (row sums placed on the diagonal, with a
  negative sign) is added to the connectivity operator, which is a common way of keeping
  large-scale network dynamics stable when connection weights are themselves being learned.

Conduction delays between nodes are handled with an explicit **delay buffer**: a rolling
history ``hE`` of a state variable (e.g. the pyramidal population current) is kept for as
many past integration steps as the longest delay requires. At every step, each node's delayed
input is read out of this buffer at an index computed from a distance matrix and a
conduction-velocity-like parameter (``mu``), i.e. ``delay = distance / mu``. The buffer is
carried over between windows (but, as described below, detached from the autograd graph),
so that delayed dynamics are continuous across the training loop even though the loss is only
backpropagated through one window at a time.


Parameters and priors
----------------------

Every model-specific numerical constant (e.g. ``A``, ``a``, ``B``, ``b`` in the Jansen-Rit
model) is wrapped in a small ``Parameter`` object (``whobpyt.datatypes.Parameter``)
rather than being a plain float. This wrapper is what lets the same model class support a
spectrum of use cases, from a completely fixed forward simulation to a fully hierarchical
Bayesian-flavoured fit, with no change to the equations themselves. A ``Parameter`` stores:

- ``val`` — the current value (or, for spatially-varying parameters, one value per node).
- ``prior_mean`` / ``prior_precision`` — an optional Gaussian prior on the value.
- ``fit_par`` — whether ``val`` should be registered as a ``torch.nn.Parameter`` and thus
  optimized directly.
- ``fit_hyper`` — whether the *prior* itself (``prior_mean``/``prior_precision``) should also
  be optimized, giving a second, hierarchical level of learnable "hyperparameters".
- ``asLog`` — whether ``val`` is internally stored in log-space, which is a simple and
  effective way of constraining a parameter (e.g. a rate constant or standard deviation) to
  stay positive throughout optimization without needing a constrained optimizer.

``AbstractNeuralModel.setModelParameters()`` walks over all ``Parameter`` attributes of a
model's params object and sorts them into two groups, ``modelparameter`` and
``hyperparameter``, which are optimized with two separate ``torch.optim.Adam`` instances (see
`Training loop`_) — this is what allows model parameters and their priors to be updated with
independent learning rates.


Objective functions
---------------------

Objective/loss functions subclass ``AbstractLoss`` and implement a ``main_loss(simData,
empData)`` method that compares a simulated output (indexed out of the model's forward-pass
output dictionary by a ``simKey`` such as ``"eeg"`` or ``"bold"``) against the corresponding
empirical data for that training window. Several are provided out of the box, operating over
different summary statistics of the signal rather than the raw time series itself, which
tends to be more robust to arbitrary phase/timing mismatches between simulated and empirical
data:

- **CostsFC** — correlation between simulated and empirical functional connectivity (FC)
  matrices.
- **CostsTS** — direct time series similarity.
- **CostsPSD** — power spectral density similarity.
- **CostsMean** — target mean value of a state variable.

The total loss used for backpropagation is the data-fit term above plus an optional
**prior loss**, computed generically by ``AbstractLoss.prior_loss()`` for every parameter
that has ``fit_hyper=True``:

.. math::

   \mathcal{L}_{\text{prior}} = \sum_{\theta}
       \big(\lambda + \text{precision}_\theta\big)\,\big(\theta - \mu_\theta\big)^2
       - \log\big(\lambda + \text{precision}_\theta\big)

This is a Gaussian negative log-likelihood penalty that pulls each hierarchically-fit
parameter back towards its (also-learnable) prior mean, weighted by its (also-learnable)
prior precision — i.e. the model is discouraged from moving a parameter far from its prior
unless the data strongly support it. Custom objective functions (e.g.
``custom_cost_JR.py``) simply combine ``main_loss`` and ``prior_loss`` terms, optionally with
their own weighting.


Training loop
---------------

``ModelFitting.train()`` (``whobpyt.run.model_fitting``) implements gradient-based fitting as
**truncated backpropagation through time (BPTT)**:

1. At the start of each recording, a short *warm-up* period (100 windows on the first epoch,
   ``warmupWindow`` thereafter) is simulated with gradients disabled, purely to let the state
   variables and delay buffer settle away from their arbitrary initial conditions.
2. For each subsequent window, the model is run forward for exactly one window's worth of
   time points, a loss is computed against the corresponding empirical window, and
   ``loss.backward()`` is called.
3. Both optimizers (``modelparameter_optimizer`` and ``hyperparameter_optimizer``, each
   ``torch.optim.Adam``, optionally with a ``OneCycleLR`` schedule) take a step.
4. Critically, the state (``X``) and delay buffer (``hE``) that are carried forward into the
   *next* window are **detached** from the autograd graph before the next window starts
   (``X = next_window['current_state'].detach().clone()``). This bounds the length of the
   computation graph that gradients are backpropagated through to a single window, which
   keeps memory and compute roughly constant regardless of how long the empirical recording
   is, at the cost of gradients not seeing the effect of a parameter change on windows before
   the current one. This is what makes training WBMs on long recordings (whole resting-state
   scans, for example) tractable.

Across an epoch, the loss for every window and the current value of every tracked parameter
(including the fitted SC/gain matrices) are logged into a ``TrainingStats`` object, which is
what the visualization utilities in ``whobpyt.visualization`` plot to inspect convergence.


Practical notes
-----------------

- **Device.** All the classes above accept a ``torch.device``, so the same code path runs on
  CPU or GPU; the numerical operations are ordinary batched tensor ops with no
  device-specific logic.
- **Reproducibility.** ``AbstractNeuralModel`` records the current git commit hash of the
  installed package at construction time, so a saved/pickled fit can be traced back to the
  exact version of the equations that produced it.
- **Verifying fitted parameters.** Because the PyTorch implementation of a model may include
  optional numerical modifications relative to the original published equations (see the
  model-specific ``README`` files, e.g. ``whobpyt/models/jansen_rit/README.md``), it is good
  practice to re-simulate with the fitted parameters in an independent implementation of the
  model before drawing scientific conclusions from them.
- **Saving/loading.** A ``ModelFitting`` instance — model, objective function, and training
  history together — can be serialized as a single object with ``ModelFitting.save()``
  (pickle-based), so a fit can be resumed or inspected later without re-running training.
