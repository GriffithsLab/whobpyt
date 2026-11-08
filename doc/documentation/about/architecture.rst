.. _architecture:

Code architecture
=================

WhoBPyT is organised around four kinds of interchangeable building block.
Fitting a model means choosing one of each:

.. list-table::
   :header-rows: 1
   :widths: 22 22 56

   * - Building block
     - Sub-package
     - Role
   * - Model
     - ``whobpyt.models``
     - Simulates brain activity: a neural mass model for every region,
       coupled through the connectome, plus a measurement model that maps
       region activity to the recorded signal (for example a lead field for
       EEG).
   * - Parameters
     - ``whobpyt.datatypes``
     - Hold each model parameter's value, its prior, and whether it is
       learnable.
   * - Objective function
     - ``whobpyt.optimization``
     - Scores how well a simulation matches the data.
   * - Fitting paradigm
     - ``whobpyt.run``
     - Runs the simulate-compare-update loop that fits the parameters.

How the implementation works, including the numerical solver, delays, priors
and the training loop, is described on the :doc:`implementation` page.


Typical workflow
----------------

.. code-block:: python

   from whobpyt.datatypes import Parameter as par, Timeseries
   from whobpyt.models.jansen_rit import JansenRitModel, JansenRitParams
   from whobpyt.optimization.custom_cost_JR import CostsJR
   from whobpyt.run import ModelFitting

   emp = Timeseries(eeg, step_size)                         # the data
   params = JansenRitParams(a=par(100, 100, 2, True), ...)  # what to fit
   model = JansenRitModel(params, sc=sc, lm=lm, dist=dist, ...)
   fit = ModelFitting(model, CostsJR(model))
   fit.train(u=stimulus, empRecs=[emp], num_epochs=50, TPperWindow=20)
   fit.evaluate(u=stimulus, empRec=emp, TPperWindow=20)

See :ref:`ex-getting-started` for a complete, runnable version.


What's included
---------------

Models (``whobpyt.models``)
   ``JansenRitModel`` and ``JansenRitParams``: a Jansen-Rit neural mass model
   with distance-dependent delays, learnable connection gains, and a lead
   field to simulate EEG/MEG.

Data types (``whobpyt.datatypes``)
   ``Parameter`` (a value with an optional prior), ``Timeseries`` (a
   recording or simulation), ``TrainingStats`` (loss and parameter history),
   and the abstract base classes listed below.

Objective functions (``whobpyt.optimization``)
   ``CostsTS`` (time-series error), ``CostsFC`` and ``CostsFixedFC``
   (functional connectivity), ``CostsFixedPSD`` (power spectrum),
   ``CostsMean`` (target mean value of a state variable), and ``CostsJR``
   (time-series error plus parameter priors, used with the Jansen-Rit
   model). ``CostsPSD`` is marked as no longer supported in the code.

Fitting paradigms (``whobpyt.run``)
   ``ModelFitting`` trains on consecutive windows of a recording and is the
   one used in the examples. ``FittingFNGFPG`` and ``FittingBatch`` are
   specialised alternatives for long simulations and batched training.

Other modules
   ``whobpyt.datasets`` fetches the example datasets (see :doc:`datasets`),
   ``whobpyt.visualization`` has plotting helpers, and ``whobpyt.depr``
   keeps the older, paper-specific model code that some of the paper
   replications still use.


Extending WhoBPyT
-----------------

New components should inherit from the abstract classes in
``whobpyt.datatypes``. They define the methods the other classes rely on, so
a new model, cost or fitting paradigm works with the existing ones.

``AbstractParams``
   Container for a model's ``Parameter`` objects.

``AbstractNeuralModel``
   A ``torch.nn.Module``. Implement ``createIC`` and ``createDelayIC``
   (initial states and delay history) and ``forward(external, hx, hE)``,
   which simulates one window. Its ``setModelParameters`` method registers
   the learnable parameters with PyTorch.

``AbstractLoss``
   Implement ``main_loss(simData, empData)``. ``prior_loss`` adds the
   penalty for learnable parameters straying from their priors.

``AbstractFitting``
   Implement ``train``, ``evaluate`` and ``simulate``.
