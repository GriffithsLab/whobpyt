.. _design_philosophy:

Design philosophy
=================

WhoBPyT comes out of the research programme of the
`Whole-Brain Modelling Group <https://www.grifflab.com>`__: building
patient-personalized brain simulations that help interpret neuroimaging and
neurostimulation data. Neuroscience is very good at measuring brain activity
and at perturbing it, for example with TMS or intracranial stimulation. It is
less good at explaining *why* the measurements look the way they do, and
what actually happens in the brain when we stimulate it. Whole-brain models
are a way of answering those questions mechanistically, *in silico*. The
principles below follow from that goal.


Models are fitted to individual people
--------------------------------------

The aim is not a generic simulation of "a brain", but a model of *this*
person's brain, fitted to *their* EEG, MEG or fMRI recordings. Each model is
built on a subject's (or a template) connectome and, for EEG/MEG, on a lead
field that maps brain regions to sensors. Fitting it to their data yields
subject-specific parameter estimates that can then be compared across
people, conditions, ages, or treatment outcomes.


Fitting by gradient descent through the simulation
--------------------------------------------------

Estimating physiological parameters by fitting whole-brain models to data is
hard: networks are large, the equations are detailed and fast, and
simulations need to be long. WhoBPyT tackles this with the optimization tools
of deep learning. Its models are written in
`PyTorch <https://pytorch.org>`__ as if they were deep neural networks, so
automatic differentiation gives the gradient of a loss comparing simulated
and recorded data with respect to every learnable parameter, and
industry-standard optimizers update them all at once. This approach was introduced in
`Griffiths et al. (2022) <https://doi.org/10.1101/2022.05.19.492664>`__ and
is described in detail on the :doc:`implementation` page.


Parameters stay physiologically meaningful
------------------------------------------

Although the fitting machinery comes from deep learning, the models are not
black boxes. Every parameter is a quantity with a biophysical meaning, such
as a synaptic time constant, a connection strength or a conduction velocity.
Each one can be fixed or learnable and can carry a prior that keeps it in a
plausible range. That is what lets a fitted model say something about
mechanism. For example, in
`Momi et al. (2023) <https://doi.org/10.7554/eLife.83232>`__,
subject-specific estimates pointed to inhibitory (GABAergic) populations as a
key factor in scaling cortical excitability.


Models are tools for testing hypotheses
---------------------------------------

Once a model reproduces someone's data, it can be manipulated in ways the
real brain cannot. Connections can be cut at precise moments ("virtual
lesions"), parameters can be changed one at a time, and stimulation can be
delivered anywhere. The paper replications in the
:ref:`examples gallery <examples-gallery>` use this to ask, for instance,
which parts of a TMS-evoked response depend on recurrent feedback from the
rest of the network.


Building blocks are interchangeable
-----------------------------------

A fitting problem combines a model, a set of parameters, an objective
function and a fitting procedure. WhoBPyT keeps these separate, with abstract
base classes defining how they fit together, so that each can be swapped or
extended independently. See :doc:`architecture` for details.


Check what the fitted model captures
------------------------------------

A good fit is the start of an analysis, not the end. Fitted parameters should
be checked for how much individual-level information they actually carry,
and for how robust they are. Because the PyTorch implementations may differ
in small ways from the original form of a model, it is good practice to
confirm that fitted parameters reproduce the same behaviour in an
independent implementation.


Open and reproducible
---------------------

Whole-brain modelling should be reproducible and accessible to people
outside the groups that develop it. WhoBPyT is open source, its example
datasets are publicly hosted, and each published study that uses it comes
with a runnable, step-by-step replication in the examples gallery. Tutorials
and courses, such as the
`OHBM brain modelling course <https://www.grifflab.com/code_and_software/ohbm_wbm_course/>`__,
build on the same materials.
