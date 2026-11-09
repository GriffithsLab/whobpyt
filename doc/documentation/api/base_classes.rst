Base classes
============

Inherit from these classes to add a new model, objective function or
fitting paradigm that works with the rest of WhoBPyT. See
:doc:`/documentation/about/architecture` for an overview.

.. autoclass:: whobpyt.datatypes.AbstractParams
   :members:

.. autoclass:: whobpyt.datatypes.AbstractNeuralModel
   :members: setModelParameters, createIC, createDelayIC, forward, info

.. autoclass:: whobpyt.datatypes.AbstractLoss
   :members:

.. autoclass:: whobpyt.datatypes.AbstractFitting
   :members:
