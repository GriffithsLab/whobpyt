Overview
================

Whole Brain Modelling in PyTorch (WhoBPyT) is a Python package for fitting parameters of Whole Brain Models (WBM) to neuroimaging data. In particular, differential equation based WBMs such as Connectome-based Neural Mass Models (CNMM) can be implemented in PyTorch, and by doing so the simulated neuoimaging data can be backpropagated through time to update model parameters. This is the deep learning approach that WhoBPyT uses. 

In order to use this package, a brain model, objective function, and parameter fitting paradigm must be chosen. The appropriate choices will depend on the research question and the neuroimaging data avaliable. Data must be processed ahead of time into a consistent parcellated format.

After fitting, the result will be one or more sets of paramters. It's important to verify these parameters in another model implementation, as the models implemented in PyTorch may have default or optional moditications that deviate from the original model's dynamics. 

If you use this package, please cite the papers listed on the :doc:`cite` page.
