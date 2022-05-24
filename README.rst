WhoBPyT - Whole-Brain Modelling in PyTorch
=================

_"Or there and Back Again": https://en.wikipedia.org/wiki/The_Hobbit


Summary
-------

``KFTools`` is a Python library for Python-based analyses of fNIRS and
EEG data from the Kernel Flow system.

Please note that the code here is still very preliminary, under active
development, and subject to substantial change.

Rationale
---------

As one of the first scientific groups undertaking new research projects
with the Kernel Flow system, it became quickly apparent that
establishing a wider user community will be one of the best routes to
rapid and effective progress for all concerned.

As open source and open science advocates, we have elected to pursue a
‘fully open’ and public development approach here. A chief motivation
behind this is to bring in contributors and collaborators who are
interested in working together to move things forward more quickly that
any of us would be able to individually. So, if you are interested in
getting involved, don’t hesitate to reach out to John ( j dot
davidgriffiths at gmail dot com ), or just introduce yourself via an
issue.

Structure
---------

The objective of ``KFTools`` is to act as a set of thin wrappers on
actual analysis software. It is Python-based, with a sprinkling of
Matlab here and there. The wrapper functions do a few useful data
organization things, and have some useful expectiations / knowledge
about file structures, experiment types, event coding conventions, etc.

The ``KFTools`` functions are mostly based on two established and
best-in-class neuroimaging anaysis libraries: MNE (and especially
mne-nirs), and Nilearn. There is also some Homer3 analysis
functionality.

There are three main components to the code base:

|  ``doc`` folder - Documentation pages text and organization
|  ``kftools`` folder - The importable python library
|  ``examples`` folder - Example usage scripts that become the gallery
  items in the CI-managed sphinx gallery site
