.. _installation:

============
Installation
============

WhoBPyT needs Python 3.10 or newer. We recommend installing it into its own
conda (or virtualenv) environment:

.. code-block:: console

   $ conda create -n whobpyt python=3.10 pip
   $ conda activate whobpyt

Install WhoBPyT
---------------

WhoBPyT is not yet on PyPI, so install it from GitHub:

.. code-block:: console

   $ pip install "whobpyt[examples] @ git+https://github.com/GriffithsLab/whobpyt.git"

The ``[examples]`` part also installs the packages the
:ref:`examples gallery <examples-gallery>` uses, such as MNE-Python. Leave it
out if you only need the library itself:

.. code-block:: console

   $ pip install "git+https://github.com/GriffithsLab/whobpyt.git"

Check that it worked:

.. code-block:: console

   $ python -c "import whobpyt; print(whobpyt.__version__)"

Install for development
-----------------------

To work on WhoBPyT itself, clone the repository and install it in editable
mode, with everything needed for the examples, documentation and tests:

.. code-block:: console

   $ git clone https://github.com/GriffithsLab/whobpyt
   $ cd whobpyt
   $ pip install -e ".[dev]"
   $ pytest

See the :doc:`/documentation/development/contributing` for more.

Optional extras
---------------

=============  ==============================================================
``examples``   Packages used by the example scripts (MNE-Python, nilearn, ...)
``docs``       Everything needed to build this documentation
``test``       The test runner (pytest)
``dev``        All of the above
=============  ==============================================================

GPU support
-----------

WhoBPyT runs on the CPU by default. To use an NVIDIA GPU, install a
CUDA-enabled build of PyTorch by following the instructions at
`pytorch.org <https://pytorch.org/get-started/locally/>`_ *before* installing
WhoBPyT.

Next steps
----------

* Work through :ref:`ex-getting-started`.
* Browse the :doc:`cookbook` and the :ref:`examples gallery <examples-gallery>`.
* Prefer not to install anything? Use :doc:`running_in_codespaces` or
  :doc:`running_in_colab`.
