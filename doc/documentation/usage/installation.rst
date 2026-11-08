.. _installation:

============
Installation
============

WhoBPyT needs Python 3.10 or newer. We recommend installing it into its own
conda (or virtualenv) environment.

1. Create and activate an environment:

   .. code-block:: console

      $ conda create -n whobpyt python=3.10 git pip
      $ conda activate whobpyt

2. Clone the latest version from GitHub:

   .. code-block:: console

      $ git clone https://github.com/GriffithsLab/whobpyt
      $ cd whobpyt

3. Install the dependencies and WhoBPyT itself (in editable mode, so that
   changes you pull or make are picked up automatically):

   .. code-block:: console

      $ pip install -r requirements.txt
      $ pip install -e .

4. Check that it worked:

   .. code-block:: console

      $ python -c "import whobpyt; from whobpyt.run import ModelFitting; print('WhoBPyT OK')"

GPU support
-----------

WhoBPyT runs on the CPU by default. To use an NVIDIA GPU, install a
CUDA-enabled build of PyTorch by following the instructions at
`pytorch.org <https://pytorch.org/get-started/locally/>`_ *before* step 3.

Next steps
----------

* Work through :ref:`ex-getting-started`.
* Browse the :doc:`cookbook` and the :ref:`examples gallery <examples-gallery>`.
* Prefer not to install anything? Use :doc:`running_in_codespaces` or
  :doc:`running_in_colab`.
