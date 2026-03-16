============
Installation
============

Requirements
------------

WhoBPyT requires **Python 3.9 or higher**.

Quick Install (pip)
-------------------

The easiest way to install WhoBPyT is via pip:

.. code::

   $ pip install whobpyt


Install from Source
-------------------

To install the latest development version directly from GitHub, first create and activate a conda environment:

.. code::

   $ conda create -n "whobpyt" git pip python=3.9
   $ conda activate whobpyt

Clone the repository and install:

.. code::

   $ git clone https://github.com/GriffithsLab/whobpyt
   $ cd whobpyt
   $ pip install .

For an editable (development) install that reflects local code changes immediately:

.. code::

   $ pip install -e .


Optional: Development and Documentation Dependencies
-----------------------------------------------------

To install additional dependencies needed for running tests or building the documentation:

.. code::

   $ pip install -r requirements.txt


Verifying the Installation
--------------------------

After installation, you can verify that WhoBPyT is correctly installed by running the following in a Python session:

.. code:: python

   import whobpyt
   print(whobpyt.__version__)
