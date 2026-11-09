Datasets
========

Fetchers for the example datasets. Each downloads its dataset the first time
it is called and returns the folder it was saved to. See
:doc:`/documentation/about/datasets` for what each one contains.

.. currentmodule:: whobpyt.datasets.fetchers

.. autofunction:: fetch_egtmseeg
.. autofunction:: fetch_egmomi2023
.. autofunction:: fetch_egmomi2025
.. autofunction:: fetch_egismail2026
.. autofunction:: fetch_eggriffiths2022
.. autofunction:: get_localdefaultdatapath

Older datasets
--------------

.. autofunction:: fetch_hcpl2k8
.. autofunction:: fetch_MomiEtAlELife2023
.. autofunction:: fetch_egismail2025

Download helpers
----------------

.. autofunction:: pull_file
.. autofunction:: pull_folder
