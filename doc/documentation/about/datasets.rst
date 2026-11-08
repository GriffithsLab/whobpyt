.. _datasets:

Datasets
========

The examples in the :ref:`examples gallery <examples-gallery>` use small,
pre-processed example datasets. Each one has a fetcher function in
``whobpyt.datasets.fetchers`` that downloads it the first time it is needed:

.. code-block:: python

   from whobpyt.datasets.fetchers import fetch_egtmseeg

   data_dir = fetch_egtmseeg()   # returns the folder the files were saved to

By default, files are saved under ``~/.whobpyt/data/<dataset name>``. Every
fetcher also accepts ``dest_folder=`` to save somewhere else and
``redownload=True`` to replace an existing copy. Most datasets are hosted on
the `Open Science Framework (OSF) <https://osf.io>`__.

The :ref:`getting started tutorial <ex-getting-started>` makes its own toy
data and needs no download.


Overview
--------

.. list-table::
   :header-rows: 1
   :widths: 22 26 52

   * - Fetcher
     - Used by
     - Contents
   * - ``fetch_egtmseeg``
     - :ref:`ex-tmseeg`
     - One subject's TMS-EEG epochs, lead field, and stimulation weights; a
       200-region Schaefer parcellation (structural connectivity and region
       centroids); and a pre-computed fitting result.
   * - ``fetch_egmomi2023``
     - :ref:`ex-momi2023`
     - Grand-average and single-trial TMS-EEG data; per-subject lead fields
       and fitting results for six subjects; structural connectivity,
       distance matrix and annotation files for the 200-region Schaefer
       parcellation.
   * - ``fetch_egmomi2025``
     - :ref:`ex-momi2025`
     - Intracerebral stimulation data (sEEG epochs and evoked scalp EEG),
       anatomical files (BEM, source space, head transform, parcellation),
       an example fitting result, and inputs for the virtual dissection
       analysis.
   * - ``fetch_egismail2025``
     - Replicating Ismail et al. 2025
     - Evoked MEG for verb generation and noise trials, empirical and
       simulated source and sensor time series, lead field, structural
       weights and distances.
   * - ``fetch_egismail2026``
     - :ref:`ex-ismail2026`
     - The same kinds of files as ``fetch_egismail2025``.


File details
------------

``fetch_egtmseeg`` → ``~/.whobpyt/data/eg__tmseeg``
   ``Subject_1_low_voltage.fif`` (MNE epochs), ``Subject_1_low_voltage_lf.npy``
   (lead field), ``stim_weights.npy`` (which regions are stimulated),
   ``Schaefer2018_200Parcels_7Networks_count.csv`` (structural connectivity),
   ``Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.txt``
   (region coordinates), ``Subject_1_low_voltage_fittingresults_stim_exp.pkl``
   (a saved :class:`~whobpyt.run.model_fitting.ModelFitting` object).

``fetch_egmomi2023`` → ``~/.whobpyt/data/eg__momi2023``
   ``sub_<n>_fittingresults_stim_exp.pkl`` and ``sub_<n>_leadfield`` for
   subjects 0-5, ``leadfield``, ``stim_weights.npy``, ``real_EEG``,
   ``only_high_trial.mat``, ``all_avg.mat_avg_high_epoched``,
   ``Schaefer2018_200Parcels_7Networks_count.csv``,
   ``Schaefer2018_200Parcels_7Networks_distance.csv``, and the left and right
   hemisphere ``.annot`` parcellation files.

``fetch_egmomi2025`` → ``~/.whobpyt/data/eg__momi2025``
   Organised into sub-folders: ``anatomical/``, ``calculate_distance/``,
   ``empirical_data/``, ``example_fittingresults/`` and
   ``virtual_dissection/``.

``fetch_egismail2025`` → ``~/.whobpyt/data/eg__ismail2025`` and ``fetch_egismail2026`` → ``~/.whobpyt/data/eg__ismail2026``
   ``verb_evoked.npy`` and ``noise_evoked.npy`` (evoked MEG),
   ``emp_*_source.npy`` and ``sim_*_source.npy`` / ``sim_*_sensor.npy``
   (empirical and simulated time series), ``leadfield_3d.mat``,
   ``weights.csv`` (structural connectivity), ``distance.txt``, and
   ``info.pkl`` (MNE measurement info).


Using your own data
-------------------

To model your own recordings you need, in a consistent parcellation:

* a structural connectivity matrix (regions × regions),
* a distance matrix (regions × regions), used to compute conduction delays,
* for EEG/MEG, a lead field matrix (channels × regions), and
* the recording itself, as a channels × time array.

See the :doc:`/documentation/usage/cookbook` for how to load these into
WhoBPyT.
