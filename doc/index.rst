:html_theme.sidebar_secondary.remove:

.. title:: WhoBPyT

.. The page title must be in rST for it to show in next/prev page buttons.
   Therefore we add a special style rule to only this page that hides h1 tags

.. raw:: html

    <style type="text/css">h1 {display:none;}</style>

WhoBPyT Homepage
===================

.. LOGO

.. image:: _static/whobpyt_logo_shire.png
   :alt: WhoBPyT
   :class: logo, mainlogo, only-light
   :align: center

.. image:: _static/whobpyt_logo_feet_dark.png
   :alt: WhoBPyT
   :class: logo, mainlogo, only-dark
   :align: center

.. rst-class:: text-center font-weight-light my-4
   
   *WhoBPyT* is a PyTorch-based Python library for mathematical modelling of large-scale brain network dynamics, obtuse literary allusion, and model-based analysis of neuroimaging and neurophysiology data. It is developed primarily by researchers in the `Whole-Brain Modelling Group <https://www.grifflab.com>`_ at the `CAMH Krembil Centre for Neuroinformatics <https://www.krembilneuroinformatics.ca>`_ & University of Toronto.

.. frontpage gallery is added by a conditional in _templates/layout.html

.. toctree::
   :hidden:

   About <about_whobpyt/index>
   Documentation <documentation/index>
   Contribute <development/index>
   handpicked_gallery   


.. meta::
   :page-layout: wide        # removes both sidebars and lets the body span the page

