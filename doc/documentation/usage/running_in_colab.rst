.. _colab:

================================
Running WhoBPyT in Google Colab
================================

`Google Colab <https://colab.research.google.com>`_ runs Jupyter notebooks in
the cloud for free, with optional GPU access. Every example in the
:ref:`examples gallery <examples-gallery>` can be run there.

1. Open the example you want in the gallery, scroll to the bottom, and click
   **Download Jupyter notebook**.
2. In Colab, choose **File > Upload notebook** and select the downloaded
   ``.ipynb`` file.
3. Add a new cell at the top of the notebook and run it to install WhoBPyT:

   .. code-block:: text

      !pip install "whobpyt[examples] @ git+https://github.com/GriffithsLab/whobpyt.git"

   This takes a few minutes. If Colab asks you to restart the session
   afterwards, do so, then carry on from the next cell.

4. Run the rest of the notebook as normal. Example data are downloaded
   automatically the first time they are needed.

To use a GPU, choose **Runtime > Change runtime type** and select a GPU
before you run the notebook.
