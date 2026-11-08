.. _codespaces:

====================================
Running WhoBPyT in GitHub Codespaces
====================================

`GitHub Codespaces <https://github.com/features/codespaces>`_ gives you a full
VS Code environment in the cloud, in your browser. The WhoBPyT repository
includes ready-made configurations that clone the code, install Python, and
install all the dependencies for you.

1. Go to the `WhoBPyT GitHub repository <https://github.com/griffithslab/whobpyt>`_
   (or the branch you want to explore).
2. Click the green **Code** button and select the **Codespaces** tab.
3. Next to **Codespaces**, click the **...** button and select
   **New with options**.
4. Select your branch, then choose one of the configurations:

   * ``WhoBPyT-CPU``: a CPU-only environment.
   * ``WhoBPyT-GPU``: a GPU-enabled environment.
   * ``WhoBPyT-docs``: a CPU-only environment that also builds the HTML
     documentation, useful for working on the docs.

5. Choose a region and machine type.
6. Click **Create codespace**. Creating the environment takes 1-2 minutes, and
   installing the Python packages takes a further 4-5 minutes.

You now have a full VS Code environment in your browser. You can run the
example scripts in ``examples/``, edit the code, and commit your changes back
to GitHub.
