======================================
Running whobpyt in GitHub Codespaces
======================================

GitHub Codespaces is a new feature that allows you to run a full VS Code environment in the cloud and access it from your browser.

We have added pre-made configurations for you to use `whobpyt` in GitHub Codespaces, which takes care of cloning, installing python, setting up the environment, and installing the pertinent packages for you.

Steps to use `whobpyt` in GitHub Codespaces:
1. Go to [the GitHub page for the repo](https://github.com/griffithslab/whobpyt) (or any specific branch you want to explore).
2. Click on the green "Code" button on the top right and select the second tab "Codespaces".
3. In front of the first option "Codespaces", click on the three-dot button.
4. Select "New with options".
5. In the new page, select your desired branch. From the second field, choose either of the three configurations we have provided:
    a. `WhoBPyT-CPU` for a CPU-only environment.
    b. `WhoBPyT-GPU` for a GPU-enabled environment.
    c. `WhoBPyT-docs` for a CPU-only environment that also builds the `html` documentation (useful for exploring and debugging the documentation)
6. Choose your desired region and machine type.
7. Click on "Create codespace" and wait for the environment to be created. This should take 1-2 minutes.
8. Once the environment is created, the python environment will install the necessary packages. This should take 4-5 minutes.
9. Your configured codespace should be ready. Now you get a full VS Code environment in your browser. You can now explore the code, run the examples, and even edit the code and commit your changes back to the repo.

Now you should be good to continue with the rest of the example code in `.py` files, `.ipynb` notebooks, and experiment with new ideas. 


**Notes**:

- The codespaces environment is ephemeral, meaning that it will be deleted after 30 minutes of inactivity. You can always create a new codespace from the same branch and continue your work. You can change this idle timeout settings according to [this guideline](https://docs.github.com/en/codespaces/customizing-your-codespace/setting-your-timeout-period-for-github-codespaces).
- Given the full implementation of VS Code on the codespaces instance with a full Azure Virtual Machine (VM) in the backend, you can use the full power of VS Code, including the terminal, debugging, GitHub Copilot, and even the integrated Jupyter notebook. You can find more information about the VS Code integration with GitHub Codespaces [here](https://docs.github.com/en/codespaces/developing-in-codespaces/using-visual-studio-code-in-codespaces).