# Contributing to WhoBPyT Codebase
Authors: Andrew Clappison, Kevin Kadak, John Griffiths

The below instuctions outline how to properly adhere to version control for contributing to the WhoBPyT repo.

## Setting Up (Done Once):

- **Downloading your Fork**
  - Must have already configured an authentication key and have forked the repository on github.com; ensure your fork is up-to-date with the whobpyt/dev branch from which your new branch will be created.
  - Open terminal and go to the desired directory.
  - `git clone git@github.com:<<your_github_account>>/whobpyt.git`

- **Adding Upstream**
  - `cd whobpyt`
  - `git remote add upstream https://github.com/GriffithsLab/whobpyt.git`
  - `git fetch upstream`

## Coding Cycle (Done for each new feature or group of features):

- **Creating a New Branch**
  - `git fetch upstream`
  - `git checkout --track upstream/dev`
  - `git push origin dev`
  - `git checkout -b <<new_branch_name>>`

- **Editing Code**
  - Add/Delete/Edit code

- **Testing (WhoBPyT Sphinx Examples should run successfully on Linux, but may fail to run on Windows)**
  - Optionally: Rename sphinx examples ending in “r” to “x” if it is not relevant to the code changes done (for quicker debugging). Example: “eg001r...” to “eg001x...”.
  - `cd doc`
  - `make clean`
  - `make html`
  - Open and inspect in a web browser: whobpyt/doc/_build/html/html.txt
  - Additional other testing may also be advised.

- **Committing Code**
  - `git status`
  - `git add <<edited_file_names>>`
  - `git commit -m “<<commit_message>>”`

- **Pushing Code**
  - `git push --set-upstream origin <<new_branch_name>>`

- **Creating a pull request**
  - On github.com do a pull request from the new branch on your fork to the main repo’s dev branch. If there is a merging conflict, it will have to be addressed before proceeding.

