.. _contributing:

Contributing Guide
==================

Welcome to whobpyt! We are delighted that you are interested in contributing.
whobpyt is a community-driven project, and contributions of any kind are warmly
welcomed — whether you are a seasoned developer, a neuroscientist, or someone
taking their first steps in open-source science.

This guide is inspired by contributor guides from
`MNE-Python <https://mne.tools/stable/development/contributing.html>`__ and
`Nilearn <https://nilearn.github.io/stable/development/contributing.html>`__,
two excellent examples of community-driven neuroimaging software.

- **Issue tracker:** https://github.com/GriffithsLab/whobpyt/issues
- **Discussions:** https://github.com/GriffithsLab/whobpyt/discussions


.. _ways-to-contribute:

Ways to Contribute
------------------

There are many ways to contribute to whobpyt beyond writing code:

- **Report bugs** — Open an issue describing what went wrong and how to
  reproduce it.
- **Suggest enhancements** — Open an issue or discussion proposing a new
  feature or improvement.
- **Write or improve documentation** — Fix typos, clarify explanations, add
  missing docstrings, or write new tutorial pages.
- **Add examples or tutorials** — Share a Jupyter notebook demonstrating a use
  case that others would find helpful.
- **Triage issues** — Help by reproducing reported bugs, labelling issues, or
  pointing reporters to relevant existing discussions.
- **Review pull requests** — Provide constructive feedback on open PRs, even if
  you are not a core maintainer.
- **Contribute code** — Fix bugs, implement new models or features, or improve
  existing implementations.

No contribution is too small. Even fixing a typo helps!


.. _code-style:

Code Style
----------

whobpyt follows `PEP 8 <https://peps.python.org/pep-0008/>`__ style
conventions. We use `Black <https://black.readthedocs.io/>`__ for automatic
code formatting and `flake8 <https://flake8.pycqa.org/>`__ for linting.

**Formatting**

Before opening a pull request, please format your code with Black::

   pip install black
   black .

**Docstrings**

All public functions, classes, and methods should include docstrings following
the `NumPy docstring standard
<https://numpydoc.readthedocs.io/en/latest/format.html>`__.  This is the
convention used throughout the codebase and is required for the API reference
documentation to build correctly.

**Pre-commit (optional but recommended)**

We recommend using `pre-commit <https://pre-commit.com/>`__ to run formatting
and linting checks automatically on every commit::

   pip install pre-commit
   pre-commit install


.. _pull-request-process:

Pull Request Process
--------------------

Follow these steps to submit a pull request:

1. **Fork the repository** — Click *Fork* on the
   `whobpyt GitHub page <https://github.com/GriffithsLab/whobpyt>`__ to create
   your own copy.

2. **Clone your fork** ::

      git clone https://github.com/<your-username>/whobpyt.git
      cd whobpyt

3. **Create a feature branch** — Use a short, descriptive name::

      git checkout -b fix/my-bug-fix
      # or
      git checkout -b feature/new-model

4. **Set up a development environment** ::

      pip install -e ".[dev]"

5. **Make your changes** — Keep commits focused. Write clear, concise commit
   messages.

6. **Run the tests** to ensure nothing is broken::

      pytest tests/

7. **Push your branch** to your fork::

      git push origin fix/my-bug-fix

8. **Open a pull request** against the ``main`` branch of
   ``GriffithsLab/whobpyt``.  Fill in the pull request template, describing
   *what* changed and *why*.

9. **Respond to review feedback** — A maintainer will review your PR.  Address
   any requested changes and push additional commits to your branch.  Once
   approved, a maintainer will merge it.

**What to expect**

- PRs are typically reviewed within a few weeks.
- You may be asked to make changes before a PR is accepted — this is a normal
  part of collaborative development and is not a rejection.
- Automated checks (CI) must pass before a PR can be merged.


.. _issue-reporting:

Issue Reporting
---------------

**Reporting a bug**

Before opening a new bug report, please search the
`issue tracker <https://github.com/GriffithsLab/whobpyt/issues>`__ to see
whether the problem has already been reported.

A helpful bug report includes:

- A clear, descriptive title.
- A minimal, self-contained code example that reproduces the problem
  (*minimal reproducible example*).
- The full error traceback (if applicable).
- Your environment details: Python version, whobpyt version, operating system,
  and relevant dependency versions (e.g. PyTorch).

**Suggesting an enhancement**

Open an issue or start a
`discussion <https://github.com/GriffithsLab/whobpyt/discussions>`__ describing:

- The use case or problem the enhancement would address.
- Your proposed solution or approach (if you have one).
- Any alternatives you have considered.


.. _community-standards:

Community Standards
-------------------

whobpyt is committed to providing a welcoming and inclusive environment for
everyone. All participants are expected to adhere to respectful and
constructive communication in all project spaces (issues, pull requests,
discussions, and any other forums).

Please:

- Be respectful and considerate of differing viewpoints and experiences.
- Use welcoming and inclusive language.
- Accept constructive criticism graciously.
- Focus on what is best for the community and the project.

Unacceptable behavior includes harassment, discrimination, or personal attacks
of any kind. If you experience or witness such behavior, please report it to
the maintainers via email or by opening a confidential issue.

We follow the spirit of the
`Contributor Covenant Code of Conduct <https://www.contributor-covenant.org/version/2/1/code_of_conduct/>`__
(version 2.1).


.. _getting-help:

Getting Help
------------

If you have a question about using or developing whobpyt, the best places to
ask are:

- **GitHub Discussions** — https://github.com/GriffithsLab/whobpyt/discussions
  — For general questions, ideas, and open-ended conversations.
- **GitHub Issues** — https://github.com/GriffithsLab/whobpyt/issues
  — For bug reports and concrete feature requests.
- **Lab website** — https://griffithslab.github.io/
  — For information about the Griffiths Lab and related projects.

We do our best to respond promptly, but please be patient — whobpyt is
maintained by a small team of researchers.

