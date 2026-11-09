.. _releasing:

Making a release
================

Releases are published to `PyPI <https://pypi.org>`__ and get their own
frozen copy of the documentation, both automatically, when a version tag is
pushed.

One-time setup
--------------

1. On PyPI, add a *trusted publisher* for this repository at
   https://pypi.org/manage/account/publishing/ with owner ``GriffithsLab``,
   repository ``whobpyt``, workflow ``release.yml`` and environment ``pypi``.
   No API token is needed.
2. In the GitHub repository settings, under **Environments**, create an
   environment called ``pypi``. You can require a maintainer's approval
   there before anything is published.

Each release
------------

1. On ``main``, set the version in ``whobpyt/__init__.py``, for example
   ``__version__ = "0.1.0"``.
2. Add the new version to ``doc/_static/versions.json`` so it appears in the
   version switcher, and mark it as the preferred one:

   .. code-block:: json

      {
        "name": "0.1.0 (stable)",
        "version": "0.1.0",
        "url": "https://griffithslab.github.io/whobpyt/0.1.0/",
        "preferred": true
      }

   Remove ``"preferred": true`` from the previous release.
3. Merge these changes, then tag the merge commit and push the tag:

   .. code-block:: console

      $ git tag v0.1.0
      $ git push origin v0.1.0

   The tag must match the version exactly, or the release workflow stops.
   Creating a release on GitHub with this tag works too.
4. The **Release** workflow builds the package and publishes it to PyPI,
   and the **Docs** workflow publishes the documentation at
   ``https://griffithslab.github.io/whobpyt/0.1.0/``.
5. Back on ``main``, set the version to the next development version, for
   example ``__version__ = "0.2.0.dev0"``.

The documentation at the site root always shows the latest ``main`` branch,
labelled "dev" in the version switcher.
