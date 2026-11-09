"""Every public sub-package imports cleanly."""

import importlib

import pytest

MODULES = [
    "whobpyt",
    "whobpyt.datatypes",
    "whobpyt.models.jansen_rit",
    "whobpyt.optimization",
    "whobpyt.optimization.custom_cost_JR",
    "whobpyt.run",
    "whobpyt.visualization",
    "whobpyt.datasets.fetchers",
]


@pytest.mark.parametrize("name", MODULES)
def test_import(name):
    importlib.import_module(name)


def test_import_is_quiet(capfd):
    """Importing and building a model must not print anything (e.g. git errors)."""
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-c",
         "from whobpyt.datatypes.abstract_neural_model import get_git_commit_hash as g; g()"],
        capture_output=True, text=True, cwd="/",
    )
    assert out.returncode == 0
    assert out.stderr == ""
