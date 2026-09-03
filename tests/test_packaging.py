"""Packaging identity tests for the MyOpenFund/data-orchestrator rename.

The repo was renamed and transferred to the MyOpenFund org (see README) and
the Python import package was renamed to match the new repo name. These
tests pin the things that change in ``pyproject.toml``: the distribution and
console-script identity (the ``data-orchestrator`` command, with the
pre-rename command name dropped outright rather than kept as an alias — this
is pre-release, so there is nothing to stay compatible with) and the argparse
``prog`` shown in ``--help``/error output, so a regression in either is caught
even though neither is exercised by the rest of the suite.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from data_orchestrator import cli

PYPROJECT_PATH = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _load_pyproject() -> dict:
    with PYPROJECT_PATH.open("rb") as fh:
        return tomllib.load(fh)


def test_distribution_name_is_data_orchestrator():
    """The distribution must self-identify by its new GitHub repo name."""
    data = _load_pyproject()
    assert data["project"]["name"] == "data-orchestrator"


def test_console_scripts_are_the_new_names_only():
    """Exactly the two intended console scripts, both on the renamed package.

    Pinning the whole key set rather than the absence of one known-bad name
    catches *any* stray entry point — an alias under the pre-rename command
    name, a half-finished addition, a typo'd duplicate — instead of only the
    one alias we happen to have removed. An alias left behind would keep the
    old command working, so nobody's muscle memory (or pinned script) would
    ever surface the rename while the product is still pre-release and free to
    break it."""
    data = _load_pyproject()
    scripts = data["project"]["scripts"]
    assert set(scripts) == {"data-orchestrator", "rag-dashboard"}
    assert scripts["data-orchestrator"] == "data_orchestrator.cli:main"
    assert scripts["rag-dashboard"] == "data_orchestrator.dashboard.__main__:main"


def test_cli_prog_is_data_orchestrator(capsys):
    """argparse's ``prog`` drives the usage/error text a user actually sees
    (e.g. ``data-orchestrator: error: ...``); it must match the renamed
    console script, not the old repo or command name."""
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    out = capsys.readouterr().out
    assert out.split()[1] == "data-orchestrator"


def test_import_resolves_to_this_checkout():
    """``import data_orchestrator`` must resolve to *this* checkout, not some
    other install on the path.

    We deliberately do not assert that importing the pre-rename package name
    raises ``ModuleNotFoundError``: a stale editable install of it elsewhere
    on this machine can still satisfy that import, which would make the
    assertion environment-dependent rather than hermetic. Pinning the
    resolved path of the new package is the hermetic version of the same
    check.
    """
    import data_orchestrator

    repo_root = Path(__file__).resolve().parents[1]
    resolved = Path(data_orchestrator.__file__).resolve()
    assert resolved.is_relative_to(repo_root)
