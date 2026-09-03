"""Packaging identity tests for the MyOpenFund/data-orchestrator rename.

The repo was renamed and transferred to the MyOpenFund org (see README) and
the Python import package was renamed to match the new repo name. These
tests pin the things that change in
``pyproject.toml``: the distribution/console-script identity (new
``data-orchestrator`` command, ``rag-orchestrator`` kept one release as a
compat alias so existing scripts/muscle-memory don't break, now pointing at
the renamed package) and the argparse ``prog`` shown in ``--help``/error
output, so a regression in either is caught even though neither is
exercised by the rest of the suite.
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


def test_console_scripts_include_new_name_and_compat_alias():
    """Both the new ``data-orchestrator`` command and the old ``rag-orchestrator``
    name (kept one release as a compat alias) must resolve to the same entry
    point, so neither a fresh install nor an existing pinned script breaks."""
    data = _load_pyproject()
    scripts = data["project"]["scripts"]
    assert scripts["data-orchestrator"] == "data_orchestrator.cli:main"
    assert scripts["rag-orchestrator"] == "data_orchestrator.cli:main"


def test_cli_prog_is_data_orchestrator(capsys):
    """argparse's ``prog`` drives the usage/error text a user actually sees
    (e.g. ``data-orchestrator: error: ...``); it must match the renamed
    console script, not the old repo name or the compat-alias command."""
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    out = capsys.readouterr().out
    assert out.split()[1] == "data-orchestrator"


def test_import_resolves_to_this_worktrees_package():
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
