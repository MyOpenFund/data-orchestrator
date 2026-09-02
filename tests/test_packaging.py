"""Packaging identity tests for the MyOpenFund/data-orchestrator rename.

The repo was renamed and transferred (jeulinmarc/RAGDataOrchestrator ->
MyOpenFund/data-orchestrator) but the Python import package stays
``rag_orchestrator`` (out of scope). These tests pin the two things that
*do* change: the distribution/console-script identity in ``pyproject.toml``
(new ``data-orchestrator`` command, ``rag-orchestrator`` kept one release as
a compat alias so existing scripts/muscle-memory don't break) and the
argparse ``prog`` shown in ``--help``/error output, so a regression in
either is caught even though neither is exercised by the rest of the suite.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from rag_orchestrator import cli

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
    assert scripts["data-orchestrator"] == "rag_orchestrator.cli:main"
    assert scripts["rag-orchestrator"] == "rag_orchestrator.cli:main"


def test_cli_prog_is_data_orchestrator(capsys):
    """argparse's ``prog`` drives the usage/error text a user actually sees
    (e.g. ``data-orchestrator: error: ...``); it must match the renamed
    console script, not the old repo name or the compat-alias command."""
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    out = capsys.readouterr().out
    assert out.startswith("usage: data-orchestrator ")
