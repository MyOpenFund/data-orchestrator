"""Guards for the unit suite's isolation from any ``.env`` on disk (issue #8)."""
import os

import pytest

from rag_orchestrator import config, vault

from .conftest import DOTENV_KEYS


def test_unit_tests_start_without_any_dotenv_key():
    """No test may inherit a machine's DATABASE_URL / corpus roots / QDRANT_*.

    These keys reach the process through two loaders that run before any test
    body (this repo's ``load_dotenv``, and python-dotenv at ``import
    eigenmind.config`` time). If one survives into a test, assertions become
    machine-dependent and code under test can reach a live service.
    """
    for key in DOTENV_KEYS:
        assert key not in os.environ, f"{key} leaked into the unit suite"


def test_dotenv_on_disk_cannot_make_connect_dial_a_database(tmp_path, monkeypatch):
    """A ``.env`` where the loader would find it must not re-populate a deleted
    DATABASE_URL — the exact failure of issue #8, where ``vault.connect()``
    called ``load_dotenv()`` internally and dialled the production vault.

    ``_LOADED`` is reset first so the loader is in its "not yet run" state: the
    bug only bit when this test file ran before whichever file happened to flip
    the flag, and the isolation must not depend on collection order.
    """
    (tmp_path / ".env").write_text(
        "DATABASE_URL=postgresql://nobody:nope@127.0.0.1:1/nope\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "_LOADED", False)

    def _no_dialling(*args, **kwargs):
        raise AssertionError("psycopg2.connect() must never run in the unit suite")

    monkeypatch.setattr(vault.psycopg2, "connect", _no_dialling)

    with pytest.raises(RuntimeError, match="DATABASE_URL is not set"):
        vault.connect()
