"""Guards for the unit suite's isolation from any ``.env`` on disk (issue #8)."""
import os
from pathlib import Path

import pytest

from rag_orchestrator import config, vault

from .conftest import DOTENV_KEYS


def test_unit_tests_start_without_any_dotenv_key():
    """Tripwire on the ``isolated_env`` fixture's key-dropping loop: if a key is
    ever removed from DOTENV_KEYS, or the loop stops running, this test fails
    instead of the suite quietly becoming machine-dependent (a real
    DATABASE_URL or corpus root reaching code under test).
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


def test_a_test_that_monkeypatches_a_dotenv_key_restores_the_machine_value(pytester):
    """The environment a test inherits must also be the environment it leaves.

    ``isolated_env`` drops CB_CORPUS_ROOT before each test, so a test doing
    ``monkeypatch.setenv("CB_CORPUS_ROOT", ...)`` records it as *absent*; if
    that undo runs after the fixture's restore, it deletes the machine's real
    value and everything afterwards (the integration suite, a later session in
    the same process) sees no corpus root at all.

    Reproduced deterministically by running a throwaway two-test session, in
    this process, against the real fixture, and reading the environment back
    once that session has finished.
    """
    repo_root = Path(__file__).resolve().parents[1]
    pytester.makeconftest(
        "import sys\n"
        f"sys.path.insert(0, {str(repo_root)!r})\n"
        "from tests.conftest import isolated_env  # noqa: F401 — fixture under test\n"
    )
    pytester.makepyfile(
        """
        import os

        def test_overrides_the_corpus_root(monkeypatch, tmp_path):
            monkeypatch.setenv("CB_CORPUS_ROOT", str(tmp_path))
            assert os.environ["CB_CORPUS_ROOT"] == str(tmp_path)

        def test_sees_no_corpus_root_of_its_own():
            assert "CB_CORPUS_ROOT" not in os.environ
        """
    )

    os.environ["CB_CORPUS_ROOT"] = "the-machines-corpus-root"
    try:
        result = pytester.runpytest("-q", "-p", "no:cacheprovider")
        result.assert_outcomes(passed=2)
        assert os.environ.get("CB_CORPUS_ROOT") == "the-machines-corpus-root"
    finally:
        os.environ.pop("CB_CORPUS_ROOT", None)


def test_a_key_written_straight_into_os_environ_does_not_survive_the_test(pytester):
    """``config.load_dotenv`` assigns to ``os.environ`` directly, so monkeypatch
    records nothing and can undo nothing: only the fixture's snapshot restore
    keeps such a write from outliving the test that made it.

    Same throwaway-session trick: let a test load a real ``.env`` through the
    production loader, then read the environment back once the session is over.
    """
    repo_root = Path(__file__).resolve().parents[1]
    pytester.makeconftest(
        "import sys\n"
        f"sys.path.insert(0, {str(repo_root)!r})\n"
        "from tests.conftest import isolated_env  # noqa: F401 — fixture under test\n"
    )
    pytester.makepyfile(
        """
        import os

        from rag_orchestrator import config

        def test_loads_a_dotenv_through_the_production_loader(tmp_path):
            env = tmp_path / ".env"
            env.write_text("RAGO_TEST_LEAKED=from-the-file", encoding="utf-8")
            assert config.load_dotenv(env) is True
            assert os.environ["RAGO_TEST_LEAKED"] == "from-the-file"
        """
    )

    result = pytester.runpytest("-q", "-p", "no:cacheprovider")
    result.assert_outcomes(passed=1)
    assert "RAGO_TEST_LEAKED" not in os.environ
