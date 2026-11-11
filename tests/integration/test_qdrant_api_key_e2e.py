"""The orchestrator against a Qdrant that requires an API key."""
import os

import psycopg2
import pytest

from data_orchestrator import cli
from tests.integration.conftest import IT_QDRANT_KEY
# corpus_dir: the vault-realistic fixture (files under <root>/raw/...), reused
# from the wave-2 module, not conftest's flat disk-source one.
from tests.integration.test_wave2_e2e import _seed, corpus_dir  # noqa: F401

pytestmark = pytest.mark.integration


def _run(monkeypatch, pg_url, addr, corpus_dir, key):
    monkeypatch.setenv("DATABASE_URL", pg_url)
    monkeypatch.setenv("CB_CORPUS_ROOT", str(corpus_dir))
    monkeypatch.setenv("QDRANT_HOST", addr[0])
    monkeypatch.setenv("QDRANT_PORT", str(addr[1]))
    if key is None:
        monkeypatch.delenv("QDRANT_API_KEY", raising=False)
    else:
        monkeypatch.setenv("QDRANT_API_KEY", key)
    return cli.main(["vault", "--corpus", "central-bank"])


def _done(pg_url):
    conn = psycopg2.connect(pg_url)
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM rag_ingestions")
        n = cur.fetchone()[0]
        cur.execute("SELECT extra->>'fatal' FROM runs ORDER BY finished_at DESC LIMIT 1")
        row = cur.fetchone()
    conn.close()
    return n, (row[0] if row else None)


@pytest.mark.parametrize("key", [None, "", "wrong-key"])
def test_a_missing_or_wrong_key_is_a_fatal_with_nothing_ingested(
        clean_pg, qdrant_keyed_addr, corpus_dir, monkeypatch, capsys, key):
    _seed(clean_pg, corpus_dir)
    assert _run(monkeypatch, clean_pg, qdrant_keyed_addr, corpus_dir, key) == 1
    n, fatal = _done(clean_pg)
    assert n == 0
    assert fatal is not None and "check QDRANT_API_KEY" in fatal
    captured = capsys.readouterr()
    assert "check QDRANT_API_KEY" in captured.err
    if key:
        assert key not in captured.out and key not in captured.err
    # Regression: the repo .env (which may hold a real key) was never loaded.
    if key is None:
        assert "QDRANT_API_KEY" not in os.environ


def test_the_right_key_ingests(clean_pg, qdrant_keyed_addr, corpus_dir, monkeypatch, capsys):
    _seed(clean_pg, corpus_dir)
    # Same seed as test_cli_vault_ingests_selection_and_records_state: doc-gone
    # fails 1 of 3 (above the 10 % ratio), so the run is degraded (exit 3).
    assert _run(monkeypatch, clean_pg, qdrant_keyed_addr, corpus_dir, IT_QDRANT_KEY) == 3
    n, fatal = _done(clean_pg)
    assert n == 2 and fatal is None
    captured = capsys.readouterr()
    out = captured.out
    assert IT_QDRANT_KEY not in captured.err
    assert f"Qdrant: 127.0.0.1:{qdrant_keyed_addr[1]} (api key: set)" in out.splitlines()
    assert IT_QDRANT_KEY not in out


def test_a_key_against_a_server_without_auth_still_works(
        clean_state, qdrant_addr, corpus_dir, monkeypatch):
    """The spec's last adversarial case: a key sent to an unkeyed Qdrant is ignored."""
    _seed(clean_state, corpus_dir)
    assert _run(monkeypatch, clean_state, qdrant_addr, corpus_dir, "any-key") == 3
    n, fatal = _done(clean_state)
    assert n == 2 and fatal is None
