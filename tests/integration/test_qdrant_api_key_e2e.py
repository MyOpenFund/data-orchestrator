"""The orchestrator against a Qdrant that requires an API key."""
import os

import psycopg2
import pytest

from data_orchestrator import cli, config
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
    # Order-independent: whatever the loader read must be the fixture's empty file.
    loaded = config.loaded_dotenv()
    assert loaded is None or loaded.stat().st_size == 0, f"real .env loaded: {loaded}"


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


def test_a_key_refused_in_the_middle_of_a_run_reports_what_was_done_before_it(
        clean_pg, qdrant_keyed_addr, corpus_dir, monkeypatch, capsys):
    """The key stops being accepted after the first document's write (the
    server answers a real 401 to the following one): one fatal, and the
    ``runs`` row counts what ``rag_ingestions`` holds, not zeros."""
    from qdrant_client import QdrantClient

    import data_orchestrator.core as core

    host, port = qdrant_keyed_addr
    real_build_store = core.build_store

    def build_store(settings):
        store = real_build_store(settings)
        accepted = store.client.upsert
        refused = QdrantClient(host=host, port=port, api_key="revoked-key", https=False,
                               timeout=5).upsert
        writes = []

        def upsert(*args, **kwargs):
            writes.append(1)
            return (accepted if len(writes) == 1 else refused)(*args, **kwargs)

        store.client.upsert = upsert
        return store

    monkeypatch.setattr(core, "build_store", build_store)
    _seed(clean_pg, corpus_dir)
    assert _run(monkeypatch, clean_pg, qdrant_keyed_addr, corpus_dir, IT_QDRANT_KEY) == 1

    conn = psycopg2.connect(clean_pg)
    with conn.cursor() as cur:
        cur.execute("SELECT doc_id, chunk_count FROM rag_ingestions")
        ingested = cur.fetchall()
        cur.execute("SELECT outcome, exit_code, totals, sources, extra->>'fatal' FROM runs")
        (outcome, exit_code, totals, sources, fatal), = cur.fetchall()
    conn.close()
    assert (outcome, exit_code) == ("failed", 1)
    assert "(HTTP 401): check QDRANT_API_KEY (while writing document" in fatal
    assert "after 1 document(s) ingested in this run" in fatal
    # One document was written and recorded before the refusal; the report says so.
    assert len(ingested) == 1 and ingested[0][1] > 0
    assert totals["docs_new"] == 1 and totals["docs_seen"] >= 2
    assert [s["source_code"] for s in sources if s["docs_new"]] == [
        {"doc-text": "us", "doc-md": "ecb"}[ingested[0][0]]]
    assert sum(s["docs_seen"] for s in sources) == totals["docs_seen"]
    assert sum(s["docs_failed"] for s in sources) == totals["docs_failed"]
    captured = capsys.readouterr()
    assert "Done." in captured.out.splitlines()
    assert "  docs ingested  : 1" in captured.out.splitlines()
    assert captured.err.count("error: fatal: QdrantAuthError") == 1
    assert IT_QDRANT_KEY not in captured.out and IT_QDRANT_KEY not in captured.err
