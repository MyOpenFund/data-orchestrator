"""End to end on real PostgreSQL: a run whose vault session is terminated
(``pg_terminate_backend``, as a laptop asleep or a dropped link does) stops
instead of going on without the collection lock."""
from __future__ import annotations

import json

import psycopg2
import pytest
from qdrant_client import QdrantClient, models

from data_orchestrator import cli, core, sync_metadata as sm
from data_orchestrator.routing import collection_name
from data_orchestrator.vault import COLLECTION_LOCK_CLASS

from .conftest import insert_documents, write_text_pdf

pytestmark = pytest.mark.integration

N_DOCS = 4


def _terminate(pg_url, mode):
    """Terminate the session holding the collection lock in ``mode``; its pid."""
    conn = psycopg2.connect(pg_url)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT pid FROM pg_locks WHERE locktype = 'advisory' AND classid = %s "
                        "AND mode = %s AND granted", (COLLECTION_LOCK_CLASS, mode))
            (pid,) = cur.fetchone()
            cur.execute("SELECT pg_terminate_backend(%s)", (pid,))
            for _ in range(100):
                cur.execute("SELECT count(*) FROM pg_stat_activity WHERE pid = %s", (pid,))
                if cur.fetchone()[0] == 0:
                    return pid
                import time
                time.sleep(0.05)
    finally:
        conn.close()
    raise AssertionError("the session did not end")


def _lock_free(pg_url, coll):
    conn = psycopg2.connect(pg_url)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_try_advisory_lock(%s, hashtext(%s))",
                        (COLLECTION_LOCK_CLASS, coll))
            return cur.fetchone()[0]
    finally:
        conn.close()


def _env(monkeypatch, pg_url, addr, root, tmp_path):
    monkeypatch.setenv("DATABASE_URL", pg_url)
    monkeypatch.setenv("CB_CORPUS_ROOT", str(root))
    monkeypatch.setenv("QDRANT_HOST", addr[0])
    monkeypatch.setenv("QDRANT_PORT", str(addr[1]))
    monkeypatch.setenv("DATA_ORCHESTRATOR_STATE_DIR", str(tmp_path / "state"))


def _corpus(tmp_path):
    root = tmp_path / "corpus"
    lines = []
    for i in range(N_DOCS):
        write_text_pdf(root / "raw" / "gb" / "A3" / "2016" / f"doc{i}.pdf")
        lines.append(json.dumps({"doc_id": f"doc{i}", "date": "2016-01-01", "title": f"T{i}"}))
    (root / "manifest").mkdir(parents=True)
    (root / "manifest" / "gb.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


def _seed_vault(pg_url):
    insert_documents(pg_url, [
        {"doc_id": f"doc{i}", "corpus": "central-bank", "source_code": "gb", "doc_type": "A3",
         "date": "2016-01-01", "year": 2016, "local_path": f"data/raw/gb/A3/2016/doc{i}.pdf",
         "extra": json.dumps({"date_precision": "month"})}
        for i in range(N_DOCS)])


def _ingested_doc_ids(addr, coll):
    client = QdrantClient(host=addr[0], port=addr[1])
    if not client.collection_exists(coll):
        return set()
    points, _ = client.scroll(coll, limit=1000, with_payload=["doc_id"])
    return {p.payload["doc_id"] for p in points}


@pytest.mark.parametrize("argv", [["vault"], ["cb_corpus"], ["cb_corpus", "--no-resume"]],
                         ids=["vault", "cb_corpus-ledger", "cb_corpus-no-resume"])
def test_an_ingestion_whose_vault_session_is_terminated_stops(
        clean_state, qdrant_addr, tmp_path, monkeypatch, capsys, argv):
    pg = clean_state
    root = _corpus(tmp_path)
    _seed_vault(pg)
    _env(monkeypatch, pg, qdrant_addr, root, tmp_path)
    coll = collection_name("central-bank")
    real = core.ingest_item
    calls = []

    def ingest_then_lose_the_session(*args, **kwargs):
        n = real(*args, **kwargs)
        calls.append(1)
        if len(calls) == 1:
            _terminate(pg, "ShareLock")
        return n

    monkeypatch.setattr(core, "ingest_item", ingest_then_lose_the_session)
    rc = cli.main([*argv, "--corpus", "central-bank"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "VaultSessionLost" in err
    assert len(calls) == 1, "the run went on after its vault session was terminated"
    assert len(_ingested_doc_ids(qdrant_addr, coll)) == 1
    assert _lock_free(pg, coll)
    # The fatal report was written on a new session.
    conn = psycopg2.connect(pg)
    with conn.cursor() as cur:
        cur.execute("SELECT outcome, extra->>'fatal' FROM runs")
        rows = cur.fetchall()
    conn.close()
    assert len(rows) == 1 and rows[0][0] == "failed" and "VaultSessionLost" in rows[0][1]


def test_a_sync_metadata_write_whose_vault_session_is_terminated_stops(
        clean_state, qdrant_addr, tmp_path, monkeypatch, capsys):
    pg = clean_state
    root = _corpus(tmp_path)
    _seed_vault(pg)
    _env(monkeypatch, pg, qdrant_addr, root, tmp_path)
    coll = collection_name("central-bank")
    assert cli.main(["vault", "--corpus", "central-bank"]) == 0
    client = QdrantClient(host=qdrant_addr[0], port=qdrant_addr[1])
    client.delete_payload(coll, keys=["date_start", "date_end", "date_precision"],
                          points=models.Filter(must=[]), wait=True)
    client.create_payload_index(coll, "doc_id", field_schema="keyword", wait=True)
    capsys.readouterr()

    real = sm.read_points
    calls = []

    def read_then_lose_the_session(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            _terminate(pg, "ExclusiveLock")
        return real(*args, **kwargs)

    monkeypatch.setattr(sm, "read_points", read_then_lose_the_session)
    rc = cli.main(["sync-metadata", "--corpus", "central-bank", "--write", "--doc-batch", "1"])
    out, err = capsys.readouterr()
    assert rc == 1
    assert "VaultSessionLost" in err and "can be run again" in err

    def corrected():
        points, _ = client.scroll(coll, limit=1000, with_payload=True)
        return {p.payload["doc_id"] for p in points if "date_end" in p.payload}

    assert corrected() == {"doc0"}, "a write went out after the session was terminated"
    assert _lock_free(pg, coll)  # another run could have taken it

    monkeypatch.setattr(sm, "read_points", real)
    assert cli.main(["sync-metadata", "--corpus", "central-bank", "--write"]) == 0
    assert corrected() == {f"doc{i}" for i in range(N_DOCS)}
    assert "  documents written  : 3" in capsys.readouterr().out
