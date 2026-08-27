"""End-to-end wave 1: disk-provided items -> eigenmind engine -> Qdrant + vault."""
import psycopg2
import pytest
from qdrant_client import QdrantClient

from rag_orchestrator.core import SourceItem, run_ingest
from rag_orchestrator.vault import VaultLedger

from .conftest import insert_documents

pytestmark = pytest.mark.integration

COLL = "central-bank-e5b-v1"


def _items(corpus_dir):
    return [
        SourceItem("doc-text", corpus_dir / "text.pdf", {"bank_code": "us", "corpus": "central-bank"}),
        SourceItem("doc-empty", corpus_dir / "empty.pdf", {"bank_code": "fr"}),
        SourceItem("doc-md", corpus_dir / "note.md", {"bank_code": "ecb"}),
    ]


def _run(pg_url, qdrant_addr, corpus_dir, items=None):
    from eigenmind.vectordb.store import QdrantStore

    conn = psycopg2.connect(pg_url)
    try:
        ledger = VaultLedger(conn, COLL, "central-bank",
                             "tiny-test-model", "e5-prefixes-v1")
        store = QdrantStore(host=qdrant_addr[0], port=qdrant_addr[1])
        return run_ingest(items or _items(corpus_dir), collection=COLL,
                          ledger=ledger, store=store)
    finally:
        conn.close()


def _rows(pg_url, sql):
    conn = psycopg2.connect(pg_url)
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
    conn.close()
    return rows


def _seed_documents(pg_url):
    insert_documents(pg_url, [
        {"doc_id": "doc-text", "corpus": "central-bank", "source_code": "us"},
        {"doc_id": "doc-empty", "corpus": "central-bank", "source_code": "fr"},
        {"doc_id": "doc-md", "corpus": "central-bank", "source_code": "ecb"},
    ])


def test_e2e_ingest_writes_qdrant_then_vault(clean_state, qdrant_addr, corpus_dir):
    _seed_documents(clean_state)
    stats = _run(clean_state, qdrant_addr, corpus_dir)
    assert stats.docs_ingested == 2 and stats.docs_empty == 1 and stats.docs_error == 0

    client = QdrantClient(host=qdrant_addr[0], port=qdrant_addr[1])
    assert client.count(COLL).count == stats.chunks_written

    rows = dict(_rows(clean_state,
        "SELECT doc_id, chunk_count FROM rag_ingestions ORDER BY doc_id"))
    assert set(rows) == {"doc-text", "doc-empty", "doc-md"}
    assert rows["doc-empty"] == 0
    # PDF pages made it into the payload.
    pts, _ = client.scroll(COLL, limit=200, with_payload=True)
    pages = {p.payload["page"] for p in pts if p.payload["doc_id"] == "doc-text"}
    assert pages and all(isinstance(p, int) for p in pages)


def test_e2e_rerun_is_idempotent_noop(clean_state, qdrant_addr, corpus_dir):
    _seed_documents(clean_state)
    first = _run(clean_state, qdrant_addr, corpus_dir)
    second = _run(clean_state, qdrant_addr, corpus_dir)
    assert second.docs_skipped_resume == 3 and second.docs_ingested == 0
    client = QdrantClient(host=qdrant_addr[0], port=qdrant_addr[1])
    assert client.count(COLL).count == first.chunks_written


def test_e2e_crash_between_qdrant_and_vault_heals(clean_state, qdrant_addr, corpus_dir):
    _seed_documents(clean_state)
    first = _run(clean_state, qdrant_addr, corpus_dir)
    # Simulate the crash window: Qdrant has the points, the vault row is gone.
    conn = psycopg2.connect(clean_state); conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("DELETE FROM rag_ingestions WHERE doc_id = 'doc-text'")
    conn.close()
    second = _run(clean_state, qdrant_addr, corpus_dir)
    assert second.docs_ingested == 1  # re-ingested, deterministic ids overwrote
    client = QdrantClient(host=qdrant_addr[0], port=qdrant_addr[1])
    assert client.count(COLL).count == first.chunks_written  # no duplicates
    assert len(_rows(clean_state, "SELECT doc_id FROM rag_ingestions")) == 3


def test_e2e_doc_unknown_to_vault_is_isolated_fk_error(clean_state, qdrant_addr, corpus_dir):
    # documents table deliberately NOT seeded for doc-md.
    insert_documents(clean_state, [
        {"doc_id": "doc-text", "corpus": "central-bank", "source_code": "us"},
        {"doc_id": "doc-empty", "corpus": "central-bank", "source_code": "fr"},
    ])
    stats = _run(clean_state, qdrant_addr, corpus_dir)
    assert stats.docs_error == 1  # doc-md: FK violation isolated, run completed
    assert stats.docs_ingested == 1 and stats.docs_empty == 1
