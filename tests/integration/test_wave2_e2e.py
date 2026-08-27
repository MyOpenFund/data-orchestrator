"""End-to-end wave 2: vault-selected documents through the full chain via the CLI."""
import psycopg2
import pytest
from qdrant_client import QdrantClient

from rag_orchestrator import cli

from .conftest import insert_documents

pytestmark = pytest.mark.integration

COLL = "central-bank-e5b-v1"


def _seed(pg_url, corpus_dir):
    insert_documents(pg_url, [
        {"doc_id": "doc-text", "corpus": "central-bank", "source_code": "us",
         "doc_type": "C1", "year": 2015, "local_path": "text.pdf"},
        {"doc_id": "doc-md", "corpus": "central-bank", "source_code": "ecb",
         "doc_type": "A3", "year": 2020, "local_path": "note.md"},
        {"doc_id": "doc-gone", "corpus": "central-bank", "source_code": "fr",
         "doc_type": "C1", "year": 2019, "local_path": "missing.pdf"},
        {"doc_id": "doc-other-corpus", "corpus": "company", "source_code": "edgar",
         "local_path": "text.pdf"},
    ])


def _cli(pg_url, qdrant_addr, corpus_dir, extra_args=()):
    import os
    os.environ["DATABASE_URL"] = pg_url
    os.environ["CB_CORPUS_ROOT"] = str(corpus_dir)
    os.environ["QDRANT_HOST"] = qdrant_addr[0]
    os.environ["QDRANT_PORT"] = str(qdrant_addr[1])
    return cli.main(["vault", "--corpus", "central-bank", *extra_args])


def test_cli_vault_ingests_selection_and_records_state(
    clean_state, qdrant_addr, corpus_dir
):
    _seed(clean_state, corpus_dir)
    rc = _cli(clean_state, qdrant_addr, corpus_dir)
    assert rc == 0
    conn = psycopg2.connect(clean_state)
    with conn.cursor() as cur:
        cur.execute("SELECT doc_id FROM rag_ingestions ORDER BY doc_id")
        done = [r[0] for r in cur.fetchall()]
    conn.close()
    # doc-gone errored (missing file), doc-other-corpus filtered out by corpus.
    assert done == ["doc-md", "doc-text"]
    client = QdrantClient(host=qdrant_addr[0], port=qdrant_addr[1])
    assert client.count(COLL).count > 0


def test_cli_vault_second_run_selects_nothing(clean_state, qdrant_addr, corpus_dir):
    _seed(clean_state, corpus_dir)
    _cli(clean_state, qdrant_addr, corpus_dir)
    before = QdrantClient(host=qdrant_addr[0], port=qdrant_addr[1]).count(COLL).count
    rc = _cli(clean_state, qdrant_addr, corpus_dir)
    assert rc == 0
    after = QdrantClient(host=qdrant_addr[0], port=qdrant_addr[1]).count(COLL).count
    assert after == before  # anti-join found nothing new (doc-gone re-errors)


def test_cli_vault_filters(clean_state, qdrant_addr, corpus_dir):
    _seed(clean_state, corpus_dir)
    rc = _cli(clean_state, qdrant_addr, corpus_dir, ["--doctypes", "A3"])
    assert rc == 0
    conn = psycopg2.connect(clean_state)
    with conn.cursor() as cur:
        cur.execute("SELECT doc_id FROM rag_ingestions")
        assert [r[0] for r in cur.fetchall()] == ["doc-md"]
    conn.close()
