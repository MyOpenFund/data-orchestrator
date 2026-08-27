"""Probe end-to-end: unprobed vault rows get facts; probed rows are untouched."""
import psycopg2
import pytest

from rag_orchestrator import cli

from .conftest import insert_documents

pytestmark = pytest.mark.integration


def test_probe_fills_facts_and_is_resumable(clean_state, qdrant_addr, corpus_dir):
    import os
    os.environ["DATABASE_URL"] = clean_state
    os.environ["CB_CORPUS_ROOT"] = str(corpus_dir)
    insert_documents(clean_state, [
        {"doc_id": "p1", "corpus": "central-bank", "local_path": "text.pdf"},
        {"doc_id": "p2", "corpus": "central-bank", "local_path": "note.md"},
        {"doc_id": "p3", "corpus": "central-bank", "local_path": "gone.pdf"},
    ])
    assert cli.main(["probe", "--corpus", "central-bank"]) == 0
    conn = psycopg2.connect(clean_state)
    with conn.cursor() as cur:
        cur.execute("SELECT doc_id, has_text_layer, page_count FROM documents ORDER BY doc_id")
        rows = {r[0]: (r[1], r[2]) for r in cur.fetchall()}
    conn.close()
    assert rows["p1"] == (True, 2)
    assert rows["p2"] == (True, None)
    assert rows["p3"] == (None, None)  # missing file: left unprobed
    # Second run only re-attempts the missing one (resumable by IS NULL).
    assert cli.main(["probe", "--corpus", "central-bank"]) == 0
