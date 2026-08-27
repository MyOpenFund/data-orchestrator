"""Probe end-to-end: unprobed vault rows get facts; probed rows are untouched."""
import psycopg2
import pytest

from rag_orchestrator import cli

from .conftest import insert_documents, write_note_md, write_text_pdf

pytestmark = pytest.mark.integration


@pytest.fixture()
def corpus_dir(tmp_path):
    """Vault-realistic corpus root, overriding conftest's flat wave-1 fixture.

    Real vault rows have local_path values like "data/raw/<doc_id>.pdf"
    (relative to the cb_corpus REPO root), while CB_CORPUS_ROOT points at the
    data dir that already contains raw/ — so fixtures must live under
    <corpus_dir>/raw/... with CB_CORPUS_ROOT=<corpus_dir> to actually exercise
    the CB_CORPUS_ROOT/local_path join (F6), instead of masking it with bare
    filenames.
    """
    d = tmp_path / "corpus"
    d.mkdir()
    write_text_pdf(d / "raw" / "text.pdf")
    write_note_md(d / "raw" / "note.md")
    # "data/raw/gone.pdf" deliberately left absent.
    return d


def test_probe_fills_facts_and_is_resumable(clean_state, qdrant_addr, corpus_dir):
    import os
    os.environ["DATABASE_URL"] = clean_state
    os.environ["CB_CORPUS_ROOT"] = str(corpus_dir)
    insert_documents(clean_state, [
        {"doc_id": "p1", "corpus": "central-bank", "local_path": "data/raw/text.pdf"},
        {"doc_id": "p2", "corpus": "central-bank", "local_path": "data/raw/note.md"},
        {"doc_id": "p3", "corpus": "central-bank", "local_path": "data/raw/gone.pdf"},
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


def test_probe_excludes_deleted_missing_path_and_other_corpus(clean_state, qdrant_addr, corpus_dir):
    """Rows the SELECT's WHERE clause must skip stay untouched and uncounted:
    soft-deleted, missing local_path, and a different corpus."""
    import os

    from rag_orchestrator import vault as vault_mod
    from rag_orchestrator.probe import run_probe

    os.environ["DATABASE_URL"] = clean_state
    os.environ["CB_CORPUS_ROOT"] = str(corpus_dir)
    insert_documents(clean_state, [
        {"doc_id": "n1", "corpus": "central-bank", "local_path": "data/raw/text.pdf"},
        {"doc_id": "n2", "corpus": "central-bank", "local_path": "data/raw/note.md"},
        {"doc_id": "excl-deleted", "corpus": "central-bank", "local_path": "data/raw/text.pdf",
         "deleted_at": "2020-01-01T00:00:00Z"},
        {"doc_id": "excl-nopath", "corpus": "central-bank", "local_path": None},
        {"doc_id": "excl-corpus", "corpus": "company", "local_path": "data/raw/text.pdf"},
    ])

    conn = vault_mod.connect()
    try:
        stats = run_probe(conn, "central-bank")
    finally:
        conn.close()

    # Only the two legitimate central-bank rows were selected/probed at all.
    assert stats == {"probed": 2, "skipped": 0, "errors": 0}

    check_conn = psycopg2.connect(clean_state)
    with check_conn.cursor() as cur:
        cur.execute("SELECT doc_id, has_text_layer, page_count FROM documents ORDER BY doc_id")
        rows = {r[0]: (r[1], r[2]) for r in cur.fetchall()}
    check_conn.close()

    assert rows["n1"] == (True, 2)
    assert rows["n2"] == (True, None)
    assert rows["excl-deleted"] == (None, None)
    assert rows["excl-nopath"] == (None, None)
    assert rows["excl-corpus"] == (None, None)


def test_probe_savepoint_isolates_poisoned_row_on_real_postgres(clean_state, qdrant_addr, corpus_dir):
    """Real-Postgres repro of the transaction-poisoning bug this branch fixes.

    A trigger raises on UPDATE for one doc mid-batch. Without a per-doc
    SAVEPOINT, that raise aborts the whole transaction; every later
    statement in the batch would raise InFailedSqlTransaction, and the
    eventual commit() would silently discard the batch's already-"probed"
    healthy docs even though `stats` already counted them. With the fix,
    the poisoned doc is rolled back to its own savepoint and the healthy
    docs' facts are actually persisted.

    `poisoned` is chosen to sort alphabetically between the two healthy
    doc_ids (SELECT is ORDER BY doc_id), so all three land in one batch.
    """
    import os

    from rag_orchestrator import vault as vault_mod
    from rag_orchestrator.probe import run_probe

    os.environ["DATABASE_URL"] = clean_state
    os.environ["CB_CORPUS_ROOT"] = str(corpus_dir)

    ddl_conn = psycopg2.connect(clean_state)
    ddl_conn.autocommit = True
    with ddl_conn.cursor() as cur:
        cur.execute("""
            CREATE OR REPLACE FUNCTION poison() RETURNS trigger AS $$
            BEGIN
                IF NEW.doc_id = 'poisoned' THEN
                    RAISE EXCEPTION 'poisoned row';
                END IF;
                RETURN NEW;
            END
            $$ LANGUAGE plpgsql;
        """)
        cur.execute("""
            CREATE TRIGGER poison_trg BEFORE UPDATE ON documents
            FOR EACH ROW EXECUTE FUNCTION poison();
        """)

    try:
        insert_documents(clean_state, [
            {"doc_id": "aaa_healthy", "corpus": "central-bank", "local_path": "data/raw/text.pdf"},
            {"doc_id": "poisoned", "corpus": "central-bank", "local_path": "data/raw/note.md"},
            {"doc_id": "zzz_healthy", "corpus": "central-bank", "local_path": "data/raw/note.md"},
        ])

        conn = vault_mod.connect()
        try:
            stats = run_probe(conn, "central-bank", batch=50)
        finally:
            conn.close()

        assert stats == {"probed": 2, "skipped": 0, "errors": 1}

        check_conn = psycopg2.connect(clean_state)
        with check_conn.cursor() as cur:
            cur.execute(
                "SELECT doc_id, has_text_layer, page_count FROM documents ORDER BY doc_id"
            )
            rows = {r[0]: (r[1], r[2]) for r in cur.fetchall()}
        check_conn.close()

        # This is the assertion that fails WITHOUT the per-doc SAVEPOINT fix:
        # the aborted transaction would silently drop these two healthy
        # docs' UPDATEs at the final commit, even though `stats` already
        # counted them as "probed".
        assert rows["aaa_healthy"] == (True, 2)
        assert rows["zzz_healthy"] == (True, None)
        assert rows["poisoned"] == (None, None)
    finally:
        with ddl_conn.cursor() as cur:
            cur.execute("DROP TRIGGER IF EXISTS poison_trg ON documents;")
            cur.execute("DROP FUNCTION IF EXISTS poison();")
        ddl_conn.close()
