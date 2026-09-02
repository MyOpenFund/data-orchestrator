"""End-to-end wave 2: vault-selected documents through the full chain via the CLI."""
import psycopg2
import pytest
from qdrant_client import QdrantClient

from rag_orchestrator import cli
from rag_orchestrator.routing import collection_name

from .conftest import insert_documents, write_note_md, write_text_pdf

pytestmark = pytest.mark.integration


@pytest.fixture()
def corpus_dir(tmp_path):
    """Vault-realistic corpus root, overriding conftest's flat wave-1 fixture.

    Real vault rows have local_path values like "data/raw/us/C1/2010/<doc_id>.pdf"
    (relative to the cb_corpus REPO root), while CB_CORPUS_ROOT points at the
    data dir that already contains raw/ — so fixtures must live under
    <corpus_dir>/raw/... with CB_CORPUS_ROOT=<corpus_dir> to actually exercise
    the CB_CORPUS_ROOT/local_path join (F6), instead of masking it with bare
    filenames.
    """
    d = tmp_path / "corpus"
    d.mkdir()
    write_text_pdf(d / "raw" / "us" / "C1" / "2015" / "text.pdf")
    write_note_md(d / "raw" / "ecb" / "A3" / "2020" / "note.md")
    # "data/raw/fr/C1/2019/missing.pdf" deliberately left absent (doc-gone).
    return d


def _coll():
    """Derive collection name at test runtime from RAGO_EMBEDDING_MODEL."""
    return collection_name("central-bank")


def _seed(pg_url, corpus_dir):
    insert_documents(pg_url, [
        {"doc_id": "doc-text", "corpus": "central-bank", "source_code": "us",
         "doc_type": "C1", "year": 2015, "local_path": "data/raw/us/C1/2015/text.pdf"},
        {"doc_id": "doc-md", "corpus": "central-bank", "source_code": "ecb",
         "doc_type": "A3", "year": 2020, "local_path": "data/raw/ecb/A3/2020/note.md"},
        {"doc_id": "doc-gone", "corpus": "central-bank", "source_code": "fr",
         "doc_type": "C1", "year": 2019, "local_path": "data/raw/fr/C1/2019/missing.pdf"},
        {"doc_id": "doc-other-corpus", "corpus": "company", "source_code": "edgar",
         "local_path": "data/raw/text.pdf"},
    ])


def _cli(monkeypatch, pg_url, qdrant_addr, corpus_dir, extra_args=()):
    """Run the CLI against this test's containers.

    The CLI reads its targets from the process environment, so they are set
    through ``monkeypatch`` rather than ``os.environ`` directly: a bare
    assignment survives the test and leaves a live DATABASE_URL / corpus root
    behind for whatever runs next (issue #8).
    """
    monkeypatch.setenv("DATABASE_URL", pg_url)
    monkeypatch.setenv("CB_CORPUS_ROOT", str(corpus_dir))
    monkeypatch.setenv("QDRANT_HOST", qdrant_addr[0])
    monkeypatch.setenv("QDRANT_PORT", str(qdrant_addr[1]))
    return cli.main(["vault", "--corpus", "central-bank", *extra_args])


def test_cli_vault_ingests_selection_and_records_state(
    clean_state, qdrant_addr, corpus_dir, monkeypatch
):
    _seed(clean_state, corpus_dir)
    rc = _cli(monkeypatch, clean_state, qdrant_addr, corpus_dir)
    assert rc == 0
    conn = psycopg2.connect(clean_state)
    with conn.cursor() as cur:
        cur.execute("SELECT doc_id FROM rag_ingestions ORDER BY doc_id")
        done = [r[0] for r in cur.fetchall()]
    conn.close()
    # doc-gone errored (missing file), doc-other-corpus filtered out by corpus.
    assert done == ["doc-md", "doc-text"]
    client = QdrantClient(host=qdrant_addr[0], port=qdrant_addr[1])
    assert client.count(_coll()).count > 0

    with psycopg2.connect(clean_state) as conn2, conn2.cursor() as cur:
        cur.execute("SELECT tool, outcome FROM runs")
        runs = cur.fetchall()
    assert runs and runs[0][0] == "rag-orchestrator"


def test_cli_vault_second_run_selects_nothing(
    clean_state, qdrant_addr, corpus_dir, monkeypatch
):
    _seed(clean_state, corpus_dir)
    _cli(monkeypatch, clean_state, qdrant_addr, corpus_dir)
    before = QdrantClient(host=qdrant_addr[0], port=qdrant_addr[1]).count(_coll()).count
    rc = _cli(monkeypatch, clean_state, qdrant_addr, corpus_dir)
    # Honest exit code (item 9/C1): the anti-join leaves only doc-gone as a
    # candidate on the second run, and it re-errors (missing file), so this
    # run has docs_ingested == 0 and docs_error > 0 — degraded, exit 3 — not
    # the old always-0 behavior.
    assert rc == 3
    after = QdrantClient(host=qdrant_addr[0], port=qdrant_addr[1]).count(_coll()).count
    assert after == before  # anti-join found nothing new (doc-gone re-errors)


def test_cli_vault_filters(clean_state, qdrant_addr, corpus_dir, monkeypatch):
    _seed(clean_state, corpus_dir)
    rc = _cli(monkeypatch, clean_state, qdrant_addr, corpus_dir, ["--doctypes", "A3"])
    assert rc == 0
    conn = psycopg2.connect(clean_state)
    with conn.cursor() as cur:
        cur.execute("SELECT doc_id FROM rag_ingestions")
        assert [r[0] for r in cur.fetchall()] == ["doc-md"]
    conn.close()


def test_cli_vault_count_only_does_not_ingest(
    clean_state, qdrant_addr, corpus_dir, capsys, monkeypatch
):
    """`vault --count-only` must only count (item 3 of the fix wave): no
    engine work, no ledger writes, no Qdrant collection created."""
    _seed(clean_state, corpus_dir)
    rc = _cli(monkeypatch, clean_state, qdrant_addr, corpus_dir, ["--count-only"])
    assert rc == 0

    out = capsys.readouterr().out
    assert "Matching documents: 3" in out  # doc-text, doc-md, doc-gone
    assert "by source_code" in out
    assert "by doc_type" in out

    conn = psycopg2.connect(clean_state)
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM rag_ingestions")
        (count,) = cur.fetchone()
    conn.close()
    assert count == 0

    client = QdrantClient(host=qdrant_addr[0], port=qdrant_addr[1])
    assert _coll() not in {c.name for c in client.get_collections().collections}
