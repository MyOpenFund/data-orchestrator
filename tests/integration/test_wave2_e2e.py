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
    assert client.count(_coll()).count > 0


def test_cli_vault_second_run_selects_nothing(clean_state, qdrant_addr, corpus_dir):
    _seed(clean_state, corpus_dir)
    _cli(clean_state, qdrant_addr, corpus_dir)
    before = QdrantClient(host=qdrant_addr[0], port=qdrant_addr[1]).count(_coll()).count
    rc = _cli(clean_state, qdrant_addr, corpus_dir)
    assert rc == 0
    after = QdrantClient(host=qdrant_addr[0], port=qdrant_addr[1]).count(_coll()).count
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
