"""End-to-end, vault source: vault-selected documents through the full chain via the CLI."""
import socket

import psycopg2
import pytest
from qdrant_client import QdrantClient

from data_orchestrator import cli
from data_orchestrator.routing import collection_name

from .conftest import insert_documents, write_note_md, write_text_pdf

pytestmark = pytest.mark.integration


@pytest.fixture()
def corpus_dir(tmp_path):
    """Vault-realistic corpus root, overriding conftest's flat disk-source fixture.

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
    """Derive collection name at test runtime from DATA_ORCHESTRATOR_EMBEDDING_MODEL."""
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
    # doc-gone is 1 failure of 3 attempted (33 %), above the 10 % default
    # failure ratio (policy.py): the run ingested something but is degraded.
    assert rc == 3
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
        cur.execute("SELECT tool, outcome, corpus, totals->>'docs_unsupported' FROM runs")
        runs = cur.fetchall()
    # Every seeded row is a supported type (NULL mime + .pdf/.md path).
    assert runs == [("data-orchestrator", "degraded", "central-bank", "0")]


def test_cli_vault_second_run_selects_nothing(
    clean_state, qdrant_addr, corpus_dir, monkeypatch
):
    _seed(clean_state, corpus_dir)
    _cli(monkeypatch, clean_state, qdrant_addr, corpus_dir)
    before = QdrantClient(host=qdrant_addr[0], port=qdrant_addr[1]).count(_coll()).count
    rc = _cli(monkeypatch, clean_state, qdrant_addr, corpus_dir)
    # Honest exit code: the anti-join leaves only doc-gone as a
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
    """`vault --count-only` must only count: no engine work, no ledger
    writes, no Qdrant collection created."""
    _seed(clean_state, corpus_dir)
    rc = _cli(monkeypatch, clean_state, qdrant_addr, corpus_dir, ["--count-only"])
    assert rc == 0

    out = capsys.readouterr().out
    assert "Matching documents: 3" in out  # doc-text, doc-md, doc-gone
    assert "by source_code" in out
    assert "by doc_type" in out
    assert "Unsupported (not offered to the engine): 0" in out

    conn = psycopg2.connect(clean_state)
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM rag_ingestions")
        (count,) = cur.fetchone()
    conn.close()
    assert count == 0

    client = QdrantClient(host=qdrant_addr[0], port=qdrant_addr[1])
    assert _coll() not in {c.name for c in client.get_collections().collections}


PROSE = ("Central banks track consumer prices closely and adjust policy rates "
         "when inflation drifts away from target. This note discusses how "
         "inflation expectations feed back into wage negotiations.\n")


def _seed_mixed_types(pg_url, corpus_dir):
    """One row per edge of the supported-type predicate: typed PDF, an
    unsupported mime, a NULL mime on an upper-case extension, a mime with
    parameters, a blank mime, and no local_path."""
    base = corpus_dir / "raw" / "us" / "C1" / "2015"
    write_text_pdf(base / "typed.pdf")
    write_text_pdf(base / "UPPER.PDF")
    (base / "page.html").write_text(f"<html><body><p>{PROSE}</p></body></html>")
    (base / "plain.txt").write_text(PROSE * 3)
    write_note_md(base / "blank.md")
    common = {"corpus": "central-bank", "source_code": "us", "doc_type": "C1", "year": 2015}
    rel = "data/raw/us/C1/2015/"
    insert_documents(pg_url, [
        {"doc_id": "doc-pdf", **common, "mime_type": "application/pdf",
         "local_path": rel + "typed.pdf"},
        {"doc_id": "doc-html", **common, "mime_type": "text/html",
         "local_path": rel + "page.html"},
        {"doc_id": "doc-upper", **common, "mime_type": None,
         "local_path": rel + "UPPER.PDF"},
        {"doc_id": "doc-charset", **common, "mime_type": "text/plain; charset=utf-8",
         "local_path": rel + "plain.txt"},
        {"doc_id": "doc-blank", **common, "mime_type": "", "local_path": rel + "blank.md"},
        {"doc_id": "doc-nopath", **common, "mime_type": None, "local_path": None},
    ])


def test_cli_vault_offers_only_supported_types_and_counts_the_rest(
    clean_state, qdrant_addr, corpus_dir, monkeypatch
):
    _seed_mixed_types(clean_state, corpus_dir)
    rc = _cli(monkeypatch, clean_state, qdrant_addr, corpus_dir)
    assert rc == 0  # unsupported documents never degrade a run
    with psycopg2.connect(clean_state) as conn, conn.cursor() as cur:
        cur.execute("SELECT doc_id FROM rag_ingestions ORDER BY doc_id")
        done = [r[0] for r in cur.fetchall()]
        cur.execute("SELECT outcome, corpus, totals->>'docs_unsupported', "
                    "totals->>'docs_failed', sources, extra->'unsupported_by_source', "
                    "extra ? 'fatal', extra ? 'error' FROM runs")
        (outcome, corpus, unsupported, failed, sources, split,
         has_fatal, has_error), = cur.fetchall()
    # html: unsupported mime; nopath: no file extension to dispatch on.
    assert done == ["doc-blank", "doc-charset", "doc-pdf", "doc-upper"]
    assert (outcome, corpus, unsupported, failed) == ("ok", "central-bank", "2", "0")
    # "us" was touched by the run, so its entry carries its unsupported count.
    assert [(s["source_code"], s["docs_unsupported"]) for s in sources] == [("us", 2)]
    # The full split is a report field without a runs column: swept into extra.
    assert split == {"us": 2}
    assert (has_fatal, has_error) == (False, False)


def _closed_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_a_fatal_vault_run_reports_fatal_not_error(clean_pg, corpus_dir, monkeypatch):
    """Forced fatal: nothing listens on the Qdrant port, so creating the
    collection fails before any document. Postgres only (no Qdrant needed),
    but the embedding model is still loaded first (its dimension is checked
    against the collection): it needs the tiny embedding model
    (``tiny_model_env``, DATA_ORCHESTRATOR_EMBEDDING_MODEL) downloadable or
    cached."""
    _seed(clean_pg, corpus_dir)
    rc = _cli(monkeypatch, clean_pg, ("127.0.0.1", _closed_port()), corpus_dir)
    assert rc == 1
    with psycopg2.connect(clean_pg) as conn, conn.cursor() as cur:
        cur.execute("SELECT outcome, exit_code, corpus, extra->>'fatal', extra ? 'error' "
                    "FROM runs")
        (outcome, exit_code, corpus, fatal, has_error), = cur.fetchall()
        cur.execute("SELECT count(*) FROM rag_ingestions")
        (ingested,) = cur.fetchone()
    assert (outcome, exit_code, corpus) == ("failed", 1, "central-bank")
    assert fatal and len(fatal) <= 500
    assert has_error is False
    assert ingested == 0


def test_cli_vault_count_only_splits_the_mixed_types(clean_pg, corpus_dir, capsys, monkeypatch):
    """Same seed as above, Postgres only: the selection and its unsupported
    complement, as the operator sees them before a run."""
    _seed_mixed_types(clean_pg, corpus_dir)
    rc = _cli(monkeypatch, clean_pg, ("127.0.0.1", _closed_port()), corpus_dir, ["--count-only"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Matching documents: 4" in out
    assert "Unsupported (not offered to the engine): 2" in out
    assert "by source_code: {'us': 2}" in out
