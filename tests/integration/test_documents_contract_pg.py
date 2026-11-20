"""The vault's identity (corpus, doc_id), against a real Postgres at the
schema of vault 12ffd11 (documents contract 1.0.0): two corpora may both
have a doc_id, and every read and write of this repo tells them apart."""
import psycopg2
import pytest

from data_orchestrator import sync_metadata as sm
from data_orchestrator.sources.vault import build_selection_sql
from data_orchestrator.vault import VaultLedger, VaultSchemaTooOld

from .conftest import insert_documents

pytestmark = pytest.mark.integration

SHARED = "shared-collection"  # a collection both corpora write to, for the test only


def _ledger(conn, corpus, collection=SHARED):
    return VaultLedger(conn, collection=collection, corpus=corpus, embedding_model="m",
                       embedding_version="v")


@pytest.fixture()
def two_corpora(clean_pg):
    insert_documents(clean_pg, [
        {"doc_id": "same", "corpus": "central-bank", "source_code": "ecb",
         "mime_type": "application/pdf", "local_path": "raw/ecb/same.pdf",
         "date": "2016-07-01", "year": 2016, "date_precision": "month"},
        {"doc_id": "same", "corpus": "company", "source_code": "sec",
         "mime_type": "application/pdf", "local_path": "raw/0000320193/A1/2024/same.pdf",
         "date": "2024-11-01", "year": 2024, "date_precision": "day",
         "entity_scheme": "cik", "entity_value": "0000320193"},
    ])
    conn = psycopg2.connect(clean_pg)
    try:
        yield conn
    finally:
        conn.close()


def test_a_document_of_one_corpus_does_not_hide_the_same_doc_id_of_another(two_corpora):
    _ledger(two_corpora, "central-bank").mark("same", 3, {"source_code": "ecb"})
    for corpus, selected in (("central-bank", []), ("company", ["same"])):
        sql, params = build_selection_sql(corpus, SHARED)
        with two_corpora.cursor() as cur:
            cur.execute(sql, params)
            assert [r[0] for r in cur.fetchall()] == selected, corpus
    two_corpora.rollback()
    assert "same" in _ledger(two_corpora, "central-bank")
    assert "same" not in _ledger(two_corpora, "company")  # the resume set is per corpus


def test_mark_upserts_on_the_vault_s_key(two_corpora):
    led = _ledger(two_corpora, "company", "company-e5b-v1")
    led.mark("same", 3, {"source_code": "sec"})
    led.mark("same", 5, {"source_code": "sec"})  # a re-ingestion: same row, updated
    with two_corpora.cursor() as cur:
        cur.execute("SELECT corpus, doc_id, collection, chunk_count FROM rag_ingestions")
        assert cur.fetchall() == [("company", "same", "company-e5b-v1", 5)]


def test_the_ledger_rows_join_the_document_of_their_own_corpus(two_corpora):
    _ledger(two_corpora, "company", "company-e5b-v1").mark("same", 3, {})
    (row,) = sm.ledger_rows(two_corpora, "company", "company-e5b-v1")
    assert row["source_code"] == "sec" and row["entity_value"] == "0000320193"
    assert row["date_precision"] == "day"


def test_a_vault_before_the_contract_stops_the_run_with_its_reason(clean_pg):
    conn = psycopg2.connect(clean_pg)
    conn.autocommit = True
    with conn.cursor() as cur:  # rag_ingestions as vault 2d0457e had it
        cur.execute("DROP TABLE rag_ingestions; CREATE TABLE rag_ingestions ("
                    "doc_id TEXT NOT NULL, collection TEXT NOT NULL, corpus TEXT NOT NULL, "
                    "source_code TEXT, embedding_model TEXT NOT NULL, embedding_version TEXT, "
                    "chunk_count INTEGER NOT NULL, ingested_at TIMESTAMPTZ NOT NULL DEFAULT now(), "
                    "PRIMARY KEY (doc_id, collection))")
    conn.autocommit = False
    insert_documents(clean_pg, [{"doc_id": "d", "corpus": "central-bank"}])
    led = _ledger(conn, "central-bank")
    with pytest.raises(VaultSchemaTooOld, match="predates the documents contract"):
        led.mark("d", 1, {})
    with conn.cursor() as cur:  # the connection is healthy again
        cur.execute("SELECT 1")
    conn.close()


def test_the_company_corpus_is_ingested_from_the_vault(clean_state, qdrant_addr, tmp_path,
                                                       monkeypatch):
    """``vault --corpus company``: a company row in contract form (golden SEC
    10-K, its rendered file under COMPANY_CORPUS_ROOT), next to a central-bank
    row with the same doc_id; the points carry the contract's columns."""
    from qdrant_client import QdrantClient
    from vault_contract import golden

    from data_orchestrator import cli
    from data_orchestrator.payload import manifest_row_to_vault_row
    from data_orchestrator.routing import collection_name

    from .conftest import write_text_pdf

    import json

    case = golden.accepted()["sec_10k"]
    row = manifest_row_to_vault_row(case["row"], "company")
    row["extra"] = json.dumps(row["extra"])
    doc_id = row["doc_id"]
    insert_documents(clean_state, [row, {"doc_id": doc_id, "corpus": "central-bank",
                                         "local_path": "raw/x.pdf"}])
    root = tmp_path / "company-data"
    write_text_pdf(root / row["local_path"])
    monkeypatch.setenv("DATABASE_URL", clean_state)
    monkeypatch.setenv("COMPANY_CORPUS_ROOT", str(root))
    monkeypatch.setenv("QDRANT_HOST", qdrant_addr[0])
    monkeypatch.setenv("QDRANT_PORT", str(qdrant_addr[1]))
    assert cli.main(["vault", "--corpus", "company"]) == 0
    with psycopg2.connect(clean_state) as conn, conn.cursor() as cur:
        cur.execute("SELECT corpus, doc_id, collection FROM rag_ingestions")
        assert cur.fetchall() == [("company", doc_id, collection_name("company"))]
    points, _ = QdrantClient(host=qdrant_addr[0], port=qdrant_addr[1]).scroll(
        collection_name("company"), limit=100, with_payload=True)
    assert points
    for point in points:
        payload = point.payload
        assert payload["corpus"] == "company" and payload["doc_type"] == "A1"
        assert (payload["entity_scheme"], payload["entity_value"]) == ("cik", "0000320193")
        assert (payload["date_precision"], payload["date_kind"]) == ("day", "publication")
        assert payload["date_start"] == "2024-11-01T00:00:00Z"
        assert payload["accession"] == "0000320193-24-000123"
        assert not {"local_path", "pdf_path", "text_path", "primary_path"} & set(payload)
