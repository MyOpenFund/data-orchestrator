"""End to end, sync-metadata: real PostgreSQL and a real Qdrant (the version
production runs) with an API key and a read-only key.

The collection is built the way production was: documents ingested by the
``vault`` source, then turned back into payload schema v1 points (no
interval, no precision, no kind) with the seven indexes of v1. The vault rows
are then corrected as the policy-dates backfill corrected them. The run must
change the owned keys and nothing else, never touch a vector, never write
``rag_ingestions``, and find nothing to do the second time.
"""
from __future__ import annotations

import csv
import hashlib
import json
import struct

import psycopg2
import pytest
from qdrant_client import QdrantClient, models
from rag_engine.schema import PAYLOAD_INDEXES, point_id

from data_orchestrator import cli, sync_metadata as sm
from data_orchestrator.routing import collection_name
from data_orchestrator.vault import COLLECTION_LOCK_CLASS

from .conftest import (
    IT_QDRANT_KEY, IT_QDRANT_READ_ONLY_KEY, insert_documents, write_note_md, write_text_pdf,
)

pytestmark = pytest.mark.integration

V1_INDEXES = [entry for entry in PAYLOAD_INDEXES if entry[0] not in ("date_start", "date_end")]
OWNED = set(sm.OWNED_KEYS)


@pytest.fixture()
def corpus_dir(tmp_path):
    d = tmp_path / "corpus"
    for name in ("a", "b", "c", "e", "g"):
        write_text_pdf(d / "raw" / "gb" / "A3" / f"{name}.pdf")
    write_note_md(d / "raw" / "gb" / "A3" / "f.md")
    # Several sections: several points, so that one of them can disagree.
    sections = "".join(
        f"## Section {i}\n\nThe committee discussed inflation, wages and the outlook for "
        f"policy rates in part {i}. Members noted that price pressures had eased and that "
        f"the balance of risks to growth had shifted over the quarter.\n\n"
        for i in range(1, 6))
    (d / "raw" / "gb" / "A3" / "d.md").write_text("# Minutes\n\n" + sections, encoding="utf-8")
    return d


def _client(addr, key=IT_QDRANT_KEY):
    return QdrantClient(host=addr[0], port=addr[1], https=False, api_key=key, timeout=30)


def _env(monkeypatch, pg_url, addr, corpus_dir, tmp_path, key):
    monkeypatch.setenv("DATABASE_URL", pg_url)
    monkeypatch.setenv("CB_CORPUS_ROOT", str(corpus_dir))
    monkeypatch.setenv("QDRANT_HOST", addr[0])
    monkeypatch.setenv("QDRANT_PORT", str(addr[1]))
    monkeypatch.setenv("QDRANT_API_KEY", key)
    monkeypatch.setenv("DATA_ORCHESTRATOR_STATE_DIR", str(tmp_path / "state"))


def _sql(pg_url, sql, params=None, fetch=False):
    conn = psycopg2.connect(pg_url)
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall() if fetch else None
        conn.commit()
        return rows
    finally:
        conn.close()


def _snapshot(client, coll):
    points, offset = [], None
    while True:
        page, offset = client.scroll(coll, limit=256, offset=offset, with_payload=True,
                                     with_vectors=True)
        points.extend(page)
        if offset is None:
            break
    return {p.id: (dict(p.payload), list(p.vector)) for p in points}


def _vector_digest(snapshot):
    """SHA-256 of every vector as float32 bytes, in point-id order."""
    h = hashlib.sha256()
    for pid in sorted(snapshot):
        h.update(struct.pack(f"<{len(snapshot[pid][1])}f", *snapshot[pid][1]))
    return h.hexdigest()


def _rows(path):
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _build(pg_url, addr, corpus_dir, tmp_path, monkeypatch):
    """Production in miniature: v1 points, seven indexes, then corrected vault rows."""
    coll = collection_name("central-bank")
    client = _client(addr)
    for c in client.get_collections().collections:
        client.delete_collection(c.name)
    gb = "data/raw/gb/A3/"
    insert_documents(pg_url, [
        {"doc_id": "a-moved-year", "corpus": "central-bank", "source_code": "gb",
         "doc_type": "A3", "title": "Minutes", "date": "2016-01-01", "year": 2016,
         "language": "en", "local_path": gb + "a.pdf"},
        {"doc_id": "b-month", "corpus": "central-bank", "source_code": "gb", "doc_type": "E1",
         "title": "Report", "date": "2016-07-01", "year": 2016, "language": "en",
         "local_path": gb + "b.pdf"},
        {"doc_id": "c-undated", "corpus": "central-bank", "source_code": "jp", "doc_type": "A3",
         "title": "Minutes", "date": "2026-01-01", "year": 2026, "language": "en",
         "local_path": gb + "c.pdf"},
        {"doc_id": "d-disagree", "corpus": "central-bank", "source_code": "gb",
         "doc_type": "A3", "title": "Minutes", "date": "2016-01-01", "year": 2016,
         "language": "en", "local_path": gb + "d.md"},
        {"doc_id": "e-deleted", "corpus": "central-bank", "source_code": "gb",
         "doc_type": "A3", "title": "Gone", "date": "2000-01-01", "year": 2000,
         "language": "en", "local_path": gb + "e.pdf"},
        {"doc_id": "f-unchanged-date", "corpus": "central-bank", "source_code": "us",
         "doc_type": "A2", "title": "Statement", "date": "2010-03-04", "year": 2010,
         "language": "en", "local_path": gb + "f.md",
         "extra": json.dumps({"date_precision": "day", "date_kind": "meeting"})},
        {"doc_id": "g-garbage-precision", "corpus": "central-bank", "source_code": "gb",
         "doc_type": "A3", "title": "Minutes", "date": "2012-05-09", "year": 2012,
         "language": "en", "local_path": gb + "g.pdf"},
    ])
    _env(monkeypatch, pg_url, addr, corpus_dir, tmp_path, IT_QDRANT_KEY)
    assert cli.main(["vault", "--corpus", "central-bank"]) == 0
    # Back to payload v1, as the 33,601 production points are; the v1 indexes.
    client.delete_payload(coll, keys=["date_start", "date_end", "date_precision", "date_kind"],
                          points=models.Filter(must=[]), wait=True)
    for field, kind in V1_INDEXES:
        client.create_payload_index(coll, field, field_schema=kind, wait=True)
    # A stale point of d-disagree left by an older ingestion that cut the
    # document into more chunks, with the date that ingestion had.
    (d_point,) = client.scroll(coll, scroll_filter=models.Filter(must=[models.FieldCondition(
        key="doc_id", match=models.MatchValue(value="d-disagree"))]), limit=100,
        with_vectors=True)[0]
    stale = dict(d_point.payload, chunk_number=7, date="2014-02-03", year=2014)
    client.upsert(coll, points=[models.PointStruct(
        id=point_id("d-disagree", 0, 7), vector=d_point.vector, payload=stale)], wait=True)
    # The corrections of the vault (what the policy-dates backfill did).
    _sql(pg_url, "UPDATE documents SET date='2015-12-17', year=2015, extra=%s WHERE doc_id=%s",
         (json.dumps({"date_precision": "day", "date_kind": "meeting", "date_source": "title"}),
          "a-moved-year"))
    _sql(pg_url, "UPDATE documents SET extra=%s WHERE doc_id=%s",
         (json.dumps({"date_precision": "month", "date_kind": "period",
                      "date_start": "1999-01-01T00:00:00Z"}), "b-month"))
    _sql(pg_url, "UPDATE documents SET date=NULL, extra=%s WHERE doc_id=%s",
         (json.dumps({"date_kind": "meeting"}), "c-undated"))
    _sql(pg_url, "UPDATE documents SET extra=%s WHERE doc_id=%s",
         (json.dumps({"date_precision": "fortnight"}), "g-garbage-precision"))
    _sql(pg_url, "UPDATE documents SET deleted_at=now() WHERE doc_id='e-deleted'")
    return coll, client


def _sync(monkeypatch, pg_url, addr, corpus_dir, tmp_path, key, *args):
    _env(monkeypatch, pg_url, addr, corpus_dir, tmp_path, key)
    return cli.main(["sync-metadata", "--corpus", "central-bank", *args])


def test_sync_metadata_changes_the_owned_keys_only(
        clean_pg, qdrant_keyed_addr, corpus_dir, tmp_path, monkeypatch, capsys):
    pg, addr = clean_pg, qdrant_keyed_addr
    coll, client = _build(pg, addr, corpus_dir, tmp_path, monkeypatch)
    capsys.readouterr()
    ledger_before = _sql(pg, "SELECT doc_id, collection, chunk_count, ingested_at "
                             "FROM rag_ingestions ORDER BY doc_id", fetch=True)
    before = _snapshot(client, coll)
    digest_before = _vector_digest(before)

    # 1. Dry run with the read-only key: counts, CSV, nothing written.
    dry_csv = tmp_path / "dry.csv"
    assert _sync(monkeypatch, pg, addr, corpus_dir, tmp_path, IT_QDRANT_READ_ONLY_KEY,
                 "--csv", str(dry_csv), "--page-size", "3", "--doc-batch", "2") == 0
    out = capsys.readouterr().out
    assert _snapshot(client, coll) == before
    assert "  documents          : 7" in out
    assert "  updated            : 5" in out and "  undated            : 1" in out
    assert "  missing-in-vault   : 1" in out and "  documents to write : 6" in out
    assert "  points disagreeing : 1 document(s)" in out
    assert "payload indexes missing: date_start, date_end" in out
    rows = _rows(dry_csv)
    by = {(r["doc_id"], r["field"]): r for r in rows}
    assert by[("a-moved-year", "date")]["qdrant_value"] == "2016-01-01"
    assert by[("a-moved-year", "date")]["vault_value"] == "2015-12-17"
    assert by[("a-moved-year", "year")]["vault_value"] == "2015"
    assert by[("b-month", "date_end")]["vault_value"] == "2016-07-31T00:00:00Z"
    assert ("b-month", "date") not in by
    assert by[("c-undated", "date")]["vault_value"] == "" and by[("c-undated", "date")]["action"] == "undated"
    assert json.loads(by[("d-disagree", "date")]["qdrant_value"]) == ["2014-02-03", "2016-01-01"]
    assert by[("e-deleted", "")]["action"] == "missing-in-vault"
    assert by[("g-garbage-precision", "date_end")]["vault_value"] == "2012-05-09T00:00:00Z"

    # 2. --write with the read-only key: refused, clearly, nothing written.
    assert _sync(monkeypatch, pg, addr, corpus_dir, tmp_path, IT_QDRANT_READ_ONLY_KEY,
                 "--write") == 1
    err = capsys.readouterr().err
    assert "QdrantAuthError" in err and "a read-only key is refused" in err
    assert _snapshot(client, coll) == before

    # 3. --write with the write key.
    write_csv = tmp_path / "write.csv"
    assert _sync(monkeypatch, pg, addr, corpus_dir, tmp_path, IT_QDRANT_KEY,
                 "--write", "--csv", str(write_csv)) == 0
    out = capsys.readouterr().out
    assert "payload indexes created: date_start, date_end" in out
    assert "  documents written  : 6" in out
    after = _snapshot(client, coll)
    assert set(after) == set(before)
    assert _vector_digest(after) == digest_before
    changed_keys = set()
    for pid, (payload, vector) in after.items():
        old_payload, old_vector = before[pid]
        assert vector == old_vector
        changed_keys |= {k for k in set(payload) | set(old_payload)
                         if payload.get(k) != old_payload.get(k)}
    assert changed_keys <= OWNED
    schema = client.get_collection(coll).payload_schema
    assert {f for f, _ in PAYLOAD_INDEXES} <= set(schema)
    assert _sql(pg, "SELECT doc_id, collection, chunk_count, ingested_at FROM rag_ingestions "
                    "ORDER BY doc_id", fetch=True) == ledger_before

    def payloads(doc_id):
        return [p for p, _ in after.values() if p["doc_id"] == doc_id]

    for p in payloads("a-moved-year"):
        assert (p["date"], p["year"], p["date_precision"], p["date_kind"]) == (
            "2015-12-17", 2015, "day", "meeting")
        assert p["date_start"] == p["date_end"] == "2015-12-17T00:00:00Z"
        assert p["date_source"] == "title"  # owned: as a fresh ingestion writes it
    for p in payloads("b-month"):
        assert (p["date_start"], p["date_end"]) == ("2016-07-01T00:00:00Z", "2016-07-31T00:00:00Z")
    for p in payloads("c-undated"):
        assert not {"date", "date_start", "date_end", "date_precision"} & set(p)
        assert p["date_kind"] == "meeting" and p["year"] == 2026
    assert {(p["date"], p["year"]) for p in payloads("d-disagree")} == {("2016-01-01", 2016)}
    for p in payloads("e-deleted"):
        assert "date_start" not in p and p["date"] == "2000-01-01"  # left as it was
    for p in payloads("g-garbage-precision"):
        assert p["date_precision"] == "fortnight" and p["date_end"] == "2012-05-09T00:00:00Z"

    # 4. Second run: nothing to write, nothing moves.
    assert _sync(monkeypatch, pg, addr, corpus_dir, tmp_path, IT_QDRANT_KEY, "--write",
                 "--csv", str(tmp_path / "again.csv")) == 0
    out = capsys.readouterr().out
    assert "  documents written  : 0" in out and "  points disagreeing : 0 document(s)" in out
    assert _snapshot(client, coll) == after
    assert [r["doc_id"] for r in _rows(tmp_path / "again.csv")] == ["e-deleted"]

    # 5. What a search now answers (interval mode; include_undated).
    from rag_engine.embedding import FakeEmbedder, ModelSpec
    from rag_engine.search import SearchFilters, search
    from rag_engine.store import QdrantStore

    store = QdrantStore(addr[0], addr[1], api_key=IT_QDRANT_READ_ONLY_KEY, https=False)
    embedder = FakeEmbedder(ModelSpec(name="fake", dim=768, passage_prefix="passage: ",
                                      query_prefix="query: "))

    def found(**filters):
        hits = search("minutes", collection=coll, store=store, embedder=embedder,
                      filters=SearchFilters(**filters), limit=50, max_limit=50,
                      date_filter_mode="auto", date_filter_refresh=0)
        return {h.doc_id for h in hits}

    assert "b-month" in found(date_from="2016-07-14", date_to="2016-07-14")
    assert "a-moved-year" in found(date_from="2015-12-17", date_to="2015-12-17")
    assert "a-moved-year" not in found(date_from="2016-01-01", date_to="2016-01-01")
    assert "c-undated" not in found(date_from="2000-01-01", date_to="2030-12-31")
    assert "c-undated" in found(date_from="2000-01-01", date_to="2030-12-31", include_undated=True)

    print(f"\nsync-metadata e2e: {len(after)} points, vectors sha256 {digest_before} "
          f"before and after; keys changed: {sorted(changed_keys)}")


def test_a_trial_on_a_few_documents_with_limit_and_doc_ids(
        clean_pg, qdrant_keyed_addr, corpus_dir, tmp_path, monkeypatch, capsys):
    pg, addr = clean_pg, qdrant_keyed_addr
    coll, client = _build(pg, addr, corpus_dir, tmp_path, monkeypatch)
    before = _snapshot(client, coll)
    assert _sync(monkeypatch, pg, addr, corpus_dir, tmp_path, IT_QDRANT_KEY, "--write",
                 "--doc-ids", "b-month,not-ingested", "--csv", str(tmp_path / "t.csv")) == 0
    captured = capsys.readouterr()
    assert "  documents          : 1" in captured.out
    assert "not in rag_ingestions" in captured.err and "not-ingested" in captured.err
    after = _snapshot(client, coll)
    moved = {after[pid][0]["doc_id"] for pid in after if after[pid][0] != before[pid][0]}
    assert moved == {"b-month"}
    assert _sync(monkeypatch, pg, addr, corpus_dir, tmp_path, IT_QDRANT_KEY, "--limit", "2",
                 "--csv", str(tmp_path / "l.csv")) == 0
    assert "  documents          : 2" in capsys.readouterr().out


def test_a_strict_collection_without_the_doc_id_index(
        clean_pg, qdrant_keyed_addr, corpus_dir, tmp_path, monkeypatch, capsys):
    pg, addr = clean_pg, qdrant_keyed_addr
    client = _client(addr)
    coll = "zz-strict"
    if client.collection_exists(coll):
        client.delete_collection(coll)
    client.create_collection(
        coll, vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
        strict_mode_config=models.StrictModeConfig(
            enabled=True, unindexed_filtering_retrieve=False, unindexed_filtering_update=False))
    client.upsert(coll, points=[models.PointStruct(id=1, vector=[0.1] * 768, payload={
        "doc_id": "s1", "date": "2016-01-01", "year": 2016})], wait=True)
    insert_documents(pg, [{"doc_id": "s1", "corpus": "central-bank", "date": "2016-02-01",
                           "year": 2016, "extra": json.dumps({"date_precision": "month"})}])
    _sql(pg, "INSERT INTO rag_ingestions (doc_id, collection, corpus, embedding_model, "
             "chunk_count) VALUES ('s1', %s, 'central-bank', 'm', 1)", (coll,))

    assert _sync(monkeypatch, pg, addr, corpus_dir, tmp_path, IT_QDRANT_READ_ONLY_KEY,
                 "--collection", coll) == 1
    err = capsys.readouterr().err
    assert "MissingIndexError" in err and "doc_id" in err
    assert f"sync-metadata --create-indexes --collection {coll}" in err

    assert _sync(monkeypatch, pg, addr, corpus_dir, tmp_path, IT_QDRANT_READ_ONLY_KEY,
                 "--collection", coll, "--create-indexes") == 1
    assert "QdrantAuthError" in capsys.readouterr().err
    assert _sync(monkeypatch, pg, addr, corpus_dir, tmp_path, IT_QDRANT_KEY,
                 "--collection", coll, "--create-indexes") == 0
    assert "date_start, date_end" in capsys.readouterr().out
    assert _sync(monkeypatch, pg, addr, corpus_dir, tmp_path, IT_QDRANT_KEY,
                 "--collection", coll, "--write") == 0
    (point,) = client.retrieve(coll, [1])
    assert point.payload["date"] == "2016-02-01"
    assert point.payload["date_end"] == "2016-02-29T00:00:00Z"
    client.delete_collection(coll)


def test_sync_metadata_and_an_ingestion_exclude_each_other(
        clean_pg, qdrant_keyed_addr, corpus_dir, tmp_path, monkeypatch, capsys):
    pg, addr = clean_pg, qdrant_keyed_addr
    coll = collection_name("central-bank")
    client = _client(addr)
    if not client.collection_exists(coll):
        client.create_collection(coll, vectors_config=models.VectorParams(
            size=768, distance=models.Distance.COSINE))
    holder = psycopg2.connect(pg)
    try:
        with holder.cursor() as cur:  # an ingestion running: the shared lock
            cur.execute("SELECT pg_advisory_lock_shared(%s, hashtext(%s))",
                        (COLLECTION_LOCK_CLASS, coll))
        assert _sync(monkeypatch, pg, addr, corpus_dir, tmp_path, IT_QDRANT_KEY, "--write") == 1
        assert "an ingestion (or another sync-metadata run) holds collection" in \
            capsys.readouterr().err
        # A dry run only reads: it runs beside an ingestion.
        assert _sync(monkeypatch, pg, addr, corpus_dir, tmp_path, IT_QDRANT_KEY) == 0
        capsys.readouterr()
        with holder.cursor() as cur:
            cur.execute("SELECT pg_advisory_unlock_shared(%s, hashtext(%s))",
                        (COLLECTION_LOCK_CLASS, coll))
            cur.execute("SELECT pg_advisory_lock(%s, hashtext(%s))",  # a sync-metadata running
                        (COLLECTION_LOCK_CLASS, coll))
        _env(monkeypatch, pg, addr, corpus_dir, tmp_path, IT_QDRANT_KEY)
        assert cli.main(["vault", "--corpus", "central-bank"]) == 1
        assert "a sync-metadata --write run holds collection" in capsys.readouterr().err
        assert _sync(monkeypatch, pg, addr, corpus_dir, tmp_path, IT_QDRANT_KEY) == 1
    finally:
        holder.close()
    # Released with the session: both run again.
    assert _sync(monkeypatch, pg, addr, corpus_dir, tmp_path, IT_QDRANT_KEY) == 0


def test_a_refusal_names_the_session_that_holds_the_lock(
        clean_pg, qdrant_keyed_addr, corpus_dir, tmp_path, monkeypatch, capsys):
    """A stale lock (a dead client whose session the server still keeps) must
    be traceable: the refusal names the holder's pid and, when the role may
    read them, its client and start, with the statement that ends it."""
    pg, addr = clean_pg, qdrant_keyed_addr
    coll = collection_name("central-bank")
    holder = psycopg2.connect(pg, application_name="data-orchestrator@sleeping-mac")
    try:
        with holder.cursor() as cur:
            cur.execute("SELECT pg_backend_pid()")
            (pid,) = cur.fetchone()
            cur.execute("SELECT pg_advisory_lock_shared(%s, hashtext(%s))",
                        (COLLECTION_LOCK_CLASS, coll))
        assert _sync(monkeypatch, pg, addr, corpus_dir, tmp_path, IT_QDRANT_KEY, "--write") == 1
        err = capsys.readouterr().err
        assert f"held by pid {pid} (ShareLock" in err
        assert "application data-orchestrator@sleeping-mac" in err and "session started" in err
        assert "/32" not in err  # the address, not the inet with its mask
        assert f"SELECT pg_terminate_backend({pid});" in err

        # A role that may not read another role's session details.
        _sql(pg, "DROP ROLE IF EXISTS it_limited")
        _sql(pg, "CREATE ROLE it_limited LOGIN PASSWORD 'it'")
        limited = psycopg2.extensions.make_dsn(pg, user="it_limited", password="it")
        try:
            assert _sync(monkeypatch, limited, addr, corpus_dir, tmp_path, IT_QDRANT_KEY,
                         "--write") == 1
            err = capsys.readouterr().err
            assert f"held by pid {pid} (ShareLock, user " in err
            assert "its client and start are not visible to this role" in err
            assert f"SELECT pg_terminate_backend({pid});" in err
        finally:
            _sql(pg, "DROP ROLE IF EXISTS it_limited")
    finally:
        holder.close()
