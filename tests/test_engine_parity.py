"""Parity with the pipeline this repository ran before it called rag-engine.

``tests/fixtures/engine_parity/expected.json`` holds the points the previous
``ingest_item`` (own point ids and payloads, the engine fork's chunker) wrote
for six documents; its README says how it was captured. Today's ``ingest_item``
must write the same points for the same vault rows, with the same test
embedder and OCR off (MIGRATION.md, "What was checked", says what that covers).

Never adapt the fixture to make this pass.
"""
import datetime
import hashlib
import json
from pathlib import Path

import pytest
from qdrant_client import QdrantClient
from rag_engine.config import Settings
from rag_engine.embedding import FakeEmbedder, ModelSpec
from rag_engine.store import QdrantStore

from data_orchestrator.core import ingest_item
from data_orchestrator.sources.vault import row_to_item

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "engine_parity"
EXPECTED = json.loads((FIXTURE / "expected.json").read_text(encoding="utf-8"))
DOCUMENTS = EXPECTED["documents"]


class RecordingClient:
    """A local-mode Qdrant that also keeps the points in the order they were written."""

    def __init__(self):
        self._inner = QdrantClient(":memory:")
        self.written = []

    def upsert(self, collection_name, points, **kwargs):
        self.written.extend(points)
        return self._inner.upsert(collection_name=collection_name, points=points, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _vault_row(row):
    """The row as the vault hands it over: ``date`` a date, ``sha256`` of the file."""
    out = dict(row)
    if out.get("date"):
        out["date"] = datetime.date.fromisoformat(out["date"])
    path = FIXTURE / "corpus" / row["local_path"].strip()[len("data/"):]
    out["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def _ingest(document, monkeypatch):
    monkeypatch.setenv("CB_CORPUS_ROOT", str(FIXTURE / "corpus"))
    spec = dict(EXPECTED["embedder"])
    seed = spec.pop("seed")
    embedder = FakeEmbedder(ModelSpec(**spec), seed=seed)
    client = RecordingClient()
    store = QdrantStore(client=client, settings=Settings())
    store.ensure_collection(EXPECTED["collection"], embedder.dim)
    item = row_to_item(_vault_row(document["row"]), EXPECTED["corpus"])
    written = ingest_item(item, EXPECTED["collection"], store=store, embedder=embedder,
                          ocr=EXPECTED["ocr"])
    return item, written, client


def test_the_fixture_is_the_one_captured_before_the_switch():
    assert EXPECTED["captured_from"]["data_orchestrator"].startswith("7196bfc")
    assert EXPECTED["captured_from"]["engine_fork"].startswith("6e9093a")
    assert len(DOCUMENTS) == 6
    assert sum(len(d["points"]) for d in DOCUMENTS) == 23
    assert {Path(d["row"]["local_path"].strip()).suffix for d in DOCUMENTS} == {
        ".md", ".txt", ".csv", ".pdf", ".docx"}


@pytest.mark.parametrize("document", DOCUMENTS, ids=lambda d: d["row"]["doc_id"])
def test_ingest_item_writes_the_points_of_the_previous_pipeline(document, monkeypatch):
    item, written, client = _ingest(document, monkeypatch)
    expected = document["points"]

    assert written == document["written"] == len(expected)
    assert [p.id for p in client.written] == [p["id"] for p in expected]
    for got, want in zip(client.written, expected):
        payload, want_payload = dict(got.payload), dict(want["payload"])
        # The one value that is the time of the run.
        date, want_date = payload.pop("ingestion_date"), want_payload.pop("ingestion_date")
        datetime.datetime.fromisoformat(date)
        datetime.datetime.fromisoformat(want_date)
        assert payload == want_payload
        assert set(got.payload) == set(want["payload"])
        assert got.vector == want["vector"]
    assert len({p.payload["ingestion_date"] for p in client.written}) == 1

    # What the server stores is what was sent (ids, payloads, vectors).
    stored, _ = client.scroll(EXPECTED["collection"], limit=100, with_payload=True,
                              with_vectors=True)
    by_id = {p.id: p for p in stored}
    assert set(by_id) == {p["id"] for p in expected}
    for want in expected:
        got = by_id[want["id"]]
        assert {k: v for k, v in got.payload.items() if k != "ingestion_date"} == {
            k: v for k, v in want["payload"].items() if k != "ingestion_date"}
        assert got.vector == pytest.approx(want["vector"], abs=1e-6)


@pytest.mark.parametrize("document", DOCUMENTS, ids=lambda d: d["row"]["doc_id"])
def test_payload_keys_keep_their_order_unless_a_reserved_key_was_dropped(document, monkeypatch):
    """Key order means nothing to Qdrant, but it shows the payload is assembled
    the same way. The one document whose ``extra`` held keys the engine writes
    itself is the exception: the previous pipeline overwrote them in place, the
    source now leaves them out (same keys, same values, another position)."""
    item, _written, client = _ingest(document, monkeypatch)
    same_order = all(list(got.payload) == list(want["payload"])
                     for got, want in zip(client.written, document["points"]))
    assert same_order == (not item.dropped_keys)


def test_the_stale_reserved_keys_of_an_extra_are_dropped_and_reported(monkeypatch):
    (document,) = [d for d in DOCUMENTS if d["row"]["doc_id"] == "us-a3-2024-06-12-minutes"]
    item, _written, client = _ingest(document, monkeypatch)
    assert sorted(item.dropped_keys) == ["chunk_number", "filename", "ingestion_date", "text"]
    first = client.written[0].payload
    assert first["filename"] == "us-a3-2024-06-12-minutes.txt" and first["chunk_number"] == 0
    assert first["meeting"] == "June 2024"  # the rest of the extra is kept
