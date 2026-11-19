"""Parity with the pipeline this repository ran before it called rag-engine.

``tests/fixtures/engine_parity/expected.json`` holds the points the previous
``ingest_item`` (own point ids and payloads, the engine fork's chunker) wrote
for six documents; its README says how it was captured. Today's ``ingest_item``
must write the same points for the same vault rows, with the same test
embedder and OCR off (MIGRATION.md, "What was checked", says what that covers).

Since rag-engine's payload schema v2 a point also carries ``date_start`` and
``date_end``, derived by the engine from ``date`` and ``date_precision`` and
written after every key of v1. The fixture keeps the v1 points as captured;
:func:`_v2` adds exactly those two keys to them, computed here independently
of the engine (no row of the fixture states a precision: a day), and nothing
else. Every other key, value, position, id and vector must still be the
captured one.

Never adapt the fixture to make this pass. The one translation of the
captured rows, :func:`_vault_row` (``pdf_url`` moved from ``extra`` to its
column), is not an adaptation of the expected points: the captured row with
``pdf_url`` in ``extra`` is synthetic, the vault cannot produce it
(``pdf_url`` is one of its columns, ``ingestion/ingest.py`` ``KNOWN_FIELDS``),
so today's selection of the same document carries the value in the column.
The expected points stay the captured ones.
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


def _v2(payload):
    """The v1 payload as captured, with the interval schema v2 appends: the
    day of ``date`` twice, at midnight UTC, last; nothing for an undated row."""
    out = dict(payload)
    assert "date_precision" not in out  # else the interval below would be wrong
    if out.get("date"):
        day = datetime.date.fromisoformat(out["date"]).isoformat()
        out["date_start"] = out["date_end"] = f"{day}T00:00:00Z"
    return out


def test_v2_adds_exactly_the_two_derived_keys_to_dated_points_only():
    added = {doc["row"]["doc_id"]: [k for k in _v2(doc["points"][0]["payload"])
                                    if k not in doc["points"][0]["payload"]]
             for doc in DOCUMENTS}
    assert added == {doc["row"]["doc_id"]: ([] if doc["row"].get("date") is None
                                            else ["date_start", "date_end"])
                     for doc in DOCUMENTS}
    assert sum(1 for keys in added.values() if keys) == 5  # one fixture row is undated


#: The columns the selection reads since the fixture was captured: the captured
#: rows do not carry them.
LINK_COLUMNS = ("pdf_url", "source_url", "mime_type")


def _vault_row(row):
    """The row as the vault hands it over: ``date`` a date, ``sha256`` of the file.

    Since the fixture was captured the selection also reads ``pdf_url``,
    ``source_url`` and ``mime_type`` (columns win over ``extra``, a NULL one
    removes the same key of ``extra``). The captured rows have none of these
    columns, and one of them holds ``pdf_url`` in ``extra``, where the vault
    never puts it (``pdf_url`` is one of its columns): the row of today's
    selection carries that value in the column. The expected points are the
    captured ones, unchanged."""
    out = dict(row)
    for col in LINK_COLUMNS:
        assert col not in out  # the captured row: the translation below is the whole change
        out[col] = (row.get("extra") or {}).get(col)
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
        payload, want_payload = dict(got.payload), _v2(want["payload"])
        # The one value that is the time of the run.
        date, want_date = payload.pop("ingestion_date"), want_payload.pop("ingestion_date")
        datetime.datetime.fromisoformat(date)
        datetime.datetime.fromisoformat(want_date)
        assert payload == want_payload
        assert set(got.payload) == set(_v2(want["payload"]))
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
            k: v for k, v in _v2(want["payload"]).items() if k != "ingestion_date"}
        assert got.vector == pytest.approx(want["vector"], abs=1e-6)


@pytest.mark.parametrize("document", DOCUMENTS, ids=lambda d: d["row"]["doc_id"])
def test_payload_keys_keep_their_order_unless_a_reserved_key_was_dropped(document, monkeypatch):
    """Key order means nothing to Qdrant, but it shows the payload is assembled
    the same way (the v1 keys in their captured order, then the interval). The one document whose ``extra`` held keys the engine writes
    itself is the exception: the previous pipeline overwrote them in place, the
    source now leaves them out (same keys, same values, another position)."""
    item, _written, client = _ingest(document, monkeypatch)
    same_order = all(list(got.payload) == list(_v2(want["payload"]))
                     for got, want in zip(client.written, document["points"]))
    assert same_order == (not item.dropped_keys)


def test_the_stale_reserved_keys_of_an_extra_are_dropped_and_reported(monkeypatch):
    (document,) = [d for d in DOCUMENTS if d["row"]["doc_id"] == "us-a3-2024-06-12-minutes"]
    item, _written, client = _ingest(document, monkeypatch)
    assert sorted(item.dropped_keys) == ["chunk_number", "filename", "ingestion_date", "text"]
    first = client.written[0].payload
    assert first["filename"] == "us-a3-2024-06-12-minutes.txt" and first["chunk_number"] == 0
    assert first["meeting"] == "June 2024"  # the rest of the extra is kept


def test_a_precision_and_a_kind_from_the_extra_add_only_themselves_and_their_interval(monkeypatch):
    """The corrected vault rows carry ``date_precision`` and ``date_kind`` in
    ``extra``: on the captured documents they reach the payload as two more
    keys of the metadata, the interval follows the precision, and the ids, the
    vectors and every other key stay the captured ones."""
    (document,) = [d for d in DOCUMENTS if d["row"]["doc_id"] == "au-e1-2022-02-statement"]
    row = dict(document["row"], extra={"date_precision": "month", "date_kind": "period"})
    item, written, client = _ingest({**document, "row": row}, monkeypatch)
    assert written == len(document["points"])
    assert [p.id for p in client.written] == [p["id"] for p in document["points"]]
    for got, want in zip(client.written, document["points"]):
        assert got.vector == want["vector"]
        want_payload = dict(want["payload"])
        payload = dict(got.payload)
        payload.pop("ingestion_date"), want_payload.pop("ingestion_date")
        assert payload.pop("date_precision") == "month"
        assert payload.pop("date_kind") == "period"
        assert payload.pop("date_start") == "2022-02-01T00:00:00Z"
        assert payload.pop("date_end") == "2022-02-28T00:00:00Z"
        assert payload == want_payload
