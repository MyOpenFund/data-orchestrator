"""Unit tests for the rewired engine internals (fake store/embedder, real chunking on .md)."""
from pathlib import Path

import numpy as np

from rag_orchestrator.core import (
    OCR_TO_ENGINE,
    Ledger,
    SourceItem,
    _point_id,
    ingest_item,
    run_ingest,
)


class FakeEmbedder:
    dim = 8

    def __init__(self):
        self.calls = []

    def encode_passage(self, texts):
        self.calls.append(list(texts))
        return np.zeros((len(texts), self.dim), dtype="float32")


class FakeClient:
    def __init__(self, log):
        self._log = log
        self.points = []

    def upsert(self, collection_name, points, **kwargs):
        self._log.append(("qdrant", collection_name, len(points)))
        self.points.extend(points)


class FakeStore:
    def __init__(self, log):
        self._log = log
        self.client = FakeClient(log)
        self.ensured = []

    def ensure_collection(self, name, vector_size):
        self.ensured.append((name, vector_size))
        return vector_size


class RecordingLedger:
    def __init__(self, log):
        self._log = log
        self.marks = []

    def __contains__(self, doc_id):
        return False

    def __len__(self):
        return 0

    def mark(self, doc_id, chunks, payload=None):
        self._log.append(("ledger", doc_id, chunks))
        self.marks.append((doc_id, chunks, payload))


def _md_item(tmp_path, doc_id="doc1", payload=None):
    p = tmp_path / f"{doc_id}.md"
    p.write_text("""# Title

This is a substantial paragraph with enough content to ensure the chunking algorithm produces chunks. It contains multiple sentences and enough text to meet the minimum length requirements for proper chunk creation. Let's add more content here to ensure we get at least one chunk from this markdown file. The chunker needs sufficient text volume to work properly.

And here's another paragraph with additional content. We want to ensure we have enough material to trigger the chunking process correctly. The algorithm likely has minimum thresholds that must be met for chunks to be produced from the input text.
""")
    return SourceItem(doc_id=doc_id, path=p, payload=payload or {"bank_code": "us"})


def test_ocr_mapping_constant():
    assert OCR_TO_ENGINE == {"auto": None, "always": "always", "never": "never"}


def test_ingest_item_md_yields_points_with_deterministic_ids(tmp_path):
    log = []
    store, embedder = FakeStore(log), FakeEmbedder()
    item = _md_item(tmp_path)
    n = ingest_item(item, "coll", store=store, embedder=embedder)
    assert n >= 1
    ids = [pt.id for pt in store.client.points]
    # Non-paginated formats: page falls back to 0.
    assert ids[0] == _point_id("doc1", 0, 0)
    payload = store.client.points[0].payload
    assert payload["doc_id"] == "doc1"
    assert payload["bank_code"] == "us"
    assert payload["chunk_index"] == 0
    assert "text" in payload and payload["filename"] == item.path.name


def test_vectors_come_from_encode_passage(tmp_path):
    log = []
    store, embedder = FakeStore(log), FakeEmbedder()
    ingest_item(_md_item(tmp_path), "coll", store=store, embedder=embedder)
    assert embedder.calls, "encode_passage must be used (E5 prefix discipline)"
    assert len(store.client.points[0].vector) == embedder.dim


def test_run_ingest_marks_ledger_after_qdrant(tmp_path):
    log = []
    store, embedder, ledger = FakeStore(log), FakeEmbedder(), RecordingLedger(log)
    stats = run_ingest(
        [_md_item(tmp_path)], collection="coll",
        ledger=ledger, store=store, embedder=embedder,
    )
    assert stats.docs_ingested == 1
    kinds = [e[0] for e in log]
    assert kinds.index("qdrant") < kinds.index("ledger"), "write protocol: Qdrant first"
    assert store.ensured == [("coll", embedder.dim)]
    assert ledger.marks[0][2] == {"bank_code": "us"}  # payload forwarded


def test_missing_file_is_isolated(tmp_path):
    log = []
    store, embedder = FakeStore(log), FakeEmbedder()
    bad = SourceItem(doc_id="ghost", path=tmp_path / "nope.pdf", payload={})
    ok = _md_item(tmp_path)
    stats = run_ingest(
        [bad, ok], collection="coll",
        ledger=RecordingLedger(log), store=store, embedder=embedder,
    )
    assert stats.docs_error == 1 and stats.docs_ingested == 1


def test_ledger_failure_is_isolated(tmp_path):
    log = []

    class ExplodingLedger(RecordingLedger):
        def mark(self, doc_id, chunks, payload=None):
            raise RuntimeError("FK violation")

    stats = run_ingest(
        [_md_item(tmp_path)], collection="coll",
        ledger=ExplodingLedger(log), store=FakeStore(log), embedder=FakeEmbedder(),
    )
    assert stats.docs_error == 1 and stats.docs_ingested == 0


def test_file_ledger_accepts_payload_kwarg(tmp_path):
    led = Ledger(tmp_path / "ledger.jsonl")
    led.mark("d1", 3, payload={"bank_code": "us"})
    assert "d1" in led


def test_limit_stops_after_n_newly_ingested(tmp_path):
    log = []
    items = [_md_item(tmp_path, doc_id=f"doc{i}") for i in range(3)]
    stats = run_ingest(
        items, collection="coll", ledger=RecordingLedger(log),
        store=FakeStore(log), embedder=FakeEmbedder(), limit=2,
    )
    assert stats.docs_ingested == 2 and stats.docs_seen == 2
