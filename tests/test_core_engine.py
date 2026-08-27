"""Unit tests for the rewired engine internals (fake store/embedder, real chunking on .md)."""
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

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
    assert payload["chunk_number"] == 0
    assert "chunk_index" not in payload  # eigenmind reads "chunk_number", not this
    assert "text" in payload and payload["filename"] == item.path.name


def test_ingest_item_payload_has_ingestion_date_constant_across_doc(tmp_path):
    log = []
    store, embedder = FakeStore(log), FakeEmbedder()
    item = _md_item(tmp_path)
    n = ingest_item(item, "coll", store=store, embedder=embedder)
    assert n >= 1
    dates = {pt.payload["ingestion_date"] for pt in store.client.points}
    assert len(dates) == 1  # every point of the same document agrees
    (ingestion_date,) = dates
    # ISO-parseable string, as eigenmind's date_range_filter expects.
    datetime.fromisoformat(ingestion_date)


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


def test_ledger_failure_on_empty_doc_is_isolated_and_run_continues(tmp_path, monkeypatch):
    """The n == 0 ('empty doc') ledger mark must be inside the same
    per-document isolation as the n > 0 path (item 2 of the fix wave): an
    empty doc unknown to the vault (FK violation on mark) must count as a
    docs_error for that one doc and let the run continue, not abort."""
    from chunknorris.exceptions import TextNotFoundException

    import rag_orchestrator.core as core

    original_chunk = core.chunk_with_chunknorris
    calls = {"n": 0}

    def flaky_chunk(path, use_ocr=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TextNotFoundException("no text layer")
        return original_chunk(path, use_ocr=use_ocr)

    monkeypatch.setattr(core, "chunk_with_chunknorris", flaky_chunk)

    log = []

    class ExplodingLedger(RecordingLedger):
        def mark(self, doc_id, chunks, payload=None):
            if chunks == 0:
                raise RuntimeError("FK violation: doc unknown to vault")
            super().mark(doc_id, chunks, payload)

    empty_item = SourceItem(doc_id="empty-doc", path=tmp_path / "empty.pdf", payload={})
    ok_item = _md_item(tmp_path, doc_id="doc2")
    stats = run_ingest(
        [empty_item, ok_item], collection="coll",
        ledger=ExplodingLedger(log), store=FakeStore(log), embedder=FakeEmbedder(),
    )
    assert stats.docs_error == 1
    assert stats.docs_empty == 0  # mark() failed -> counted as error, not also empty
    assert stats.docs_ingested == 1  # run continued past the poisoned doc


def test_ensure_collection_failure_still_releases_owned_embedder(tmp_path, monkeypatch):
    """ensure_collection() must be inside the try/finally that releases an
    *owned* embedder (item 6 of the fix wave) — otherwise a raising
    ensure_collection leaks the loaded model."""
    import rag_orchestrator.core as core

    released = {"n": 0}

    class FakeOwnedEmbedder:
        dim = 8

        def encode_passage(self, texts):
            return np.zeros((len(texts), self.dim), dtype="float32")

        def release(self):
            released["n"] += 1

    monkeypatch.setattr(core, "EmbeddingModel", lambda **kwargs: FakeOwnedEmbedder())
    monkeypatch.setattr(core, "detect_device", lambda: "cpu")

    class BoomStore:
        def ensure_collection(self, name, vector_size):
            raise RuntimeError("qdrant unreachable")

    with pytest.raises(RuntimeError, match="qdrant unreachable"):
        run_ingest([_md_item(tmp_path)], collection="coll", store=BoomStore())

    assert released["n"] == 1  # not leaked despite ensure_collection raising


def test_ingest_item_omits_page_key_when_start_page_is_none(tmp_path):
    """Non-paginated formats (e.g. markdown) must not fake a 'page': 0 in the
    payload (item 7 of the fix wave) — the point id still resolves the
    missing page to 0 for determinism, but the payload key is absent."""
    log = []
    store, embedder = FakeStore(log), FakeEmbedder()
    item = _md_item(tmp_path)
    ingest_item(item, "coll", store=store, embedder=embedder)
    payload = store.client.points[0].payload
    assert "page" not in payload
    assert store.client.points[0].id == _point_id("doc1", 0, 0)


def test_ingest_item_includes_page_key_when_start_page_present(tmp_path, monkeypatch):
    import rag_orchestrator.core as core

    class FakeChunk:
        def __init__(self, text, start_page):
            self._text = text
            self.start_page = start_page

        def get_text(self):
            return self._text

    monkeypatch.setattr(
        core, "chunk_with_chunknorris",
        lambda path, use_ocr=None: [FakeChunk("hello world " * 10, 3)],
    )
    log = []
    store, embedder = FakeStore(log), FakeEmbedder()
    item = SourceItem(doc_id="doc-pdf", path=tmp_path / "f.pdf", payload={})
    n = ingest_item(item, "coll", store=store, embedder=embedder)
    assert n == 1
    assert store.client.points[0].payload["page"] == 3


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


def test_text_not_found_maps_to_empty_doc(tmp_path, monkeypatch):
    from chunknorris.exceptions import TextNotFoundException

    import rag_orchestrator.core as core

    def raise_tnf(path, use_ocr=None):
        raise TextNotFoundException("no text layer")

    monkeypatch.setattr(core, "chunk_with_chunknorris", raise_tnf)
    log = []
    item = _md_item(tmp_path)
    n = ingest_item(item, "coll", store=FakeStore(log), embedder=FakeEmbedder())
    assert n == 0
    assert not log  # nothing reached Qdrant
