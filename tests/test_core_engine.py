"""Unit tests for the rewired engine internals (fake store/embedder, real chunking on .md)."""
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from data_orchestrator.core import (
    OCR_TO_ENGINE,
    Ledger,
    SourceItem,
    _ensure_collection,
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

    import data_orchestrator.core as core

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
    import data_orchestrator.core as core

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
    import data_orchestrator.core as core

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


def test_file_ledger_resumes_from_disk_despite_a_corrupt_line(tmp_path):
    """A ledger is appended to after every document, so a crash mid-write
    leaves a torn last line. Restarting must resume from the lines that ARE
    intact instead of raising and forcing a full re-ingest of the corpus."""
    state = tmp_path / "state" / "central-bank-e5b-v1.jsonl"
    state.parent.mkdir()
    state.write_text(
        '{"doc_id": "done", "chunks": 12, "ts": 1}\n'
        '{"doc_id": "torn", "chunks":\n'          # crashed mid-append
        '\n',                                     # trailing blank line
        encoding="utf-8",
    )

    led = Ledger(state)

    assert "done" in led
    assert "torn" not in led
    assert len(led) == 1


def test_file_ledger_ignores_a_line_without_a_doc_id(tmp_path):
    """Well-formed JSON is not enough: a row from an older/other writer that
    carries no doc_id must be dropped, not crash the resume with a KeyError."""
    state = tmp_path / "ledger.jsonl"
    state.write_text(
        '{"chunks": 3, "ts": 1}\n'
        '{"doc_id": "done", "chunks": 12, "ts": 2}\n',
        encoding="utf-8",
    )

    led = Ledger(state)

    assert "done" in led
    assert len(led) == 1


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

    import data_orchestrator.core as core

    def raise_tnf(path, use_ocr=None):
        raise TextNotFoundException("no text layer")

    monkeypatch.setattr(core, "chunk_with_chunknorris", raise_tnf)
    log = []
    item = _md_item(tmp_path)
    n = ingest_item(item, "coll", store=FakeStore(log), embedder=FakeEmbedder())
    assert n == 0
    assert not log  # nothing reached Qdrant


def test_run_ingest_counts_path_derived_metadata_items(tmp_path):
    """The cb_corpus connector flags documents the manifest has not indexed
    with metadata_source="path" (year-only dates). The run must count them —
    including ones skipped by the resume ledger — so the summary and the run
    report can say how much of the corpus carries degraded metadata."""
    log = []

    class SkipSecond(RecordingLedger):
        def __contains__(self, doc_id):
            return doc_id == "d2"

    items = [
        _md_item(tmp_path, doc_id="d1", payload={"bank_code": "us", "metadata_source": "manifest"}),
        _md_item(tmp_path, doc_id="d2", payload={"bank_code": "us", "metadata_source": "path"}),
        _md_item(tmp_path, doc_id="d3", payload={"bank_code": "us", "metadata_source": "path"}),
        _md_item(tmp_path, doc_id="d4", payload={"bank_code": "us"}),  # no flag at all
    ]
    stats = run_ingest(items, collection="c", ledger=SkipSecond(log),
                       store=FakeStore(log), embedder=FakeEmbedder())
    assert stats.docs_path_metadata == 2
    assert stats.docs_skipped_resume == 1


class RacingClient:
    def __init__(self, exists, size=8, exists_raises=False):
        self._exists, self._size, self._raises = exists, size, exists_raises

    def collection_exists(self, name):
        if self._raises:
            raise ConnectionError("qdrant down")
        return self._exists

    def get_collection(self, collection_name):
        return SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(
            vectors=SimpleNamespace(size=self._size))))


class RacingStore:
    def __init__(self, client):
        self.client = client

    def ensure_collection(self, name, vector_size):
        raise RuntimeError("Wrong input: Collection c already exists!")


def test_losing_the_creation_race_is_not_fatal():
    _ensure_collection(RacingStore(RacingClient(exists=True)), "c", 8)


def test_a_failure_that_left_no_collection_is_reraised_unchanged():
    with pytest.raises(RuntimeError, match="already exists"):
        _ensure_collection(RacingStore(RacingClient(exists=False)), "c", 8)


def test_an_unreachable_store_reraises_the_original_error():
    with pytest.raises(RuntimeError, match="already exists"):
        _ensure_collection(RacingStore(RacingClient(exists=True, exists_raises=True)), "c", 8)


def test_race_winner_with_another_dimension_is_refused():
    with pytest.raises(RuntimeError, match="384-d"):
        _ensure_collection(RacingStore(RacingClient(exists=True, size=384)), "c", 8)


def test_existing_collection_with_another_dimension_is_refused():
    class Existing:
        client = None

        def ensure_collection(self, name, vector_size):
            return 384

    with pytest.raises(RuntimeError, match="384-d"):
        _ensure_collection(Existing(), "c", 768)


def test_run_ingest_refuses_a_wrong_dimension_before_touching_any_document(tmp_path):
    log = []
    store = FakeStore(log)
    store.ensure_collection = lambda name, vector_size: vector_size * 2
    with pytest.raises(RuntimeError, match="-d vectors"):
        run_ingest([_md_item(tmp_path)], collection="c", ledger=RecordingLedger(log),
                   store=store, embedder=FakeEmbedder())
    assert log == []


def test_race_winner_with_named_vectors_is_refused():
    """A named-vector collection has no single ``size``: the dimension cannot
    be verified, so the fallback refuses it rather than assuming a match."""
    client = RacingClient(exists=True, size=None)
    with pytest.raises(RuntimeError, match="cannot verify the dimension of a named-vector collection"):
        _ensure_collection(RacingStore(client), "c", 8)


def test_a_failing_get_collection_after_the_race_reraises_the_original_error():
    class GetFails(RacingClient):
        def get_collection(self, collection_name):
            raise ConnectionError("qdrant went away")

    with pytest.raises(RuntimeError, match="already exists") as excinfo:
        _ensure_collection(RacingStore(GetFails(exists=True)), "c", 8)
    assert excinfo.value.__suppress_context__  # raised "from None"


def test_an_unreachable_store_reraises_the_original_error_from_none():
    with pytest.raises(RuntimeError, match="already exists") as excinfo:
        _ensure_collection(RacingStore(RacingClient(exists=True, exists_raises=True)), "c", 8)
    assert excinfo.value.__suppress_context__


def _racing_qdrant_store(other_dim):
    """eigenmind's real QdrantStore over an in-memory Qdrant, raced.

    eigenmind's ensure_collection is get-then-create. The wrapper makes its
    first get_collection see nothing, then lets "another process" create the
    collection (at ``other_dim``) just before eigenmind's own create, which the
    real client then refuses ("already exists").
    """
    qdrant_client = pytest.importorskip("qdrant_client")
    store_mod = pytest.importorskip("eigenmind.vectordb.store")
    from qdrant_client import models

    inner = qdrant_client.QdrantClient(":memory:")

    class Racing:
        def __init__(self):
            self.gets = 0

        def get_collection(self, collection_name):
            self.gets += 1
            if self.gets == 1:
                raise ValueError(f"Collection {collection_name} not found")
            return inner.get_collection(collection_name=collection_name)

        def create_collection(self, collection_name, vectors_config):
            inner.create_collection(  # the concurrent winner
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=other_dim,
                                                   distance=models.Distance.COSINE),
            )
            return inner.create_collection(collection_name=collection_name,
                                           vectors_config=vectors_config)

        def __getattr__(self, name):
            return getattr(inner, name)

    store = store_mod.QdrantStore.__new__(store_mod.QdrantStore)
    store._client = Racing()
    return store, inner


def test_real_qdrant_store_losing_the_race_is_not_fatal():
    store, inner = _racing_qdrant_store(other_dim=8)
    with pytest.raises(Exception):  # the bare engine call does lose the race
        store.ensure_collection("c", 8)
    store2, inner2 = _racing_qdrant_store(other_dim=8)
    _ensure_collection(store2, "c", 8)
    assert inner2.collection_exists("c")


def test_real_qdrant_store_race_winner_with_another_dimension_is_refused():
    store, _inner = _racing_qdrant_store(other_dim=384)
    with pytest.raises(RuntimeError, match="384-d"):
        _ensure_collection(store, "c", 8)


# ---------------------------------------------------------------------------
# batch_size: an explicit value is honoured as given (never `or`-defaulted),
# and a value below 1 is refused before any store/engine work.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [0, -1])
def test_run_ingest_refuses_a_batch_size_below_one_before_any_work(tmp_path, bad):
    log = []
    store = FakeStore(log)
    with pytest.raises(ValueError, match="batch_size"):
        run_ingest([_md_item(tmp_path)], collection="coll", ledger=RecordingLedger(log),
                   store=store, embedder=FakeEmbedder(), batch_size=bad)
    assert store.ensured == [] and log == []


@pytest.mark.parametrize("bad", [0, -1])
def test_ingest_item_refuses_a_batch_size_below_one(tmp_path, bad):
    with pytest.raises(ValueError, match="batch_size"):
        ingest_item(_md_item(tmp_path), "coll", store=FakeStore([]),
                    embedder=FakeEmbedder(), batch_size=bad)


def test_batch_size_one_upserts_one_point_per_call(tmp_path):
    log = []
    store = FakeStore(log)
    n = ingest_item(_md_item(tmp_path), "coll", store=store, embedder=FakeEmbedder(),
                    batch_size=1)
    assert n >= 1
    assert [e[2] for e in log if e[0] == "qdrant"] == [1] * n


def test_batch_size_none_uses_the_engine_default(tmp_path, monkeypatch):
    import data_orchestrator.core as core_mod
    monkeypatch.setattr(core_mod, "_default_batch_size", lambda: 1)
    log = []
    n = ingest_item(_md_item(tmp_path), "coll", store=FakeStore(log), embedder=FakeEmbedder())
    assert [e[2] for e in log if e[0] == "qdrant"] == [1] * n
