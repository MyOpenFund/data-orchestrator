"""Unit tests for the run loop around the engine (recording store, fake embedder, real chunking)."""
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from qdrant_client import QdrantClient, models
from rag_engine.chunking import Chunk
from rag_engine.config import ConfigError, Settings
from rag_engine.embedding import FakeEmbedder as EngineFakeEmbedder
from rag_engine.embedding import ModelSpec
from rag_engine.schema import point_id
from rag_engine.store import CollectionMismatchError, QdrantAuthError, QdrantStore

from data_orchestrator.core import (
    Ledger,
    SourceItem,
    engine_settings,
    ingest_item,
    run_ingest,
    split_reserved_keys,
)

DIM = 8


def FakeEmbedder(dim=DIM):  # noqa: N802 — reads as the class it builds
    """The engine's deterministic fake: ``calls`` lists the prefixed texts it embedded."""
    return EngineFakeEmbedder(ModelSpec("fake-model", dim, "passage: ", "query: "))


class FakeClient:
    """The part of a Qdrant client the engine's ``QdrantStore`` uses for a run,
    recording what it is asked to write."""

    def __init__(self, log):
        self._log = log
        self.points = []
        self.sizes = {}  # collection -> stored vector size
        self.waits = []

    def collection_exists(self, collection_name):
        return collection_name in self.sizes

    def get_collection(self, collection_name):
        return SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(
            vectors=SimpleNamespace(size=self.sizes[collection_name],
                                    distance=models.Distance.COSINE))))

    def create_collection(self, collection_name, vectors_config):
        self.sizes[collection_name] = vectors_config.size

    def upsert(self, collection_name, points, wait=True):
        self._log.append(("qdrant", collection_name, len(points)))
        self.waits.append(wait)
        self.points.extend(points)


class FakeStore(QdrantStore):
    """The engine's real store over ``FakeClient``; ``ensured`` lists its
    ``ensure_collection`` calls."""

    def __init__(self, log, settings=None):
        super().__init__(client=FakeClient(log), settings=settings or Settings())
        self.ensured = []

    def ensure_collection(self, name, dim):
        self.ensured.append((name, dim))
        return super().ensure_collection(name, dim)


def _memory_store():
    """The engine's real store over qdrant-client's local mode."""
    client = QdrantClient(":memory:")
    return QdrantStore(client=client, settings=Settings()), client


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
    return SourceItem(doc_id=doc_id, path=p, payload=payload or {"source_code": "us"})


def _write_blank_pdf(path):
    """A one-page PDF without a text layer: an empty document when read without OCR."""
    import fitz

    doc = fitz.open()
    doc.new_page()
    doc.save(path)
    doc.close()
    return path


@pytest.mark.parametrize("mode", ["auto", "always", "never"])
def test_the_ocr_mode_and_languages_reach_the_engine_as_given(tmp_path, monkeypatch, mode):
    """The CLI modes are the engine's, one to one; the languages are passed
    explicitly (left out, the engine would use "eng" whatever the settings)."""
    import rag_engine.ingest as engine_ingest

    seen = []

    def chunk_file(path, *, ocr, ocr_languages, options):
        seen.append((Path(path).name, ocr, ocr_languages))
        return [Chunk(text="hello world " * 10, page=None)]

    monkeypatch.setattr(engine_ingest, "chunk_file", chunk_file)
    stats = run_ingest([_md_item(tmp_path)], collection="c", store=FakeStore([]),
                       embedder=FakeEmbedder(), ocr=mode,
                       settings=Settings(ocr_languages="eng+fra"))
    assert stats.docs_ingested == 1
    assert seen == [("doc1.md", mode, "eng+fra")]


def test_ocr_languages_come_from_the_environment_by_default(tmp_path, monkeypatch):
    import rag_engine.ingest as engine_ingest

    seen = []

    def chunk_file(path, *, ocr, ocr_languages, options):
        seen.append(ocr_languages)
        return []

    monkeypatch.setattr(engine_ingest, "chunk_file", chunk_file)
    monkeypatch.setenv("RAG_ENGINE_OCR_LANGUAGES", "deu+eng")
    run_ingest([_md_item(tmp_path)], collection="c", store=FakeStore([]),
               embedder=FakeEmbedder())
    assert seen == ["deu+eng"]


def test_ingest_item_md_yields_points_with_deterministic_ids(tmp_path):
    log = []
    store, embedder = FakeStore(log), FakeEmbedder()
    item = _md_item(tmp_path)
    n = ingest_item(item, "coll", store=store, embedder=embedder)
    assert n >= 1
    ids = [pt.id for pt in store.client.points]
    # Non-paginated formats: page falls back to 0.
    assert ids[0] == point_id("doc1", 0, 0)
    payload = store.client.points[0].payload
    assert payload["doc_id"] == "doc1"
    assert payload["source_code"] == "us"
    assert payload["chunk_number"] == 0
    assert "chunk_index" not in payload  # schema v1 says "chunk_number", not this
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
    # ISO-parseable string (payload schema v1).
    datetime.fromisoformat(ingestion_date)


def test_vectors_come_from_encode_passage(tmp_path):
    log = []
    store, embedder = FakeStore(log), FakeEmbedder()
    ingest_item(_md_item(tmp_path), "coll", store=store, embedder=embedder)
    assert embedder.calls, "encode_passage must be used (E5 prefix discipline)"
    assert all(text.startswith("passage: ") for text in embedder.calls)
    assert len(store.client.points[0].vector) == embedder.dim
    assert store.client.waits and all(store.client.waits)  # each write waits for Qdrant


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
    assert ledger.marks[0][2] == {"source_code": "us"}  # payload forwarded


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


def test_ledger_failure_on_empty_doc_is_isolated_and_run_continues(tmp_path):
    """The n == 0 ('empty doc') ledger mark must be inside the same
    per-document isolation as the n > 0 path: an
    empty doc unknown to the vault (FK violation on mark) must count as a
    docs_error for that one doc and let the run continue, not abort."""
    log = []

    class ExplodingLedger(RecordingLedger):
        def mark(self, doc_id, chunks, payload=None):
            if chunks == 0:
                raise RuntimeError("FK violation: doc unknown to vault")
            super().mark(doc_id, chunks, payload)

    empty_item = SourceItem(doc_id="empty-doc",
                            path=_write_blank_pdf(tmp_path / "empty.pdf"), payload={})
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
    *owned* embedder — otherwise a raising
    ensure_collection leaks the loaded model."""
    import data_orchestrator.core as core

    released = {"n": 0}
    built = []

    class OwnedEmbedder(EngineFakeEmbedder):
        def release(self):
            released["n"] += 1

    def build_embedder(settings):
        built.append(settings)
        return OwnedEmbedder(ModelSpec.from_settings(settings))

    monkeypatch.setattr(core, "build_embedder", build_embedder)

    class BoomStore:
        def ensure_collection(self, name, vector_size):
            raise RuntimeError("qdrant unreachable")

    with pytest.raises(RuntimeError, match="qdrant unreachable"):
        run_ingest([_md_item(tmp_path)], collection="coll", store=BoomStore())

    assert released["n"] == 1  # not leaked despite ensure_collection raising
    assert built[0].model_name == "intfloat/multilingual-e5-base" and built[0].model_dim == 768


def test_an_injected_embedder_is_not_released(tmp_path):
    embedder = FakeEmbedder()
    run_ingest([_md_item(tmp_path)], collection="coll", store=FakeStore([]), embedder=embedder)
    embedder.encode_passage(["still usable"])


def test_ingest_item_omits_page_key_when_start_page_is_none(tmp_path):
    """Non-paginated formats (e.g. markdown) must not fake a 'page': 0 in the
    payload — the point id still resolves the
    missing page to 0 for determinism, but the payload key is absent."""
    log = []
    store, embedder = FakeStore(log), FakeEmbedder()
    item = _md_item(tmp_path)
    ingest_item(item, "coll", store=store, embedder=embedder)
    payload = store.client.points[0].payload
    assert "page" not in payload
    assert store.client.points[0].id == point_id("doc1", 0, 0)


def test_ingest_item_includes_page_key_when_start_page_present(tmp_path, monkeypatch):
    import rag_engine.ingest as engine_ingest

    monkeypatch.setattr(
        engine_ingest, "chunk_file",
        lambda path, **kwargs: [Chunk(text="hello world " * 10, page=3)],
    )
    log = []
    store, embedder = FakeStore(log), FakeEmbedder()
    item = SourceItem(doc_id="doc-pdf", path=tmp_path / "f.pdf", payload={})
    n = ingest_item(item, "coll", store=store, embedder=embedder)
    assert n == 1
    assert store.client.points[0].payload["page"] == 3
    assert store.client.points[0].id == point_id("doc-pdf", 3, 0)


def test_a_real_pdf_gets_its_pages(tmp_path):
    import fitz

    path = tmp_path / "two-pages.pdf"
    doc = fitz.open()
    for i in (1, 2):
        doc.new_page().insert_text((72, 72), f"Monetary policy page {i}. " * 30)
    doc.save(path)
    doc.close()
    store = FakeStore([])
    n = ingest_item(SourceItem("doc-pdf", path, {}), "coll", store=store,
                    embedder=FakeEmbedder(), ocr="never")
    assert n >= 1
    assert all(isinstance(p.payload["page"], int) for p in store.client.points)
    assert store.client.points[0].id == point_id("doc-pdf", store.client.points[0].payload["page"], 0)


def test_file_ledger_accepts_payload_kwarg(tmp_path):
    led = Ledger(tmp_path / "ledger.jsonl")
    led.mark("d1", 3, payload={"source_code": "us"})
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


def test_a_document_without_text_is_an_empty_doc_not_an_error(tmp_path):
    """A PDF with no text layer, read without OCR: 0 chunks, nothing reaches
    the embedder or Qdrant, and the run records it as empty (ledger mark 0)."""
    log = []
    embedder = FakeEmbedder()
    item = SourceItem("scan", _write_blank_pdf(tmp_path / "scan.pdf"), {})
    n = ingest_item(item, "coll", store=FakeStore(log), embedder=embedder, ocr="never")
    assert n == 0
    assert not log and embedder.calls == []  # nothing reached Qdrant

    ledger = RecordingLedger(log)
    stats = run_ingest([item], collection="coll", ledger=ledger, store=FakeStore(log),
                       embedder=embedder, ocr="never")
    assert (stats.docs_empty, stats.docs_ingested, stats.docs_error) == (1, 0, 0)
    assert log == [("ledger", "scan", 0)]


def test_an_empty_text_file_is_an_empty_doc(tmp_path):
    path = tmp_path / "empty.md"
    path.write_text("")
    assert ingest_item(SourceItem("e", path, {}), "coll", store=FakeStore([]),
                       embedder=FakeEmbedder()) == 0


def test_an_unsupported_format_is_one_isolated_error(tmp_path):
    path = tmp_path / "page.html"
    path.write_text("<p>hello</p>")
    stats = run_ingest([SourceItem("h", path, {}), _md_item(tmp_path)], collection="coll",
                       store=FakeStore([]), embedder=FakeEmbedder())
    assert stats.docs_error == 1 and stats.docs_ingested == 1
    assert stats.errors[0][1].startswith("UnsupportedFormatError:")


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
        _md_item(tmp_path, doc_id="d1", payload={"source_code": "us", "metadata_source": "manifest"}),
        _md_item(tmp_path, doc_id="d2", payload={"source_code": "us", "metadata_source": "path"}),
        _md_item(tmp_path, doc_id="d3", payload={"source_code": "us", "metadata_source": "path"}),
        _md_item(tmp_path, doc_id="d4", payload={"source_code": "us"}),  # no flag at all
    ]
    stats = run_ingest(items, collection="c", ledger=SkipSecond(log),
                       store=FakeStore(log), embedder=FakeEmbedder())
    assert stats.docs_path_metadata == 2
    assert stats.docs_skipped_resume == 1


# ---------------------------------------------------------------------------
# Collection guard: the engine's (QdrantStore.ensure_collection, tested there
# in full). Here: what a run does with it — one fatal before any document.
# ---------------------------------------------------------------------------
def test_run_ingest_refuses_a_wrong_dimension_before_touching_any_document(tmp_path):
    log = []
    store = FakeStore(log)
    store.client.sizes["c"] = DIM * 2
    embedder = FakeEmbedder()
    with pytest.raises(CollectionMismatchError,
                       match=r"found 16-d vectors: use another --collection$"):
        run_ingest([_md_item(tmp_path)], collection="c", ledger=RecordingLedger(log),
                   store=store, embedder=embedder)
    assert log == [] and embedder.calls == []


def test_run_ingest_creates_a_missing_collection_with_the_model_dimension(tmp_path):
    store, client = _memory_store()
    stats = run_ingest([_md_item(tmp_path)], collection="c", store=store,
                       embedder=FakeEmbedder())
    vectors = client.get_collection("c").config.params.vectors
    assert (vectors.size, vectors.distance) == (DIM, models.Distance.COSINE)
    assert client.count("c").count == stats.chunks_written >= 1


def test_run_ingest_refuses_a_named_vector_collection(tmp_path):
    store, client = _memory_store()
    client.create_collection(
        collection_name="c",
        vectors_config={"text": models.VectorParams(size=DIM, distance=models.Distance.COSINE)},
    )
    log = []
    with pytest.raises(CollectionMismatchError,
                       match=r"found named vectors \('text'\): use another --collection$"):
        run_ingest([_md_item(tmp_path)], collection="c", ledger=RecordingLedger(log),
                   store=store, embedder=FakeEmbedder())
    assert log == []


def test_run_ingest_refuses_a_collection_that_is_not_cosine(tmp_path):
    """New with rag-engine: the previous guard only looked at the dimension."""
    store, client = _memory_store()
    client.create_collection(
        collection_name="c",
        vectors_config=models.VectorParams(size=DIM, distance=models.Distance.DOT),
    )
    log = []
    with pytest.raises(CollectionMismatchError,
                       match=r"found Dot distance: use another --collection$"):
        run_ingest([_md_item(tmp_path)], collection="c", ledger=RecordingLedger(log),
                   store=store, embedder=FakeEmbedder())
    assert log == []


def test_a_collection_mismatch_is_not_a_runtime_error():
    """The engine's error types replaced this repo's RuntimeError subclasses:
    code that caught RuntimeError for them no longer does."""
    assert not issubclass(CollectionMismatchError, RuntimeError)
    assert not issubclass(QdrantAuthError, RuntimeError)


def test_losing_the_creation_race_is_not_fatal(tmp_path):
    """Two first runs both see "no collection"; the loser's create is refused.
    The engine then finds the winner's collection, checks it and carries on."""
    store, client = _memory_store()
    real_create = client.create_collection

    def racing_create(collection_name, vectors_config):
        real_create(collection_name=collection_name, vectors_config=vectors_config)  # the winner
        return real_create(collection_name=collection_name, vectors_config=vectors_config)

    client.create_collection = racing_create
    stats = run_ingest([_md_item(tmp_path)], collection="c", store=store,
                       embedder=FakeEmbedder())
    assert stats.docs_ingested == 1


def test_a_race_winner_with_another_dimension_is_refused(tmp_path):
    store, client = _memory_store()
    real_create = client.create_collection

    def racing_create(collection_name, vectors_config):
        real_create(collection_name=collection_name,
                    vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE))
        return real_create(collection_name=collection_name, vectors_config=vectors_config)

    client.create_collection = racing_create
    with pytest.raises(CollectionMismatchError, match="384-d"):
        run_ingest([_md_item(tmp_path)], collection="c", store=store, embedder=FakeEmbedder())


# ---------------------------------------------------------------------------
# batch_size: an explicit value is honoured as given (never `or`-defaulted),
# and a value below 1 is refused before any store/engine work.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [0, -1, True, 2.0, "3"])
def test_run_ingest_refuses_a_bad_batch_size_before_any_work(tmp_path, bad):
    log = []
    store = FakeStore(log)
    with pytest.raises(ValueError, match="batch_size"):
        run_ingest([_md_item(tmp_path)], collection="coll", ledger=RecordingLedger(log),
                   store=store, embedder=FakeEmbedder(), batch_size=bad)
    assert store.ensured == [] and log == []


@pytest.mark.parametrize("bad", ["0", "-1", "abc", "1.5"])
def test_a_bad_batch_size_in_the_environment_is_one_fatal_before_any_work(
        tmp_path, monkeypatch, bad):
    """RAG_ENGINE_BATCH_SIZE is read once, with the settings: ONE error for
    the run, not one failure per document inside the engine."""
    monkeypatch.setenv("RAG_ENGINE_BATCH_SIZE", bad)
    log = []
    store = FakeStore(log)
    with pytest.raises(ConfigError, match="RAG_ENGINE_BATCH_SIZE"):
        run_ingest([_md_item(tmp_path, doc_id=f"d{i}") for i in range(3)], collection="coll",
                   ledger=RecordingLedger(log), store=store, embedder=FakeEmbedder())
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


def test_batch_size_none_uses_the_engine_default(tmp_path):
    """None leaves the batch size of the store, i.e. of the engine's settings
    (RAG_ENGINE_BATCH_SIZE, default 100)."""
    log = []
    n = ingest_item(_md_item(tmp_path), "coll",
                    store=FakeStore(log, Settings(batch_size=1)), embedder=FakeEmbedder())
    assert [e[2] for e in log if e[0] == "qdrant"] == [1] * n
    assert Settings().batch_size == 100


def test_the_default_store_gets_the_batch_size_of_the_environment(tmp_path, monkeypatch):
    import data_orchestrator.core as core

    monkeypatch.setenv("RAG_ENGINE_BATCH_SIZE", "1")
    log = []
    monkeypatch.setattr(core, "build_store", lambda settings: FakeStore(log, settings))
    stats = run_ingest([_md_item(tmp_path)], collection="coll", embedder=FakeEmbedder())
    assert [e[2] for e in log if e[0] == "qdrant"] == [1] * stats.chunks_written


def test_run_ingest_forwards_its_batch_size_to_every_document(tmp_path, monkeypatch):
    import rag_engine.ingest as engine_ingest

    monkeypatch.setattr(
        engine_ingest, "chunk_file",
        lambda path, **kwargs: [Chunk(text=f"chunk {i} " * 10, page=None) for i in range(3)])
    log = []
    stats = run_ingest([_md_item(tmp_path, doc_id="a"), _md_item(tmp_path, doc_id="b")],
                       collection="coll", store=FakeStore(log), embedder=FakeEmbedder(),
                       batch_size=2)
    assert stats.chunks_written == 6
    # Three points per document, two per write: the store's own size (100) would send 3.
    assert [e[2] for e in log if e[0] == "qdrant"] == [2, 1, 2, 1]


# ---------------------------------------------------------------------------
# on_target: the Qdrant the run writes to, read from the CONSTRUCTED store,
# reported once before any collection work.
# ---------------------------------------------------------------------------
def test_on_target_reports_the_constructed_store_once_before_collection_work(tmp_path):
    log = []
    store = FakeStore(log)
    store._host, store._port = "qdrant.lan", 6433
    calls = []

    def on_target(host, port):
        calls.append((host, port, list(store.ensured)))

    run_ingest([_md_item(tmp_path)], collection="c", store=store,
               embedder=FakeEmbedder(), on_target=on_target)
    assert calls == [("qdrant.lan", 6433, [])]


def test_on_target_reads_the_default_store_after_construction(tmp_path, monkeypatch):
    import data_orchestrator.core as core

    built = []

    def fake_store(settings):
        store = FakeStore([])
        store._host, store._port = "built", 7000
        built.append(store)
        return store

    monkeypatch.setattr(core, "build_store", fake_store)
    calls = []
    run_ingest([], collection="c", embedder=FakeEmbedder(),
               on_target=lambda h, p: calls.append((h, p)))
    assert len(built) == 1 and calls == [("built", 7000)]


def test_on_target_survives_a_store_without_host_or_port(tmp_path):
    calls = []
    run_ingest([], collection="c", store=FakeStore([]), embedder=FakeEmbedder(),
               on_target=lambda h, p: calls.append((h, p)))
    assert calls == [(None, None)]


def test_the_default_store_is_the_engine_s_built_from_the_environment(monkeypatch):
    import rag_engine.store as engine_store

    import data_orchestrator.core as core

    built = []
    monkeypatch.setattr(engine_store, "QdrantClient",
                        lambda **kwargs: built.append(kwargs) or object())
    for key, value in (("QDRANT_HOST", "qdrant.lan"), ("QDRANT_PORT", "6433"),
                       ("QDRANT_API_KEY", " k3y "), ("RAG_ENGINE_QDRANT_TIMEOUT", "5")):
        monkeypatch.setenv(key, value)
    store = core.build_store(engine_settings())
    assert isinstance(store, QdrantStore) and (store.host, store.port) == ("qdrant.lan", 6433)
    # https is always explicit: left undecided, the client turns TLS on with a key.
    assert built == [{"host": "qdrant.lan", "port": 6433, "api_key": "k3y", "https": False,
                      "timeout": 5.0}]


def test_the_default_store_is_built_from_the_settings_it_is_given_not_the_environment(monkeypatch):
    """The run's settings, whatever the process environment says by then."""
    import rag_engine.store as engine_store

    import data_orchestrator.core as core

    built = []
    monkeypatch.setattr(engine_store, "QdrantClient",
                        lambda **kwargs: built.append(kwargs) or object())
    for key, value in (("QDRANT_HOST", "env.lan"), ("QDRANT_PORT", "1111"),
                       ("QDRANT_API_KEY", "env-key"), ("QDRANT_HTTPS", "false"),
                       ("RAG_ENGINE_QDRANT_TIMEOUT", "5")):
        monkeypatch.setenv(key, value)
    store = core.build_store(Settings(qdrant_host="run.lan", qdrant_port=6433,
                                      qdrant_api_key="run-key", qdrant_https=True,
                                      qdrant_timeout=7.0))
    assert (store.host, store.port) == ("run.lan", 6433)
    assert built == [{"host": "run.lan", "port": 6433, "api_key": "run-key", "https": True,
                      "timeout": 7.0}]


# ---------------------------------------------------------------------------
# source_label: ONE rule for sources[] keys, the unsupported count and the
# count-only breakdown (source_code, then corpus, then unknown). The legacy
# cb_corpus key bank_code is no longer read: every source writes source_code.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("payload, corpus, expected", [
    ({"source_code": "ecb", "bank_code": "fr", "corpus": "central-bank"}, None, "ecb"),
    ({"bank_code": "fr", "corpus": "central-bank"}, None, "central-bank"),
    ({"source_code": None, "corpus": "central-bank"}, None, "central-bank"),
    ({"source_code": None}, "central-bank", "central-bank"),
    ({"corpus": "company"}, "central-bank", "company"),
    ({}, None, "unknown"),
    ("ecb", "central-bank", "ecb"),
    (None, "central-bank", "central-bank"),
    ("", "central-bank", "central-bank"),
    (None, None, "unknown"),
])
def test_source_label(payload, corpus, expected):
    from data_orchestrator.core import source_label
    assert source_label(payload, corpus) == expected


def test_bump_files_a_null_source_vault_item_under_its_corpus():
    from data_orchestrator.core import IngestStats, _bump
    stats = IngestStats()
    _bump(stats, SourceItem("d", Path("x.pdf"), {"corpus": "central-bank"}), new=1)
    assert list(stats.by_source) == ["central-bank"]


# ---------------------------------------------------------------------------
# A Qdrant that refuses the API key (401/403): one clear fatal, at collection
# setup or in the middle of a run; other HTTP errors are not rewritten.
# ---------------------------------------------------------------------------
def _http_error(status):
    import httpx
    from qdrant_client.http.exceptions import UnexpectedResponse
    return UnexpectedResponse(status, "x", b"{}", httpx.Headers())


class _RefusingStore(FakeStore):
    """Every request is answered with ``status``, as a keyed server does to a
    client without the key: the engine's first question (collection_exists)
    is refused."""

    def __init__(self, log, status):
        super().__init__(log)
        self._status = status
        self.client.collection_exists = self._refuse

    def _refuse(self, *a, **k):
        raise _http_error(self._status)


@pytest.mark.parametrize("status", [401, 403])
def test_a_refused_api_key_is_one_clear_fatal_before_any_document(tmp_path, status):
    log = []
    embedder = FakeEmbedder()
    with pytest.raises(QdrantAuthError,
                       match=rf"^Qdrant refused the request \(HTTP {status}\): check QDRANT_API_KEY$"):
        run_ingest([_md_item(tmp_path)], collection="c", ledger=RecordingLedger(log),
                   store=_RefusingStore(log, status), embedder=embedder)
    assert log == [] and embedder.calls == []


def test_another_http_error_at_setup_is_not_rewritten(tmp_path):
    from qdrant_client.http.exceptions import UnexpectedResponse
    log = []
    with pytest.raises(UnexpectedResponse) as info:
        run_ingest([_md_item(tmp_path)], collection="c", ledger=RecordingLedger(log),
                   store=_RefusingStore(log, 500), embedder=FakeEmbedder())
    assert not isinstance(info.value, QdrantAuthError)
    assert log == []


def test_an_error_without_a_status_is_not_rewritten(tmp_path):
    class Boom(FakeStore):
        def ensure_collection(self, name, vector_size):
            raise ConnectionError("refused")
    with pytest.raises(ConnectionError):
        run_ingest([_md_item(tmp_path)], collection="c", store=Boom([]),
                   embedder=FakeEmbedder())


class _KeyRevokedStore(FakeStore):
    """Accepts ``good`` writes, then answers every write with ``status``."""

    def __init__(self, log, good, status):
        super().__init__(log)
        self._good, self._status = good, status
        accept = self.client.upsert

        def upsert(collection_name, points, wait=True):
            if self._good <= 0:
                log.append(("refused", collection_name, len(points)))
                raise _http_error(self._status)
            self._good -= 1
            return accept(collection_name, points, wait=wait)

        self.client.upsert = upsert


@pytest.mark.parametrize("status", [401, 403])
def test_a_key_refused_in_the_middle_of_a_run_is_one_fatal_not_one_error_per_document(
        tmp_path, status):
    log = []
    ledger = RecordingLedger(log)
    items = [_md_item(tmp_path, doc_id=f"doc{i}") for i in range(4)]
    with pytest.raises(QdrantAuthError) as info:
        run_ingest(items, collection="c", ledger=ledger,
                   store=_KeyRevokedStore(log, good=1, status=status),
                   embedder=FakeEmbedder())
    message = str(info.value)
    assert message.startswith(f"Qdrant refused the request (HTTP {status}): check QDRANT_API_KEY")
    assert "document 'doc1'" in message and "after 1 document(s) ingested" in message
    assert info.value.points_written == 0
    # doc0 written then recorded; doc1 refused and NOT recorded; doc2, doc3 never tried.
    assert [e[0] for e in log] == ["qdrant", "ledger", "refused"]
    assert [m[0] for m in ledger.marks] == ["doc0"]


def test_a_fatal_in_the_middle_of_a_run_carries_the_statistics_so_far(tmp_path):
    """The run report of a fatal run is built from these: what was really
    done before the failure, not zeros."""
    items = [_md_item(tmp_path, doc_id=f"doc{i}") for i in range(4)]
    store = _KeyRevokedStore([], good=2, status=401)
    with pytest.raises(QdrantAuthError) as info:
        run_ingest(items, collection="c", ledger=RecordingLedger([]), store=store,
                   embedder=FakeEmbedder())
    stats = info.value.ingest_stats
    # doc0, doc1 ingested; doc2 was being written (seen, neither new nor failed).
    assert (stats.docs_seen, stats.docs_ingested, stats.docs_error, stats.docs_empty) == (3, 2, 0, 0)
    assert stats.chunks_written == len(store.client.points) > 0
    assert stats.by_source == {"us": {"docs_seen": 3, "docs_new": 2, "docs_failed": 0}}
    assert stats.errors == []


def test_a_source_that_fails_in_the_middle_of_a_run_carries_the_statistics_so_far(tmp_path):
    """Any exception that ends the run, not only a refused key: here the
    source itself (a failing selection query, an unreadable folder)."""
    def items():
        yield _md_item(tmp_path, doc_id="doc0")
        yield SourceItem(doc_id="gone", path=tmp_path / "gone.md", payload={"source_code": "fr"})
        raise OSError("corpus folder went away")

    with pytest.raises(OSError, match="corpus folder went away") as info:
        run_ingest(items(), collection="c", store=FakeStore([]), embedder=FakeEmbedder())
    stats = info.value.ingest_stats
    assert (stats.docs_seen, stats.docs_ingested, stats.docs_error) == (2, 1, 1)
    assert stats.by_source == {"us": {"docs_seen": 1, "docs_new": 1, "docs_failed": 0},
                               "fr": {"docs_seen": 1, "docs_new": 0, "docs_failed": 1}}


def test_a_fatal_before_any_document_carries_empty_statistics(tmp_path):
    with pytest.raises(QdrantAuthError) as info:
        run_ingest([_md_item(tmp_path)], collection="c", store=_RefusingStore([], 401),
                   embedder=FakeEmbedder())
    stats = info.value.ingest_stats
    assert (stats.docs_seen, stats.by_source) == (0, {})


def test_another_http_error_in_the_middle_of_a_run_stays_a_per_document_error(tmp_path):
    log = []
    ledger = RecordingLedger(log)
    items = [_md_item(tmp_path, doc_id=f"doc{i}") for i in range(3)]
    stats = run_ingest(items, collection="c", ledger=ledger,
                       store=_KeyRevokedStore(log, good=1, status=500),
                       embedder=FakeEmbedder())
    assert (stats.docs_ingested, stats.docs_error) == (1, 2)
    assert all(msg.startswith("UnexpectedResponse:") for _path, msg in stats.errors)


# ---------------------------------------------------------------------------
# engine_settings: the engine's settings, with this repo's model variable.
# ---------------------------------------------------------------------------
def test_engine_settings_defaults_are_the_production_model():
    settings = engine_settings({})
    assert (settings.model_name, settings.model_dim) == ("intfloat/multilingual-e5-base", 768)
    assert (settings.passage_prefix, settings.backend, settings.batch_size,
            settings.ocr_languages) == ("passage: ", "torch", 100, "eng")


def test_the_orchestrator_model_variable_wins_over_the_engine_s():
    settings = engine_settings({
        "DATA_ORCHESTRATOR_EMBEDDING_MODEL": "org/ours", "RAG_ENGINE_MODEL": "org/engine",
        "DATA_ORCHESTRATOR_EMBEDDING_DIM": "384"})
    assert (settings.model_name, settings.model_dim) == ("org/ours", 384)


def test_the_engine_model_variable_is_used_when_ours_is_unset_and_names_the_collection():
    from data_orchestrator import routing

    env = {"RAG_ENGINE_MODEL": "org/engine-model", "RAG_ENGINE_DIM": "512"}
    settings = engine_settings(env)
    assert (settings.model_name, settings.model_dim) == ("org/engine-model", 512)
    # One resolution: what embeds is what the collection and the ledger are named after.
    assert routing.embedding_model_name(env) == "org/engine-model"


def test_our_dimension_variable_wins_over_the_engine_s():
    settings = engine_settings({"DATA_ORCHESTRATOR_EMBEDDING_MODEL": "org/ours",
                                "DATA_ORCHESTRATOR_EMBEDDING_DIM": "384",
                                "RAG_ENGINE_DIM": "512"})
    assert settings.model_dim == 384


def test_the_default_model_named_explicitly_needs_no_dimension():
    settings = engine_settings(
        {"DATA_ORCHESTRATOR_EMBEDDING_MODEL": "intfloat/multilingual-e5-base"})
    assert settings.model_dim == 768


@pytest.mark.parametrize("env", [
    {"DATA_ORCHESTRATOR_EMBEDDING_MODEL": "org/other"},
    {"RAG_ENGINE_MODEL": "org/other"},
    {"DATA_ORCHESTRATOR_EMBEDDING_MODEL": "org/other", "DATA_ORCHESTRATOR_EMBEDDING_DIM": " "},
])
def test_another_model_without_a_dimension_is_refused(env):
    with pytest.raises(ConfigError, match="DATA_ORCHESTRATOR_EMBEDDING_DIM") as info:
        engine_settings(env)
    assert "org/other" in str(info.value) and "RAG_ENGINE_DIM" in str(info.value)


@pytest.mark.parametrize("bad", ["0", "-1", "abc", "7.5", "+8"])
def test_an_invalid_dimension_is_refused(bad):
    with pytest.raises(ConfigError, match="DATA_ORCHESTRATOR_EMBEDDING_DIM"):
        engine_settings({"DATA_ORCHESTRATOR_EMBEDDING_DIM": bad})


@pytest.mark.parametrize("value", ["onnx", "ONNX", " onnx "])
def test_a_backend_other_than_torch_is_refused(value):
    with pytest.raises(ConfigError, match="RAG_ENGINE_BACKEND=.*ingestion uses the torch backend"):
        engine_settings({"RAG_ENGINE_BACKEND": value})


@pytest.mark.parametrize("value", ["torch", " TORCH ", ""])
def test_the_torch_backend_is_accepted(value):
    assert engine_settings({"RAG_ENGINE_BACKEND": value}).backend == "torch"


def test_engine_settings_reads_the_process_environment_at_call_time(monkeypatch):
    monkeypatch.setenv("RAG_ENGINE_OCR_LANGUAGES", "eng+fra")
    monkeypatch.setenv("RAG_ENGINE_QDRANT_TIMEOUT", "5")
    settings = engine_settings()
    assert (settings.ocr_languages, settings.qdrant_timeout) == ("eng+fra", 5.0)


def test_an_invalid_engine_setting_is_a_config_error_naming_the_variable():
    with pytest.raises(ConfigError, match="RAG_ENGINE_OCR_LANGUAGES"):
        engine_settings({"RAG_ENGINE_OCR_LANGUAGES": "eng fra"})
    assert issubclass(ConfigError, ValueError)


def test_a_model_whose_dimension_is_not_the_declared_one_is_one_fatal(tmp_path, monkeypatch):
    """The engine compares the loaded model with the declared dimension and
    raises: the run stops before the collection is touched."""
    import data_orchestrator.core as core

    def build_embedder(settings):  # what TorchEmbedder does on a mismatch
        raise ValueError(
            f"embedding model {settings.model_name!r} produces 384-dimensional vectors, but "
            f"the configured dimension is {settings.model_dim} (check RAG_ENGINE_MODEL and "
            "RAG_ENGINE_DIM)")

    monkeypatch.setattr(core, "build_embedder", build_embedder)
    monkeypatch.setenv("DATA_ORCHESTRATOR_EMBEDDING_MODEL", "org/other")
    monkeypatch.setenv("DATA_ORCHESTRATOR_EMBEDDING_DIM", "768")
    log = []
    store = FakeStore(log)
    with pytest.raises(ValueError, match="produces 384-dimensional vectors"):
        run_ingest([_md_item(tmp_path)], collection="c", ledger=RecordingLedger(log), store=store)
    assert store.ensured == [] and log == []


def test_the_default_embedder_is_built_by_the_engine_from_the_settings(monkeypatch):
    import rag_engine.embedding as engine_embedding

    import data_orchestrator.core as core

    seen = []
    monkeypatch.setattr(engine_embedding, "make_embedder",
                        lambda settings, **kwargs: seen.append((settings, kwargs)) or "embedder")
    settings = engine_settings({})
    assert core.build_embedder(settings) == "embedder"
    assert seen == [(settings, {})]  # device left to the engine: the best available


def test_the_default_embedder_gets_the_model_and_cache_of_the_run_s_settings(monkeypatch):
    """Settings that differ from the defaults and from the environment: the
    model that is loaded is the run's, from the run's cache directory, on the
    device the engine picks (none is forced here)."""
    import rag_engine.embedding as engine_embedding

    import data_orchestrator.core as core

    built = []

    class Recorder:
        def __init__(self, spec, **kwargs):
            built.append((spec.name, spec.dim, spec.passage_prefix, kwargs))

    monkeypatch.setattr(engine_embedding, "TorchEmbedder", Recorder)
    monkeypatch.setenv("RAG_ENGINE_MODEL", "env/model")
    monkeypatch.setenv("RAG_ENGINE_MODEL_CACHE", "/env/cache")
    settings = Settings(model_name="org/run-model", model_dim=8, model_cache="/run/cache",
                        passage_prefix="p: ")
    assert isinstance(core.build_embedder(settings), Recorder)
    assert built == [("org/run-model", 8, "p: ", {"cache_dir": "/run/cache"})]


# ---------------------------------------------------------------------------
# Reserved payload keys: the engine refuses metadata that sets a key it writes
# itself. Sources that copy metadata they do not own drop such keys
# (split_reserved_keys) and report them; a hand-built item is not rewritten.
# ---------------------------------------------------------------------------
def test_split_reserved_keys_drops_and_names_the_engine_s_keys():
    metadata = {"title": "t", "filename": "x.pdf", "text": "stale", "page": 3,
                "chunk_number": 1, "ingestion_date": "2020", "doc_id": "other"}
    clean, dropped = split_reserved_keys(metadata, "doc-1")
    assert clean == {"title": "t"}
    assert sorted(dropped) == ["chunk_number", "doc_id", "filename", "ingestion_date",
                               "page", "text"]
    assert "filename" in metadata  # the input is not changed


def test_split_reserved_keys_keeps_a_doc_id_equal_to_the_document_s():
    clean, dropped = split_reserved_keys({"doc_id": "doc-1", "year": 2024}, "doc-1")
    assert clean == {"doc_id": "doc-1", "year": 2024} and dropped == ()
    assert split_reserved_keys({"doc_id": "doc-1"}) == ({}, ("doc_id",))


def test_split_reserved_keys_drops_the_interval_the_engine_derives():
    """``date_start`` / ``date_end`` are derived by the engine from ``date`` and
    ``date_precision`` and refused in metadata: a free-form ``extra`` holding
    them must not fail the document."""
    from rag_engine.schema import DERIVED_KEYS

    assert DERIVED_KEYS == ("date_start", "date_end")
    metadata = {"date": "2016-07-01", "date_precision": "month", "date_kind": "meeting",
                "date_start": "1999-01-01T00:00:00Z", "date_end": "garbage"}
    clean, dropped = split_reserved_keys(metadata, "doc-1")
    assert clean == {"date": "2016-07-01", "date_precision": "month", "date_kind": "meeting"}
    assert dropped == ("date_start", "date_end")


def test_a_derived_key_in_a_hand_built_payload_fails_that_document_only(tmp_path):
    log = []
    bad = _md_item(tmp_path, doc_id="bad", payload={"date": "2016-07-01", "date_end": "x"})
    stats = run_ingest([bad, _md_item(tmp_path, doc_id="ok")], collection="c",
                       ledger=RecordingLedger(log), store=FakeStore(log),
                       embedder=FakeEmbedder())
    assert (stats.docs_error, stats.docs_ingested) == (1, 1)
    assert stats.errors[0][1].startswith(
        "ValueError: metadata may not set derived payload key(s): date_end (document 'bad')")


def test_dropped_keys_are_counted_on_every_item_seen_and_never_sent(tmp_path):
    log = []

    class SkipSecond(RecordingLedger):
        def __contains__(self, doc_id):
            return doc_id == "d2"

    items = [_md_item(tmp_path, doc_id="d1"), _md_item(tmp_path, doc_id="d2"),
             _md_item(tmp_path, doc_id="d3")]
    items[0].dropped_keys = ("filename", "text")
    items[1].dropped_keys = ("filename",)
    store = FakeStore(log)
    stats = run_ingest(items, collection="c", ledger=SkipSecond(log), store=store,
                       embedder=FakeEmbedder())
    assert stats.payload_keys_dropped == {"filename": 2, "text": 1}
    assert stats.docs_ingested == 2 and stats.docs_error == 0
    assert stats.as_dict()["payload_keys_dropped"] == {"filename": 2, "text": 1}


@pytest.mark.parametrize("key", ["filename", "chunk_number", "ingestion_date", "text", "page"])
def test_a_reserved_key_in_a_hand_built_payload_fails_that_document_only(tmp_path, key):
    """The previous pipeline overwrote such a key without a word; the engine
    refuses it. Nothing is rewritten behind a caller that built the item."""
    log = []
    bad = _md_item(tmp_path, doc_id="bad", payload={"source_code": "us", key: "stale"})
    stats = run_ingest([bad, _md_item(tmp_path, doc_id="ok")], collection="c",
                       ledger=RecordingLedger(log), store=FakeStore(log),
                       embedder=FakeEmbedder())
    assert (stats.docs_error, stats.docs_ingested) == (1, 1)
    assert stats.errors[0][1] == (
        f"ValueError: metadata may not set reserved payload key(s): {key} (document 'bad')")


def test_a_payload_doc_id_other_than_the_item_s_fails_that_document(tmp_path):
    bad = _md_item(tmp_path, doc_id="bad", payload={"doc_id": "someone-else"})
    same = _md_item(tmp_path, doc_id="same", payload={"doc_id": "same"})
    store = FakeStore([])
    stats = run_ingest([bad, same], collection="c", store=store, embedder=FakeEmbedder())
    assert (stats.docs_error, stats.docs_ingested) == (1, 1)
    assert {p.payload["doc_id"] for p in store.client.points} == {"same"}
