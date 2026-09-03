"""Core ingestion engine.

Drives the eigenmind engine (ChunkNorris chunking -> E5 embeddings -> Qdrant)
for an arbitrary stream of documents coming from any source.

Responsibilities the core owns (so sources don't have to):
- chunking + embedding + idempotent upsert into a *configurable* collection,
- merging source metadata into every chunk payload,
- a resume ledger so re-runs skip already-ingested documents,
- progress + per-document error isolation,
- the Qdrant/vault write protocol: per document, every Qdrant point is
  upserted BEFORE the ledger record, so a crash can only ever leave the
  ledger under-claiming (healed by the next resume pass). Batching never
  spans documents.

A *source* only has to yield :class:`SourceItem` objects.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

from chunknorris.exceptions import TextNotFoundException
from eigenmind.config import BATCH_SIZE
from eigenmind.core.chunking import chunk_with_chunknorris
from eigenmind.core.embeddings import EmbeddingModel, detect_device
from eigenmind.vectordb.store import QdrantStore
from qdrant_client.models import PointStruct

from .routing import embedding_model_name

# CLI --ocr mode -> engine use_ocr argument. "auto" defers to the engine's
# availability check (None), so a machine without tesseract degrades to
# "never" instead of crashing; "always"/"never" are explicit demands.
OCR_TO_ENGINE = {"auto": None, "always": "always", "never": "never"}


# ---------------------------------------------------------------------------
# Source contract
# ---------------------------------------------------------------------------
@dataclass
class SourceItem:
    """One document to ingest.

    Attributes
    ----------
    doc_id:
        Stable, source-unique id. Used for the resume ledger and as the base
        of every chunk's point id, so re-ingesting overwrites in place.
    path:
        Local filesystem path to the document (PDF for now).
    payload:
        Extra metadata merged into every chunk's Qdrant payload
        (e.g. bank_code, doc_type, year). ``filename`` is added automatically.
    """

    doc_id: str
    path: Path
    payload: dict = field(default_factory=dict)


@dataclass
class IngestStats:
    docs_seen: int = 0
    docs_ingested: int = 0
    docs_skipped_resume: int = 0
    docs_empty: int = 0
    docs_error: int = 0
    chunks_written: int = 0
    errors: list[tuple[str, str]] = field(default_factory=list)
    # Per-source_code counters (docs_seen/docs_new/docs_failed), populated by
    # ``_bump`` on every terminal path of run_ingest's loop — the shared shape
    # the run-report's "sources" breakdown is built from (see cli._build_report).
    by_source: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["errors"] = self.errors[:50]  # cap
        return d


# ---------------------------------------------------------------------------
# Resume ledger
# ---------------------------------------------------------------------------
class Ledger:
    """A tiny JSON-lines ledger of ingested doc_ids.

    Persisted so a re-run skips work already done. Upsert is idempotent, so
    the ledger is an optimisation (avoid re-extract + re-embed), not a
    correctness requirement.
    """

    def __init__(self, path: Path):
        self.path = path
        self._done: dict[str, int] = {}
        if path.exists():
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        self._done[rec["doc_id"]] = rec.get("chunks", 0)
                    except (json.JSONDecodeError, KeyError):
                        continue

    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self._done

    def __len__(self) -> int:
        return len(self._done)

    def mark(self, doc_id: str, chunks: int, payload: dict | None = None) -> None:
        # ``payload`` is part of the shared ledger interface (the vault ledger
        # uses it for corpus/source_code); the file ledger ignores it.
        self._done[doc_id] = chunks
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"doc_id": doc_id, "chunks": chunks,
                                 "ts": int(time.time())}) + "\n")


# ---------------------------------------------------------------------------
# Per-source_code stats — feeds the run-report's "sources" breakdown
# ---------------------------------------------------------------------------
def _bump(stats: IngestStats, item: SourceItem, *, new: int = 0, failed: int = 0) -> None:
    code = (item.payload.get("source_code") or item.payload.get("bank_code")
            or item.payload.get("corpus", "unknown"))
    entry = stats.by_source.setdefault(code, {"docs_seen": 0, "docs_new": 0, "docs_failed": 0})
    entry["docs_seen"] += 1
    entry["docs_new"] += new
    entry["docs_failed"] += failed


# ---------------------------------------------------------------------------
# Point id — unique per (doc_id, page, chunk_index)
# ---------------------------------------------------------------------------
def _point_id(doc_id: str, page: int, chunk_index: int) -> int:
    key = f"{doc_id}::{page}::{chunk_index}"
    digest = hashlib.sha1(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) >> 1


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------
def ingest_item(
    item: SourceItem,
    collection: str,
    *,
    store: QdrantStore,
    embedder: EmbeddingModel,
    ocr: str = "auto",
    batch_size: int = BATCH_SIZE,
) -> int:
    """Chunk, embed and upsert one document. Returns chunks written.

    All of this document's points reach Qdrant before the caller records it
    in the ledger (write protocol) — batching never spans documents.
    """
    try:
        chunks = chunk_with_chunknorris(str(item.path), use_ocr=OCR_TO_ENGINE[ocr])
    except TextNotFoundException:
        # ChunkNorris raises rather than returning [] when a document (e.g. a
        # PDF with no text layer and OCR off) has no extractable text at all.
        # That is the "empty document" case, not a per-document error.
        return 0
    texts, id_pages, payload_pages = [], [], []
    for chunk in chunks:
        text = chunk.get_text()
        if not text.strip():
            continue
        texts.append(text)
        page = getattr(chunk, "start_page", None)
        # Point ids always resolve a page (0 for non-paginated formats like
        # markdown) so ids stay deterministic; the payload, below, keeps the
        # real None instead of masking it as page 0.
        id_pages.append(page if page is not None else 0)
        payload_pages.append(page)
    if not texts:
        return 0

    vectors = embedder.encode_passage(texts)

    # One ingestion_date per document: eigenmind's only date axis (see
    # eigenmind.vectordb.store.make_point / date_range_filter) reads this ISO
    # string from every point's payload, so all of a document's points must
    # agree on it rather than drift across the batch's wall-clock time.
    ingestion_date = datetime.datetime.now().isoformat()

    base_payload = {"doc_id": item.doc_id, **item.payload}
    points = []
    for i in range(len(texts)):
        payload = {
            **base_payload,
            "filename": item.path.name,
            # Payload key is "chunk_number" (not "chunk_index") to match
            # what the eigenmind engine ecosystem reads everywhere
            # (vectordb/store.py, pipelines/rag.py, graph/singular.py).
            "chunk_number": i,
            "ingestion_date": ingestion_date,
            "text": texts[i],
        }
        if payload_pages[i] is not None:
            payload["page"] = payload_pages[i]
        points.append(PointStruct(
            id=_point_id(item.doc_id, id_pages[i], i),
            vector=vectors[i].tolist(),
            payload=payload,
        ))

    for start in range(0, len(points), batch_size):
        store.client.upsert(collection_name=collection, points=points[start:start + batch_size])

    return len(points)


def run_ingest(
    items: Iterable[SourceItem],
    *,
    collection: str,
    ledger=None,
    ocr: str = "auto",
    batch_size: int = BATCH_SIZE,
    limit: Optional[int] = None,
    on_progress: Optional[Callable[[IngestStats, SourceItem, str], None]] = None,
    store: QdrantStore | None = None,
    embedder: EmbeddingModel | None = None,
) -> IngestStats:
    """Ingest a stream of items into ``collection``.

    Skips items already recorded in ``ledger``. Isolates per-document errors
    (including ledger write failures) so one bad document never aborts the
    run. ``limit`` caps the number of *newly* ingested documents. ``store``
    and ``embedder`` are injectable for tests; by default a QdrantStore and
    an EmbeddingModel (loaded once, on the best available device) are created
    and the embedder is released at the end.
    """
    stats = IngestStats()
    owns_embedder = embedder is None
    store = store or QdrantStore()
    embedder = embedder or EmbeddingModel(
        device=detect_device(), model_name=embedding_model_name()
    )
    try:
        # Inside the try/finally that releases an owned embedder: a raising
        # ensure_collection must not leak it.
        store.ensure_collection(collection, embedder.dim)
        for item in items:
            stats.docs_seen += 1

            if ledger is not None and item.doc_id in ledger:
                stats.docs_skipped_resume += 1
                _bump(stats, item)
                if on_progress:
                    on_progress(stats, item, "skip-resume")
                continue

            # The ledger mark is inside this per-document try (both the n > 0
            # and the n == 0 "empty doc" case) so a ledger write failure
            # (e.g. an FK violation for a doc_id unknown to the vault) is
            # isolated as a docs_error for this document rather than
            # aborting the whole run — matching the "one bad document never
            # aborts the run" contract for every path, not just n > 0.
            try:
                n = ingest_item(
                    item, collection,
                    store=store, embedder=embedder,
                    ocr=ocr, batch_size=batch_size,
                )
                if ledger is not None:
                    ledger.mark(item.doc_id, n, payload=item.payload)
            except Exception as exc:  # noqa: BLE001 — isolate one bad document
                stats.docs_error += 1
                stats.errors.append((str(item.path), f"{type(exc).__name__}: {exc}"))
                _bump(stats, item, failed=1)
                if on_progress:
                    on_progress(stats, item, "error")
                continue

            if n == 0:
                stats.docs_empty += 1
                _bump(stats, item)
                if on_progress:
                    on_progress(stats, item, "empty")
                continue

            stats.docs_ingested += 1
            stats.chunks_written += n
            _bump(stats, item, new=1)
            if on_progress:
                on_progress(stats, item, "ingested")

            if limit is not None and stats.docs_ingested >= limit:
                break
    finally:
        if owns_embedder:
            embedder.release()
    return stats
