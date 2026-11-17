"""Core ingestion loop.

Drives the rag-engine library (ChunkNorris chunking -> E5 embeddings -> Qdrant)
for an arbitrary stream of documents coming from any source.

The engine owns what a point is: its id, its payload, the collection checks
and the batched writes (``rag_engine.ingest.ingest_document``,
``rag_engine.store.QdrantStore``). The core owns the run around it, so sources
don't have to:
- a resume ledger so re-runs skip already-ingested documents,
- progress + per-document error isolation,
- the Qdrant/vault write protocol: per document, every Qdrant point is
  written BEFORE the ledger record, so a crash can only ever leave the
  ledger under-claiming (healed by the next resume pass). Batching never
  spans documents,
- the settings handed to the engine (:func:`engine_settings`).

A *source* only has to yield :class:`SourceItem` objects.

Importing this module loads no embedding backend and no document parser, and
reads no environment variable: the engine imports them on first use and reads
its settings only when asked (``engine_settings``, called by ``run_ingest`` or
by the CLI after it loaded this repo's ``.env``). ``build_store`` and
``build_embedder`` are the two factories of a run's defaults; a caller (or a
test) injects its own ``store`` / ``embedder`` instead.
"""
from __future__ import annotations

import dataclasses
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional

from rag_engine.config import ConfigError, Settings
from rag_engine.ingest import ingest_document
from rag_engine.schema import RESERVED_KEYS
from rag_engine.store import CollectionMismatchError, QdrantAuthError, QdrantStore

from .routing import (
    DEFAULT_EMBEDDING_MODEL, EMBEDDING_DIM_ENV, ENGINE_DIM_ENV, embedding_model_name,
)

# ---------------------------------------------------------------------------
# Engine settings and the defaults of a run
# ---------------------------------------------------------------------------
def engine_settings(environ: Mapping[str, str] | None = None) -> Settings:
    """The engine's settings for a run, read from ``environ`` (default: the
    process environment, at call time).

    Everything is the engine's own reading (``Settings.from_env``: ``QDRANT_*``,
    ``RAG_ENGINE_*``) except the embedding model, which this repo also names
    collections and ``rag_ingestions`` rows after:

    - model: ``routing.embedding_model_name`` (``DATA_ORCHESTRATOR_EMBEDDING_MODEL``,
      else ``RAG_ENGINE_MODEL``, else the default), so the model that embeds is
      always the one the collection name and the ledger say;
    - dimension: ``DATA_ORCHESTRATOR_EMBEDDING_DIM``, else ``RAG_ENGINE_DIM``,
      else the default model's own (768). Another model has no known
      dimension: one of the two variables is then required.

    Raises ``rag_engine.config.ConfigError`` (a ``ValueError``) naming the
    variable on an invalid value. A dimension that is not the loaded model's
    is refused later, by the engine, when the model is loaded.
    """
    env = os.environ if environ is None else environ
    settings = Settings.from_env(env)
    model = embedding_model_name(env)
    raw_dim = (env.get(EMBEDDING_DIM_ENV) or "").strip()
    if raw_dim:
        if not (raw_dim.isascii() and raw_dim.isdigit() and int(raw_dim) >= 1):
            raise ConfigError(
                f"{EMBEDDING_DIM_ENV}={env[EMBEDDING_DIM_ENV]!r} is not a vector "
                "dimension (expected an integer >= 1)")
        dim = int(raw_dim)
    elif (env.get(ENGINE_DIM_ENV) or "").strip():
        dim = settings.model_dim
    elif model == DEFAULT_EMBEDDING_MODEL:
        dim = Settings.model_dim
    else:
        raise ConfigError(
            f"embedding model {model!r}: its vector dimension is not known; set "
            f"{EMBEDDING_DIM_ENV} (or {ENGINE_DIM_ENV}) to the dimension of the model")
    return dataclasses.replace(settings, model_name=model, model_dim=dim)


def build_store(settings: Settings) -> QdrantStore:
    """The Qdrant store of a run: host, port, key, TLS and timeout of ``settings``."""
    return QdrantStore(settings=settings)


def build_embedder(settings: Settings):
    """The embedder of a run, loaded once on the best available device.

    Imports the backend (PyTorch for ``RAG_ENGINE_BACKEND=torch``, the default)
    on first use. The engine refuses a model whose dimension is not
    ``settings.model_dim`` (``ValueError``).
    """
    from rag_engine.embedding import make_embedder

    return make_embedder(settings)


def _check_batch_size(batch_size: int | None) -> None:
    """``None`` leaves the store's batch size (``RAG_ENGINE_BATCH_SIZE``); an
    explicit value must be an integer >= 1. Checked once before a run: inside
    the engine a bad value would fail every document instead."""
    if batch_size is None:
        return
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")


def split_reserved_keys(metadata: Mapping, doc_id: str | None = None) -> tuple[dict, tuple[str, ...]]:
    """``metadata`` without the payload keys the engine writes itself, and the
    names of the keys left out.

    The engine refuses metadata that sets ``filename``, ``chunk_number``,
    ``ingestion_date``, ``text`` or ``page``, or a ``doc_id`` other than the
    document's (``rag_engine.schema.RESERVED_KEYS``). A source that copies
    free-form metadata it does not own (the vault's ``extra``, another
    package's payload) passes it through here, so a stray key is dropped and
    counted (``SourceItem.dropped_keys`` -> the run report) instead of failing
    the document. ``doc_id=None`` drops any ``doc_id``; a ``doc_id`` equal to
    the document's is kept.
    """
    dropped = tuple(
        key for key in RESERVED_KEYS
        if key in metadata and not (key == "doc_id" and doc_id is not None and metadata[key] == doc_id)
    )
    return {k: v for k, v in metadata.items() if k not in dropped}, dropped


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
        (e.g. bank_code, doc_type, year). The engine adds ``filename``,
        ``chunk_number``, ``ingestion_date``, ``text`` and ``page`` itself and
        refuses a payload that sets one of them (see ``split_reserved_keys``).
    dropped_keys:
        Reserved keys the source left out of ``payload`` (``split_reserved_keys``).
        Counted by the run (``IngestStats.payload_keys_dropped``); never sent
        to the engine.
    """

    doc_id: str
    path: Path
    payload: dict = field(default_factory=dict)
    dropped_keys: tuple = ()


@dataclass
class IngestStats:
    docs_seen: int = 0
    docs_ingested: int = 0
    docs_skipped_resume: int = 0
    docs_empty: int = 0
    docs_error: int = 0
    # Documents whose payload says metadata_source == "path": the source could
    # not enrich them from its manifest (year-only dates, no title/sha256).
    # Counted on every item seen, resumed or not — it describes the source's
    # metadata quality, not the ingestion outcome, and never drives exit codes.
    docs_path_metadata: int = 0
    # Reserved payload keys the sources dropped from free-form metadata
    # (SourceItem.dropped_keys), per key name. Like docs_path_metadata it is
    # counted on every item seen and never drives exit codes.
    payload_keys_dropped: dict = field(default_factory=dict)
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
def source_label(payload_or_code, corpus: str | None = None) -> str:
    """The ``sources[]`` key a document is counted under — the ONE rule shared
    by :func:`_bump`, ``sources.vault.count_unsupported`` and the vault
    ``--count-only`` breakdown, so they cannot drift apart.

    ``payload_or_code`` is a payload dict (its ``source_code``, else its
    ``bank_code``, else its ``corpus``) or a bare source code (a SQL
    ``GROUP BY d.source_code`` value). An empty/NULL code falls back to the
    payload's ``corpus``, then to ``corpus``, then to ``"unknown"`` — so a
    vault document with a NULL source_code is filed under its corpus name
    whichever way it was counted.
    """
    if isinstance(payload_or_code, dict):
        code = payload_or_code.get("source_code") or payload_or_code.get("bank_code")
        corpus = payload_or_code.get("corpus") or corpus
    else:
        code = payload_or_code
    return code or corpus or "unknown"


def _bump(stats: IngestStats, item: SourceItem, *, new: int = 0, failed: int = 0) -> None:
    code = source_label(item.payload)
    entry = stats.by_source.setdefault(code, {"docs_seen": 0, "docs_new": 0, "docs_failed": 0})
    entry["docs_seen"] += 1
    entry["docs_new"] += new
    entry["docs_failed"] += failed


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------
def ingest_item(
    item: SourceItem,
    collection: str,
    *,
    store: QdrantStore,
    embedder: object,
    ocr: str = "auto",
    ocr_languages: str = "eng",
    batch_size: int | None = None,
) -> int:
    """Chunk, embed and write one document. Returns chunks written.

    A call of ``rag_engine.ingest.ingest_document``: the engine builds the
    points (ids, payloads) and writes them through ``store``. All of this
    document's points reach Qdrant before the caller records it in the ledger
    (write protocol) — batching never spans documents. A document with no
    extractable text (an empty file, a PDF without a text layer read without
    OCR) returns 0: the "empty document" case, not an error.

    ``batch_size=None`` leaves the store's own (``RAG_ENGINE_BATCH_SIZE``).
    Errors are the engine's, raised unchanged.
    """
    result = ingest_document(
        item.path, item.doc_id, item.payload,
        collection=collection, store=store, embedder=embedder,
        ocr=ocr, ocr_languages=ocr_languages, batch_size=batch_size,
    )
    return result.chunks


def run_ingest(
    items: Iterable[SourceItem],
    *,
    collection: str,
    ledger=None,
    ocr: str = "auto",
    batch_size: int | None = None,
    limit: Optional[int] = None,
    on_progress: Optional[Callable[[IngestStats, SourceItem, str], None]] = None,
    store: QdrantStore | None = None,
    embedder: object | None = None,
    on_target: Optional[Callable[[object, object], None]] = None,
    settings: Settings | None = None,
) -> IngestStats:
    """Ingest a stream of items into ``collection``.

    Skips items already recorded in ``ledger``. Isolates per-document errors
    (including ledger write failures) so one bad document never aborts the
    run. ``limit`` caps the number of *newly* ingested documents.

    ``settings`` are the engine's (default: :func:`engine_settings`, read
    here, before any work). ``store`` and ``embedder`` are injectable; by
    default :func:`build_store` and :func:`build_embedder` make them from the
    settings (the embedder loaded once, on the best available device) and the
    embedder is released at the end.

    Fatal, before any document: an invalid setting or ``batch_size``, a
    collection that is not a single unnamed cosine vector of the model's
    dimension (``CollectionMismatchError``), a refused API key
    (``QdrantAuthError``). A refused key in the middle of a run (HTTP 401/403
    on a write) is fatal too, not one failure per remaining document: the
    documents recorded so far stay recorded, the one being written is not
    (the next run writes it again, over the same point ids).

    ``on_target(host, port)`` is called once, right after the store is
    constructed and before any collection work, with the store's ``host`` and
    ``port`` attributes (``None`` when it has none): the Qdrant the run really
    writes to.
    """
    stats = IngestStats()
    _check_batch_size(batch_size)
    if settings is None:
        settings = engine_settings()
    owns_embedder = embedder is None
    if store is None:
        store = build_store(settings)
    if on_target is not None:
        on_target(getattr(store, "host", None), getattr(store, "port", None))
    if embedder is None:
        embedder = build_embedder(settings)
    try:
        # Inside the try/finally that releases an owned embedder: a raising
        # ensure_collection must not leak it.
        try:
            store.ensure_collection(collection, embedder.dim)
        except CollectionMismatchError as exc:
            raise CollectionMismatchError(f"{exc}: use another --collection") from exc
        for item in items:
            stats.docs_seen += 1

            if item.payload.get("metadata_source") == "path":
                stats.docs_path_metadata += 1
            for key in item.dropped_keys:
                stats.payload_keys_dropped[key] = stats.payload_keys_dropped.get(key, 0) + 1

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
                    ocr=ocr, ocr_languages=settings.ocr_languages,
                    batch_size=batch_size,
                )
                if ledger is not None:
                    ledger.mark(item.doc_id, n, payload=item.payload)
            except QdrantAuthError as exc:
                # Not this document's fault, and every later one would fail
                # the same way: one fatal instead of N per-document errors.
                fatal = QdrantAuthError(
                    f"{exc} (while writing document {item.doc_id!r}, after "
                    f"{stats.docs_ingested} document(s) ingested in this run)")
                fatal.points_written = getattr(exc, "points_written", 0)
                raise fatal from exc
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
