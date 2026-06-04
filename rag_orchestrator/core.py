"""Core ingestion engine.

Drives the existing mvp-graph-rag pipeline (``load_pdf`` -> chunk -> ``embed``
-> Qdrant) for an arbitrary stream of documents coming from any source.

Responsibilities the core owns (so sources don't have to):
- importing the mvp-graph-rag ``src`` modules,
- chunking + embedding + idempotent upsert into a *configurable* collection,
- merging source metadata into every chunk payload,
- a resume ledger so re-runs skip already-ingested documents,
- progress + per-document error isolation.

A *source* only has to yield :class:`SourceItem` objects.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional

# --- make the mvp-graph-rag ``src`` package importable ----------------------
# rag_orchestrator is its own repo, *next to* mvp-graph-rag. The mvp pipeline
# modules use flat imports (``from embed_text import embed``), so the
# mvp-graph-rag ``src`` directory must be on sys.path.
#
# Resolution order:
#   1. ``MVP_GRAPH_RAG_SRC`` env var (explicit override), else
#   2. walk up the ancestors of this file looking for a sibling
#      ``mvp-graph-rag/src`` (works whatever the nesting depth / repo name).


def _resolve_mvp_src() -> Path:
    env = os.environ.get("MVP_GRAPH_RAG_SRC")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    for ancestor in here.parents:
        candidate = ancestor / "mvp-graph-rag" / "src"
        if candidate.is_dir():
            return candidate
    # Fall back to the conventional sibling location for a clear error message.
    return here.parents[2] / "mvp-graph-rag" / "src"


_MVP_SRC = _resolve_mvp_src()
if not _MVP_SRC.is_dir():
    raise RuntimeError(
        f"mvp-graph-rag 'src' not found (looked near {_MVP_SRC}). "
        "Set the MVP_GRAPH_RAG_SRC environment variable to its path, e.g. "
        r"set MVP_GRAPH_RAG_SRC=C:\path\to\mvp-graph-rag\src"
    )
if str(_MVP_SRC) not in sys.path:
    sys.path.insert(0, str(_MVP_SRC))

# Imported from mvp-graph-rag/src (resolved via the sys.path insert above).
from load_pdf import load_and_chunk  # noqa: E402
from embed_text import embed, EMBEDDING_DIM  # noqa: E402
from store_chunks import (  # noqa: E402
    get_client,
    ensure_collection,
    BATCH_SIZE,
)
from qdrant_client.models import PointStruct  # noqa: E402

import hashlib


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

    def mark(self, doc_id: str, chunks: int) -> None:
        self._done[doc_id] = chunks
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"doc_id": doc_id, "chunks": chunks,
                                 "ts": int(time.time())}) + "\n")


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
    ocr: str = "auto",
    batch_size: int = BATCH_SIZE,
) -> int:
    """Load, chunk, embed and upsert one document. Returns chunks written."""
    chunks = load_and_chunk(item.path, ocr=ocr)
    if not chunks:
        return 0

    texts = [c["text"] for c in chunks]
    vectors = embed(texts)
    assert vectors.shape[1] == EMBEDDING_DIM, (
        f"Embedding dim mismatch: got {vectors.shape[1]}, expected {EMBEDDING_DIM}"
    )

    base_payload = {"doc_id": item.doc_id, **item.payload}
    client = get_client()
    ensure_collection(client, collection)

    points = [
        PointStruct(
            id=_point_id(item.doc_id, c["page"], c["chunk_index"]),
            vector=vectors[i].tolist(),
            payload={
                **base_payload,
                "filename": c["filename"],
                "page": c["page"],
                "chunk_index": c["chunk_index"],
                "text": c["text"],
            },
        )
        for i, c in enumerate(chunks)
    ]

    for start in range(0, len(points), batch_size):
        client.upsert(collection_name=collection, points=points[start:start + batch_size])

    return len(points)


def run_ingest(
    items: Iterable[SourceItem],
    *,
    collection: str,
    ledger: Optional[Ledger] = None,
    ocr: str = "auto",
    batch_size: int = BATCH_SIZE,
    limit: Optional[int] = None,
    on_progress: Optional[Callable[[IngestStats, SourceItem, str], None]] = None,
) -> IngestStats:
    """Ingest a stream of items into ``collection``.

    Skips items already recorded in ``ledger``. Isolates per-document errors so
    one bad PDF never aborts the run. ``limit`` caps the number of *newly*
    ingested documents (handy for test subsets).
    """
    stats = IngestStats()

    for item in items:
        stats.docs_seen += 1

        if ledger is not None and item.doc_id in ledger:
            stats.docs_skipped_resume += 1
            if on_progress:
                on_progress(stats, item, "skip-resume")
            continue

        try:
            n = ingest_item(item, collection, ocr=ocr, batch_size=batch_size)
        except Exception as exc:  # noqa: BLE001 — isolate one bad document
            stats.docs_error += 1
            stats.errors.append((str(item.path), f"{type(exc).__name__}: {exc}"))
            if on_progress:
                on_progress(stats, item, "error")
            continue

        if n == 0:
            stats.docs_empty += 1
            if ledger is not None:
                ledger.mark(item.doc_id, 0)  # don't retry empty docs forever
            if on_progress:
                on_progress(stats, item, "empty")
            continue

        stats.docs_ingested += 1
        stats.chunks_written += n
        if ledger is not None:
            ledger.mark(item.doc_id, n)
        if on_progress:
            on_progress(stats, item, "ingested")

        if limit is not None and stats.docs_ingested >= limit:
            break

    return stats
