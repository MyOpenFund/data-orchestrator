"""rag_orchestrator.

Ingestion orchestrator: a policy layer that moves documents selected by the
vault (Postgres) through the **eigenmind** engine (chunking, embedding,
Qdrant upsert) into Qdrant, the derived projection.

Direction of dependency is one-way: this package *uses* eigenmind (it drives
its ``chunk_with_chunknorris`` -> ``EmbeddingModel`` -> ``QdrantStore`` pipeline).
eigenmind never imports this package and keeps working perfectly on its own —
the orchestrator is purely an add-on for bulk-ingesting the vault's selection
(or, as a fallback, a fixed on-disk corpus) into Qdrant.

Layout
------
- ``core``          reusable engine: chunk -> embed -> upsert + resume ledger
- ``routing``        per-corpus root/collection-naming rules
- ``vault``          Postgres connection + the rag_ingestions-backed ledger
- ``probe``          facts probe (has_text_layer / page_count) for OCR policy
- ``sources/``      one module per data source (``vault`` and ``cb_corpus``)
- ``cli``           console entrypoint (``data-orchestrator``)

Each source yields :class:`core.SourceItem` objects (a file path + a metadata
payload). The core handles everything else.
"""
from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["SourceItem", "run_ingest", "ingest_item", "Ledger", "IngestStats"]


def __getattr__(name: str):
    # Lazy re-export from .core so that importing the package name alone does
    # not pull in heavy deps (torch / sentence-transformers) until actually used.
    if name in __all__:
        from . import core

        return getattr(core, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
