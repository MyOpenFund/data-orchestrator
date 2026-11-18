"""data_orchestrator.

Ingestion orchestrator: a policy layer that moves documents selected by the
vault (Postgres) through the **rag-engine** library (chunking, embedding,
Qdrant write) into Qdrant, the derived projection.

Direction of dependency is one-way: this package *uses* rag-engine (it calls
``rag_engine.ingest.ingest_document`` once per document, with a
``rag_engine.store.QdrantStore`` and an embedder built once per run).
rag-engine never imports this package — the orchestrator is the policy layer
that bulk-ingests the vault's selection (or, as a fallback, a fixed on-disk
corpus) into Qdrant.

Layout
------
- ``core``          the run loop around the engine: settings, resume ledger, isolation
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
