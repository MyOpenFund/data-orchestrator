"""rag_orchestrator.

Ingestion orchestrator that feeds the **mvp-graph-rag** vector database from a
set of fixed data sources.

Direction of dependency is one-way: this package *uses* mvp-graph-rag (it drives
its ``load_pdf`` -> chunk -> ``embed`` -> Qdrant pipeline). mvp-graph-rag never
imports this package and keeps working perfectly on its own — the orchestrator
is purely an add-on for bulk-ingesting many documents from known sources.

Layout
------
- ``core``          reusable engine: walk -> load -> chunk -> embed -> upsert + ledger
- ``sources/``      one module per data source (``cb_corpus`` is source #1)
- ``cli``           console entrypoint (``rag-orchestrator``)

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
