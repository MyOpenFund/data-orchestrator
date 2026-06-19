"""Test bootstrap: make ``rag_orchestrator`` importable fully offline.

``rag_orchestrator.core`` imports the mvp-graph-rag flat scripts (``load_pdf``,
``embed_text``, ``store_chunks``) and ``qdrant_client`` at module load — none of
which are needed to exercise the *source connectors*. Install lightweight
stand-ins (and point the mvp-src resolver at an existing dir) before anything
imports the package, so the connector tests need no heavy deps, no real
mvp-graph-rag checkout and no Qdrant.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

# Ensure the repo root is importable (``import rag_orchestrator``).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _stub(name: str, **attrs) -> None:
    if name in sys.modules:
        return
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    sys.modules[name] = mod


# core._resolve_mvp_src() raises unless the resolved dir exists; give it one.
os.environ.setdefault("MVP_GRAPH_RAG_SRC", str(Path(__file__).resolve().parent))

# mvp-graph-rag flat modules (imported by core at top level).
_stub("load_pdf", load_and_chunk=lambda *a, **k: [])
_stub("embed_text", embed=lambda texts: None, EMBEDDING_DIM=384)
_stub(
    "store_chunks",
    get_client=lambda: None,
    ensure_collection=lambda *a, **k: None,
    BATCH_SIZE=64,
)

# qdrant_client + qdrant_client.models.PointStruct (core does a `from ... import`).
if "qdrant_client" not in sys.modules:
    _qc = types.ModuleType("qdrant_client")
    _models = types.ModuleType("qdrant_client.models")

    class PointStruct:  # minimal stand-in
        def __init__(self, **kw):
            self.__dict__.update(kw)

    _models.PointStruct = PointStruct
    _qc.models = _models
    sys.modules["qdrant_client"] = _qc
    sys.modules["qdrant_client.models"] = _models
