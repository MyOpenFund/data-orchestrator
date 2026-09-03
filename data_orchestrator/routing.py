"""Per-corpus routing rules and embedding/collection naming.

The orchestrator's corpus knowledge is deliberately tiny and declarative: which
env key holds the local corpus root, and how collections are named. Everything
else about a corpus (taxonomy, artifacts) travels through its manifest into the
vault. Adding a corpus = adding one CorpusRoute entry, never new connector code.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from .config import get_path

EMBEDDING_MODEL_ENV = "DATA_ORCHESTRATOR_EMBEDDING_MODEL"
DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-base"

# Policy tag recorded in rag_ingestions.embedding_version: bump when the
# embedding *policy* changes (prefixes, chunking params), not the model.
EMBEDDING_VERSION = "e5-prefixes-v1"

# Short tags for collection names. Unknown models fall back to a sanitized
# last path segment (e.g. paraphrase-albert-small-v2 -> paraphrase-albert-small-v2).
MODEL_TAGS = {
    "intfloat/multilingual-e5-base": "e5b",
}


def embedding_model_name() -> str:
    """The sentence-transformers model to embed with (env-overridable for CI)."""
    return os.environ.get(EMBEDDING_MODEL_ENV) or DEFAULT_EMBEDDING_MODEL


def model_tag(model_name: str) -> str:
    """Short collection-name tag for a model."""
    tag = MODEL_TAGS.get(model_name)
    if tag:
        return tag
    last = model_name.rsplit("/", 1)[-1].lower()
    return re.sub(r"[^a-z0-9]+", "-", last).strip("-")


def collection_name(corpus: str, model_name: str | None = None, version: int = 1) -> str:
    """Headless collection naming owned by the orchestrator: {corpus}-{tag}-v{n}."""
    return f"{corpus}-{model_tag(model_name or embedding_model_name())}-v{version}"


@dataclass(frozen=True)
class CorpusRoute:
    """Declarative routing for one corpus."""

    corpus: str
    root_env_key: str  # .env key holding this machine's corpus data root
    # Leading prefix stripped from the vault's local_path before joining with
    # corpus_root(). Only ever a leading-prefix strip (never a substring
    # removal anywhere in the path) — see resolve_local_path().
    local_path_strip: str = ""


ROUTING: dict[str, CorpusRoute] = {
    "central-bank": CorpusRoute(
        corpus="central-bank",
        root_env_key="CB_CORPUS_ROOT",
        # cb_corpus manifest rows store local_path relative to the cb_corpus
        # REPO root (e.g. "data/raw/us/C1/2010/<doc_id>.pdf"), but
        # CB_CORPUS_ROOT is documented (.env.example, sources/cb_corpus.py)
        # as the folder that already CONTAINS raw/ — i.e. the data dir
        # itself. Without stripping "data/" first, root / local_path would
        # double-nest into <CB_CORPUS_ROOT>/data/raw/... and find nothing.
        local_path_strip="data/",
    ),
    "company": CorpusRoute(
        corpus="company",
        root_env_key="BOTTOM_UP_CORPUS_ROOT",
        # bottom_up_corpus (the "company" corpus, SEC EDGAR micro layer) has
        # no manifests feeding the vault yet, so its vault local_path
        # convention isn't settled — this connector still runs disk-only
        # (bottom_up_corpus source, --no-vault). Revisit local_path_strip
        # once company documents/rag_ingestions rows exist to derive it from.
        local_path_strip="",
    ),
}


def corpus_root(corpus: str) -> Path:
    """Local data root for a corpus (joined with the vault's local_path)."""
    route = ROUTING[corpus]  # KeyError on unknown corpus is the right signal
    root = get_path(route.root_env_key)
    if root is None:
        raise RuntimeError(
            f"{route.root_env_key} is not set (.env or environment) — required "
            f"to resolve local files for corpus '{corpus}'"
        )
    return root


def resolve_local_path(corpus: str, local_path: str) -> Path:
    """Join a vault ``local_path`` with its corpus root.

    Strips the route's ``local_path_strip`` prefix first, when present, to
    reconcile manifest-relative local_paths with a root that already points
    past that prefix (see ``ROUTING["central-bank"]`` for the concrete case).
    The strip is a leading-prefix match only — a "data/" occurring anywhere
    else in the path is left untouched.
    """
    root = corpus_root(corpus)
    route = ROUTING[corpus]
    strip = route.local_path_strip
    remainder = local_path[len(strip):] if strip and local_path.startswith(strip) else local_path
    return root / remainder
