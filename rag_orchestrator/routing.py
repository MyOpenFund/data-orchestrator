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

EMBEDDING_MODEL_ENV = "RAGO_EMBEDDING_MODEL"
DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-base"

# Policy tag recorded in rag_ingestions.embedding_version: bump when the
# embedding *policy* changes (prefixes, chunking params), not the model.
EMBEDDING_VERSION = "e5-prefixes-v1"

# Short tags for collection names. Unknown models fall back to a sanitized
# last path segment. The integration-test tiny model (swapped in for speed via
# RAGO_EMBEDDING_MODEL, see tests/integration/conftest.py) is pinned to the
# same "e5b" tag as its production stand-in so collection names stay stable
# between prod and CI runs of the same corpus.
MODEL_TAGS = {
    "intfloat/multilingual-e5-base": "e5b",
    "sentence-transformers/paraphrase-albert-small-v2": "e5b",
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


ROUTING: dict[str, CorpusRoute] = {
    "central-bank": CorpusRoute(corpus="central-bank", root_env_key="CB_CORPUS_ROOT"),
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
