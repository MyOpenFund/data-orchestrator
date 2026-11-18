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

from rag_engine.chunking import SUPPORTED_EXTENSIONS

from .config import get_path

EMBEDDING_MODEL_ENV = "DATA_ORCHESTRATOR_EMBEDDING_MODEL"
# Vector dimension of the model above; required for any model but the default
# (core.engine_settings). The engine's own names are read as fallbacks.
EMBEDDING_DIM_ENV = "DATA_ORCHESTRATOR_EMBEDDING_DIM"
ENGINE_MODEL_ENV = "RAG_ENGINE_MODEL"
ENGINE_DIM_ENV = "RAG_ENGINE_DIM"
DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-base"

# Policy tag recorded in rag_ingestions.embedding_version: bump when the
# embedding *policy* changes (prefixes, chunking params), not the model.
EMBEDDING_VERSION = "e5-prefixes-v1"

# Short tags for collection names. Unknown models fall back to a sanitized
# last path segment (e.g. paraphrase-albert-small-v2 -> paraphrase-albert-small-v2).
MODEL_TAGS = {
    "intfloat/multilingual-e5-base": "e5b",
}


def embedding_model_name(environ=None) -> str:
    """The sentence-transformers model to embed with.

    ``DATA_ORCHESTRATOR_EMBEDDING_MODEL``, else the engine's own
    ``RAG_ENGINE_MODEL``, else the default. One resolution for the model that
    embeds (``core.engine_settings``), the collection name and the
    ``rag_ingestions`` rows, so they cannot disagree.
    """
    environ = os.environ if environ is None else environ
    for key in (EMBEDDING_MODEL_ENV, ENGINE_MODEL_ENV):
        value = (environ.get(key) or "").strip()
        if value:
            return value
    return DEFAULT_EMBEDDING_MODEL


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


# The mime type of each file extension the engine reads. The extensions are the
# engine's (``rag_engine.chunking.SUPPORTED_EXTENSIONS``: ``chunk_file``
# dispatches on the file extension); the mime mapping is this repo's, since
# the vault describes documents by mime type. An extension the engine reads
# and this table does not name is not offered to it;
# tests/test_routing.py asserts the two stay in sync.
_EXTENSION_MIME: dict[str, str] = {
    ".pdf": "application/pdf",
    ".md": "text/markdown",
    ".txt": "text/plain",
    ".csv": "text/csv",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}


def _mime_extensions() -> dict[str, tuple[str, ...]]:
    """{mime: extensions} for the extensions the engine reads, in this table's order."""
    table: dict[str, tuple[str, ...]] = {}
    for ext, mime in _EXTENSION_MIME.items():
        if ext in SUPPORTED_EXTENSIONS:
            table[mime] = table.get(mime, ()) + (ext,)
    return table


MIME_EXTENSIONS: dict[str, tuple[str, ...]] = _mime_extensions()
SUPPORTED_MIME_ENV = "DATA_ORCHESTRATOR_SUPPORTED_MIME_TYPES"
# Default = everything the engine reads (decided 2026-11-10). text/html is
# deliberately absent: HTML is converted to .md or PDF before the engine.
DEFAULT_SUPPORTED_MIME_TYPES: tuple[str, ...] = tuple(MIME_EXTENSIONS)


def supported_mime_types() -> tuple[str, ...]:
    """The mime types the vault selection offers to the engine.

    ``DATA_ORCHESTRATOR_SUPPORTED_MIME_TYPES`` (comma list) narrows the
    default; unset or blank -> default. Entries are trimmed, lower-cased and
    de-duplicated (order kept). Raises ``ValueError`` (caller: exit 2) on a
    list with no entry or with a type the engine cannot read.
    """
    raw = os.environ.get(SUPPORTED_MIME_ENV)
    if raw is None or not raw.strip():
        return DEFAULT_SUPPORTED_MIME_TYPES
    mimes: list[str] = []
    for part in raw.split(","):
        mime = part.strip().lower()
        if mime and mime not in mimes:
            mimes.append(mime)
    if not mimes:
        raise ValueError(f"{SUPPORTED_MIME_ENV}={raw!r} lists no mime type")
    return validate_mimes(mimes, origin=SUPPORTED_MIME_ENV)


def validate_mimes(mimes, *, origin: str) -> tuple[str, ...]:
    """``mimes`` as a tuple, checked against the engine's table.

    Entries must already be normalised (trimmed, lower-cased, no parameters):
    anything that is not a key of ``MIME_EXTENSIONS`` is refused. ``origin``
    names where the list came from, for the message. Raises ``ValueError`` on
    an empty list or on an entry the engine cannot read.
    """
    if isinstance(mimes, str):
        raise ValueError(f"{origin} must be a sequence of mime types, not the string {mimes!r}")
    mimes = tuple(mimes)
    if not mimes:
        raise ValueError(f"{origin} lists no mime type")
    not_str = [m for m in mimes if not isinstance(m, str)]
    if not_str:
        raise ValueError(f"{origin}: every entry must be a mime-type string, got {not_str!r}")
    unknown = [str(m) for m in mimes if m not in MIME_EXTENSIONS]
    if unknown:
        raise ValueError(
            f"{origin}: the engine cannot read {', '.join(unknown)} "
            f"(known: {', '.join(MIME_EXTENSIONS)})"
        )
    return mimes


def supported_extensions(mimes) -> tuple[str, ...]:
    """File extensions of ``mimes`` (the vault selection requires one on every row)."""
    exts: list[str] = []
    for mime in mimes:
        for ext in MIME_EXTENSIONS[mime]:
            if ext not in exts:
                exts.append(ext)
    return tuple(exts)


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
        root_env_key="COMPANY_CORPUS_ROOT",
        # company_corpus (the "company" corpus, SEC EDGAR micro layer) has
        # no manifests feeding the vault yet, so its vault local_path
        # convention isn't settled — this connector still runs disk-only
        # (company source, --no-vault). Revisit local_path_strip
        # once company documents/rag_ingestions rows exist to derive it from.
        local_path_strip="",
    ),
}


def corpus_root(corpus: str) -> Path:
    """Local data root for a corpus (joined with the vault's local_path).

    Reads the root through ``config.get_path``, which runs
    ``config.load_dotenv()`` first: a library caller whose
    ``DATA_ORCHESTRATOR_DOTENV`` names a missing file gets
    ``FileNotFoundError`` here.
    """
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
