"""bottom_up_corpus source connector.

Thin shim over the ``bottom_up_corpus`` project's RAG ingestion contract
(``bottom_up_corpus.rag.iter_items``). bottom_up_corpus is the **micro** layer of
the corpus family: it discovers SEC EDGAR company filings, downloads and
decomposes the complete submission, extracts clean text and renders each filing's
primary document to a human-readable, page-anchored PDF. That rendered PDF is the
artifact we ingest (it flows through the existing mvp-graph-rag PDF loader with no
change); cleaned text is the fallback when no PDF exists.

All of that corpus logic lives in the bottom_up_corpus project — this connector
only maps its items onto this repo's :class:`..core.SourceItem` and registers it
as a selectable source. Nothing is vendored here.

bottom_up_corpus: https://github.com/jeulinmarc/bottom_up_corpus
Its ``SourceItem`` already mirrors this repo's (``doc_id``, ``path``, ``payload``)
and its payload schema, citation format and family-weighting guidance are
documented in ``bottom_up_corpus/docs/INGESTION_RAG.md``. The payload carries at
least ``source="bottom_up_corpus"``, ``cik``, ``company``, ``doc_type``, ``year``
and ``url``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional, Sequence

from ..config import get_path
from ..core import SourceItem

# Name of the ``.env`` key holding this machine's corpus root. As with cb_corpus
# the path is never hard-coded: it lives in ``.env`` (see ``.env.example``) or is
# passed explicitly via the CLI ``--root`` flag. ``None`` is also acceptable —
# ``bottom_up_corpus.rag.iter_items`` has its own default root resolution.
ROOT_ENV_KEY = "BOTTOM_UP_CORPUS_ROOT"


def default_root() -> Optional[Path]:
    """Corpus root for this machine, read from ``BOTTOM_UP_CORPUS_ROOT`` in ``.env``.

    Returns ``None`` if it is not configured; bottom_up_corpus then falls back to
    its own default data dir (or the caller can pass an explicit ``--root``).
    """
    return get_path(ROOT_ENV_KEY)


def _load_iter_items():
    """Import ``bottom_up_corpus.rag.iter_items`` lazily with a helpful error.

    Kept out of module import so this connector (and its tests) can be imported
    without bottom_up_corpus installed; the dependency is only required to
    actually ingest.
    """
    try:
        from bottom_up_corpus.rag import iter_items as _iter_items
    except ImportError as exc:  # pragma: no cover — exercised via a stubbed module
        raise ImportError(
            "bottom_up_corpus is not importable. Install it from GitHub, e.g.\n"
            "    pip install 'bottom_up_corpus @ "
            "git+https://github.com/jeulinmarc/bottom_up_corpus'\n"
            "or `pip install -e .` a local checkout / add it to PYTHONPATH. "
            f"(original import error: {exc})"
        ) from exc
    return _iter_items


def iter_items(
    root: Optional[Path] = None,
    *,
    ciks: Optional[Sequence[str]] = None,
    doctypes: Optional[Sequence[str]] = None,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    prefer: str = "pdf",
) -> Iterator[SourceItem]:
    """Yield one :class:`SourceItem` per bottom_up_corpus filing.

    Parameters
    ----------
    root:
        Corpus root. Defaults to the ``BOTTOM_UP_CORPUS_ROOT`` value from
        ``.env``; if that is unset too, ``None`` is passed through so
        bottom_up_corpus uses its own default data dir.
    ciks:
        Optional allow-list of SEC CIK numbers. Accepts a sequence or a single
        comma-separated string (``"320193,789019"``).
    doctypes:
        Optional allow-list of filing-type / family codes (e.g. ``["A1"]`` for
        10-K). ``None`` means "all".
    year_min / year_max:
        Inclusive year bounds.
    prefer:
        ``"pdf"`` (default) yields the rendered, page-anchored PDF and falls back
        to cleaned text when no PDF exists; ``"text"`` prefers the cleaned text.
    """
    _iter_items = _load_iter_items()

    if root is None:
        root = default_root()
    root_arg = str(root) if root is not None else None

    if isinstance(ciks, str):
        ciks = [c.strip() for c in ciks.split(",") if c.strip()]

    for it in _iter_items(
        root=root_arg,
        ciks=list(ciks) if ciks else None,
        doctypes=list(doctypes) if doctypes else None,
        year_min=year_min,
        year_max=year_max,
        prefer=prefer,
    ):
        # bottom_up_corpus's SourceItem already exposes (doc_id, path, payload)
        # with the same meaning as ours; re-wrap into this repo's dataclass so the
        # core engine receives exactly the type it expects.
        yield SourceItem(
            doc_id=it.doc_id,
            path=Path(it.path),
            payload=dict(it.payload),
        )
