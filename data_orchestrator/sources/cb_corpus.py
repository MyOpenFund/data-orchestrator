"""cb_corpus source connector.

Reads the official central-bank corpus produced by the ``cb_corpus`` project
and yields :class:`..core.SourceItem` objects for the orchestrator.

On-disk layout (produced by cb_corpus ``storage.py``)::

    <root>/raw/<bank>/<doctype>/<year>/<doc_id>.<ext>
    <root>/manifest/<bank>.jsonl    # rich index, one JSON row per document, per bank

Hybrid strategy
---------------
The **disk** is the source of truth for *completeness*: every file under
``raw/`` is yielded, so nothing is ever silently skipped just because the
manifest lags behind the downloads (this happens — a re-download can add
thousands of PDFs before the manifest is regenerated).

The **manifest** is the source of truth for *rich metadata*: when an on-disk
file's ``doc_id`` (its filename stem, == cb_corpus's stable sha1 id) is found in
the manifest we attach the real publication ``date``, ``title``, ``pdf_url`` and
``sha256`` (``metadata_source="manifest"``). When it is not yet indexed we fall
back to what the path encodes — ``bank``/``doctype``/``year`` — and date the doc
to ``<year>-01-01`` (cb_corpus's own year→Jan-1 convention, so working papers,
which are year-only at the source anyway, match exactly). Such rows are flagged
``metadata_source="path"`` / ``date_granularity="year"`` so downstream (the
quant point-in-time layer) can treat them conservatively and so a
later ``cb_corpus reindex-from-disk`` transparently upgrades them to exact dates.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional, Sequence

from ..config import get_path
from ..core import SourceItem

# Name of the ``.env`` key holding this machine's corpus root. The path itself
# is never hard-coded here: it varies per machine and lives in ``.env`` (see
# ``.env.example``), or can be passed explicitly via the CLI ``--root`` flag.
ROOT_ENV_KEY = "CB_CORPUS_ROOT"


def default_root() -> Optional[Path]:
    """Corpus root for this machine, read from ``CB_CORPUS_ROOT`` in ``.env``.

    Returns ``None`` if it is not configured (the caller then requires an
    explicit ``--root``).
    """
    return get_path(ROOT_ENV_KEY)

# cb_corpus taxonomy (code -> human label). Kept local so this connector has no
# import dependency on the cb_corpus project. Group = first letter of the code.
DOCTYPE_LABELS = {
    "A1": "Rate-decision press release",
    "A2": "Policy statement",
    "A3": "Meeting minutes / accounts / summary of deliberations",
    "A4": "Voting record",
    "B1": "Press-conference transcript / Q&A",
    "B2": "Opening remarks / webcast notes",
    "C1": "Speech",
    "C2": "Interview / op-ed / testimony",
    "D1": "Working paper",
    "D2": "Occasional / discussion paper / staff note",
    "D3": "Economic letter / research blog",
    "E1": "Monetary policy / inflation report",
    "E2": "Financial stability report",
    "E3": "Annual report",
    "E4": "Economic / quarterly bulletin",
    "F1": "Staff economic projections / forecasts",
    "G1": "Regulatory notice / consultation",
    "G2": "Statistical release",
    "G3": "Supervisory report",
}


def _parse_year(value: str) -> Optional[int]:
    return int(value) if value.isdigit() else None


def _has_pdf_sibling(html_path: Path) -> bool:
    return html_path.with_suffix(".pdf").exists()


def _manifest_files(root: Path) -> list[Path]:
    """Per-bank manifest files ``<root>/manifest/<bank>.jsonl`` (sorted).

    cb_corpus writes one JSON-lines file per bank; the pre-split single
    ``manifest.jsonl`` no longer exists and is deliberately not read.
    Returns ``[]`` when the directory is absent or holds no ``.jsonl``.
    """
    manifest_dir = root / "manifest"
    if not manifest_dir.is_dir():
        return []
    return sorted(p for p in manifest_dir.glob("*.jsonl") if p.is_file())


def _load_manifest_index(files: Sequence[Path]) -> dict[str, dict]:
    """Build a ``doc_id -> manifest row`` index across all per-bank files.

    Keyed by the manifest ``doc_id``, which equals the on-disk filename stem
    (files are stored as ``<doc_id>.<ext>``), so the join with the disk walk is
    exact. Blank and malformed lines (torn appends) are skipped so one bad row
    costs only its own document.
    """
    index: dict[str, dict] = {}
    for manifest in files:
        with manifest.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                doc_id = rec.get("doc_id")
                if doc_id:
                    index[doc_id] = rec
    return index


def _build_payload(
    *, bank_code: str, doc_type: str, group: str, year: Optional[int],
    ext: str, rel_path: str, rec: Optional[dict],
) -> dict:
    """Chunk payload for one document — rich from the manifest, else path-derived."""
    payload = {
        "source": "cb_corpus",
        "bank_code": bank_code,
        "doc_type": doc_type,
        "doc_type_label": DOCTYPE_LABELS.get(doc_type, ""),
        "doc_group": group,
        "year": year,
        "ext": ext.lstrip("."),
        "rel_path": rel_path,
    }
    if rec is not None:
        payload.update({
            "publication_date": rec.get("date"),       # as-of key (exact ISO)
            "title": rec.get("title", ""),
            "url": rec.get("pdf_url", ""),
            "sha256": rec.get("sha256", ""),
            "provenance": rec.get("provenance", ""),
            "metadata_source": "manifest",
            "date_granularity": "source",
        })
    else:
        # Not indexed yet: derive from the path. Use cb_corpus's year->Jan-1
        # convention so the date matches the manifest's granularity for papers.
        payload.update({
            "publication_date": f"{year:04d}-01-01" if year else None,
            "title": "",
            "url": "",
            "sha256": "",
            "provenance": "disk",
            "metadata_source": "path",
            "date_granularity": "year",
        })
    return payload


def iter_items(
    root: Optional[Path] = None,
    *,
    banks: Optional[Sequence[str]] = None,
    doctypes: Optional[Sequence[str]] = None,
    groups: Optional[Sequence[str]] = None,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    include_html: bool = False,
    prefer_manifest: bool = True,
) -> Iterator[SourceItem]:
    """Walk the corpus and yield one :class:`SourceItem` per document.

    Parameters
    ----------
    root:
        Corpus root (the folder that contains ``raw/``). Defaults to the
        ``CB_CORPUS_ROOT`` value from ``.env``.
    banks / doctypes / groups:
        Optional case-insensitive allow-lists (e.g. ``["ecb", "fr"]``,
        ``["C1", "A3"]``, ``["A", "C"]``). ``None`` means "all".
    year_min / year_max:
        Inclusive year bounds.
    include_html:
        If False (default) only ``.pdf`` files are yielded. If True, ``.html``
        files are also yielded — but only when they have no ``.pdf`` sibling
        (the PDF is the canonical artifact when both exist).
    prefer_manifest:
        If True (default) on-disk files are enriched with manifest metadata when
        their ``doc_id`` is indexed. If False the manifest is ignored entirely
        (pure path-derived metadata) — mostly useful for testing.
    """
    if root is None:
        root = default_root()
    if root is None:
        raise ValueError(
            f"cb_corpus root not configured: set {ROOT_ENV_KEY} in .env "
            "(copy .env.example) or pass --root"
        )
    raw = Path(root) / "raw"
    if not raw.is_dir():
        raise FileNotFoundError(f"corpus raw dir not found: {raw}")

    bank_set = {b.lower() for b in banks} if banks else None
    doctype_set = {d.upper() for d in doctypes} if doctypes else None
    group_set = {g.upper() for g in groups} if groups else None

    # Walk the disk for completeness; enrich from the per-bank manifest index.
    # No manifest at all is a configuration error (mis-pointed root, unsynced
    # share), not a corpus of undated documents — fail loudly, like raw/.
    if prefer_manifest:
        files = _manifest_files(Path(root))
        if not files:
            raise FileNotFoundError(
                f"cb_corpus manifest dir not found or empty: {Path(root) / 'manifest'} "
                "(expected one <bank>.jsonl per bank; pass prefer_manifest=False "
                "for a disk-only walk)"
            )
        index = _load_manifest_index(files)
    else:
        index = {}

    for bank_dir in sorted(p for p in raw.iterdir() if p.is_dir()):
        bank_code = bank_dir.name
        if bank_set and bank_code.lower() not in bank_set:
            continue

        for dt_dir in sorted(p for p in bank_dir.iterdir() if p.is_dir()):
            doc_type = dt_dir.name.upper()
            group = doc_type[:1]
            if doctype_set and doc_type not in doctype_set:
                continue
            if group_set and group not in group_set:
                continue

            for year_dir in sorted(p for p in dt_dir.iterdir() if p.is_dir()):
                year = _parse_year(year_dir.name)
                bounded = year_min is not None or year_max is not None
                if year is None:
                    if bounded:
                        continue  # an unknown year cannot satisfy a bound
                else:
                    if year_min is not None and year < year_min:
                        continue
                    if year_max is not None and year > year_max:
                        continue

                for f in sorted(year_dir.iterdir()):
                    if not f.is_file():
                        continue
                    ext = f.suffix.lower()
                    if ext == ".pdf":
                        pass
                    elif ext == ".html" and include_html:
                        if _has_pdf_sibling(f):
                            continue  # PDF sibling is canonical
                    else:
                        continue  # .DS_Store, .html (when PDF exists / excluded), etc.

                    doc_id = f.stem  # == cb_corpus's stable sha1 doc_id
                    rel = f"{bank_code}/{doc_type}/{year_dir.name}/{f.name}"
                    payload = _build_payload(
                        bank_code=bank_code, doc_type=doc_type, group=group,
                        year=year, ext=ext, rel_path=rel, rec=index.get(doc_id),
                    )
                    yield SourceItem(doc_id=doc_id, path=f, payload=payload)
