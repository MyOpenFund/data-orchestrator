"""cb_corpus source connector.

Reads the official central-bank corpus produced by the ``cb_corpus`` project
and yields :class:`..core.SourceItem` objects for the orchestrator.

On-disk layout (produced by cb_corpus ``storage.py``)::

    <root>/raw/<bank>/<doctype>/<year>/<doc_id>.<ext>

There is no ``manifest.jsonl`` synced alongside the data, but ``bank``,
``doctype`` and ``year`` are fully recoverable from the path — that's exactly
the metadata we attach to every chunk so the vector DB can be filtered and
cited per bank / document-type / year.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional, Sequence

from ..core import SourceItem

# Default location of the synced corpus (OneDrive). Override via run.py --root.
DEFAULT_ROOT = Path(
    r"C:\Users\jeulin\OneDrive - Assicurazioni Generali S.p.A\DATABASE\cb_corpus"
)

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


def iter_items(
    root: Path = DEFAULT_ROOT,
    *,
    banks: Optional[Sequence[str]] = None,
    doctypes: Optional[Sequence[str]] = None,
    groups: Optional[Sequence[str]] = None,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    include_html: bool = False,
) -> Iterator[SourceItem]:
    """Walk the corpus and yield one :class:`SourceItem` per document.

    Parameters
    ----------
    root:
        Corpus root (the folder that contains ``raw/``).
    banks / doctypes / groups:
        Optional case-insensitive allow-lists (e.g. ``["ecb", "fr"]``,
        ``["C1", "A3"]``, ``["A", "C"]``). ``None`` means "all".
    year_min / year_max:
        Inclusive year bounds.
    include_html:
        If False (default) only ``.pdf`` files are yielded. If True, ``.html``
        files are also yielded — but only when they have no ``.pdf`` sibling
        (the PDF is the canonical artifact when both exist).
    """
    raw = Path(root) / "raw"
    if not raw.is_dir():
        raise FileNotFoundError(f"corpus raw dir not found: {raw}")

    bank_set = {b.lower() for b in banks} if banks else None
    doctype_set = {d.upper() for d in doctypes} if doctypes else None
    group_set = {g.upper() for g in groups} if groups else None

    for bank_dir in sorted(p for p in raw.iterdir() if p.is_dir()):
        bank_code = bank_dir.name
        if bank_set and bank_code.lower() not in bank_set:
            continue

        for dt_dir in sorted(p for p in bank_dir.iterdir() if p.is_dir()):
            doc_type = dt_dir.name
            group = doc_type[:1].upper()
            if doctype_set and doc_type.upper() not in doctype_set:
                continue
            if group_set and group not in group_set:
                continue

            for year_dir in sorted(p for p in dt_dir.iterdir() if p.is_dir()):
                year = _parse_year(year_dir.name)
                if year is not None:
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

                    rel = f"{bank_code}/{doc_type}/{year_dir.name}/{f.name}"
                    payload = {
                        "source": "cb_corpus",
                        "bank_code": bank_code,
                        "doc_type": doc_type,
                        "doc_type_label": DOCTYPE_LABELS.get(doc_type.upper(), ""),
                        "doc_group": group,
                        "year": year,
                        "ext": ext.lstrip("."),
                        "rel_path": rel,
                    }
                    yield SourceItem(doc_id=rel, path=f, payload=payload)
