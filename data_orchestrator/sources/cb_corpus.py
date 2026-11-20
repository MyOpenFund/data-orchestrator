"""cb_corpus source connector.

Reads the official central-bank corpus produced by the ``cb_corpus`` project
and yields :class:`..core.SourceItem` objects for the orchestrator.

On-disk layout (produced by cb_corpus ``storage.py``)::

    <root>/raw/<bank>/<doctype>/<year>/<doc_id>.<ext>
    <root>/manifest/<bank>.jsonl   # rich index, one JSON row per document, per bank

``<root>`` is the folder that contains ``raw/`` (a synced data folder, or the
``data/`` directory of a cb_corpus checkout). A root without any
``manifest/*.jsonl`` is rejected — see :func:`iter_items`.

Hybrid strategy
---------------
The **disk** is the source of truth for *completeness*: every file under
``raw/`` is yielded, so nothing is ever silently skipped just because the
manifest lags behind the downloads (this happens — a re-download can add
thousands of PDFs before the manifest is regenerated).

The **manifest** is the source of truth for *rich metadata*: when an on-disk
file's ``doc_id`` (its filename stem, == cb_corpus's stable sha1 id) is found in
the manifest, the payload is the one the vault source gives the same document
(``payload.manifest_row_to_payload``): the manifest row split like the vault
splits it (the ``documents`` contract's columns, the rest as ``extra``), then
the vault source's rules (columns win over ``extra``, absent/null fields left
out, the keys of ``extra`` dropped by design left out silently, the keys the
engine writes itself left out and counted). The split is the contract's own
(``vault_contract.split_row`` with ``transition=False``, as the vault ingests).
The keys are the vault's: ``corpus``, ``source_code``, ``doc_type``, ``title``,
``date``, ``year``, ``language``, ``sha256``, ``provenance``, ``document_url``,
``source_url``, ``mime_type``, ``date_precision``, ``date_kind``,
``entity_scheme``, ``entity_value`` and the other fields of the row
(``date_source``, ...), plus ``metadata_source="manifest"``. A manifest row
the contract refuses (``vault_contract.check_row`` without vocabulary and with
``transition=False``, as the vault ingests: the vault keeps no row for it) is
skipped, never read, as the vault skips it: the run logs how many there are,
with the first reasons, and the CLI counts them in the run report
(``docs_refused_contract``, ``refused_contract_doc_ids``,
``refused_contract_reasons``; ``iter_items(refused=)``) and as failures for the
outcome. A row in the pre-contract form (``bank_code``, ``pdf_url``,
``local_path`` under ``data/``, no ``corpus``) is such a row
(``missing_corpus``, ``missing_source_code``, ``bad_local_path``,
``legacy_key``): no old key is mapped, and since contract 1.1.0 a row carrying
``bank_code`` or ``pdf_url`` is refused ``legacy_key`` even beside its contract
key. Migrate the manifest (central-bank-corpus ``migrate-contract``).
A ``corpus`` that is not a corpus code (``vault_contract.CODE_PATTERN``) is a
usage error (``check_corpus``, ``ValueError``), never a refusal per row.

When it is not yet indexed we fall back to what the path encodes:
``corpus``, ``source_code`` (the bank folder), ``doc_type``, ``year``, and
``date`` = ``<year>-01-01`` with ``date_precision="year"`` (the central-bank
corpus convention: a coarse date is the first day of its period, and the
precision says so); no title, no URL, no checksum (absent, never empty
strings). Such rows are flagged ``metadata_source="path"`` so downstream (the
quant point-in-time layer) can treat them conservatively and so a later
``cb_corpus reindex-from-disk`` transparently upgrades them to exact dates.
Such documents are counted (``IngestStats.docs_path_metadata``) and reported by
the CLI, so the fallback is never silent.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional, Sequence

from vault_contract import CODE_PATTERN, check_row

from ..config import get_path
from ..core import SourceItem
from ..payload import dropped_extra_keys, manifest_row_to_payload, row_to_payload

logger = logging.getLogger(__name__)

#: How many refused manifest rows a warning names, with their reasons (this
#: source's own, or the CLI's).
REFUSED_EXAMPLES = 5

# Name of the ``.env`` key holding this machine's corpus root. The path itself
# is never hard-coded here: it varies per machine and lives in ``.env`` (see
# ``.env.example``), or can be passed explicitly via the CLI ``--root`` flag.
ROOT_ENV_KEY = "CB_CORPUS_ROOT"

#: The vault corpus of this source's documents (the vault's own default
#: ``CORPUS``, ``central-bank``): the ``corpus`` of every path-derived payload
#: (a manifest row without ``corpus`` is refused ``missing_corpus``).
#: ``iter_items(corpus=...)`` (the CLI's ``--corpus``) overrides it.
DEFAULT_CORPUS = "central-bank"

#: ``date_precision`` of a path-derived date (``<year>-01-01``).
PATH_DATE_PRECISION = "year"


def default_root() -> Optional[Path]:
    """Corpus root for this machine, read from ``CB_CORPUS_ROOT`` in ``.env``.

    Returns ``None`` if it is not configured (the caller then requires an
    explicit ``--root``).
    """
    return get_path(ROOT_ENV_KEY)

def check_corpus(corpus: str) -> None:
    """Raise ``ValueError`` unless ``corpus`` is a corpus code
    (``vault_contract.CODE_PATTERN``): every manifest row is checked against
    it, so a bad one is a usage error, not a refusal of every row."""
    if not (isinstance(corpus, str) and CODE_PATTERN.fullmatch(corpus)):
        raise ValueError(f"corpus {corpus!r} is not a corpus code "
                         f"(must match {CODE_PATTERN.pattern})")


def _parse_year(value: str) -> Optional[int]:
    return int(value) if value.isdigit() else None


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


def require_manifest(root: Path) -> list[Path]:
    """Per-bank manifest files for ``root``, raising if none exist.

    Same result as :func:`_manifest_files`, but raises the actionable
    ``FileNotFoundError`` used by both :func:`iter_items` (its default
    ``prefer_manifest=True`` path) and the CLI's eager pre-flight (cli.py),
    so the two call sites can never drift on the error message.
    """
    files = _manifest_files(Path(root))
    if not files:
        raise FileNotFoundError(
            f"cb_corpus manifest dir not found or empty: {Path(root) / 'manifest'} "
            "(expected one <bank>.jsonl per bank; Python API: pass "
            "prefer_manifest=False for a disk-only walk)"
        )
    return files


def _load_manifest_index(files: Sequence[Path]) -> dict[str, dict]:
    """Build a ``doc_id -> manifest row`` index across all per-bank files.

    Keyed by the manifest ``doc_id``, which equals the on-disk filename stem
    (files are stored as ``<doc_id>.<ext>``), so the join with the disk walk is
    exact. Blank and malformed lines (torn appends) are skipped so one bad row
    costs only its own document. A ``doc_id`` appearing in two bank files is
    last-file-wins (doc_ids are corpus-unique in practice — verified 0
    duplicates over 39,449 real rows on 2026-09-03).
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
    *, doc_id: str, bank_code: str, doc_type: str, year: Optional[int],
    rec: Optional[dict], corpus: str, drop_keys: tuple[str, ...],
) -> tuple[dict, tuple[str, ...]]:
    """Payload of one document, and the reserved keys left out: the vault
    source's payload of the manifest row, else path-derived."""
    if rec is not None:
        payload, dropped = manifest_row_to_payload(rec, corpus, drop_keys)
        payload["metadata_source"] = "manifest"
        return payload, dropped
    row = {
        "doc_id": doc_id, "corpus": corpus, "source_code": bank_code, "doc_type": doc_type,
        "year": year, "date": f"{year:04d}-01-01" if year is not None else None,
        "date_precision": PATH_DATE_PRECISION if year is not None else None,
    }
    payload, dropped = row_to_payload(row, drop_keys)
    payload["metadata_source"] = "path"
    return payload, dropped


def iter_items(
    root: Optional[Path] = None,
    *,
    banks: Optional[Sequence[str]] = None,
    doctypes: Optional[Sequence[str]] = None,
    groups: Optional[Sequence[str]] = None,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    prefer_manifest: bool = True,
    corpus: str = DEFAULT_CORPUS,
    refused: Optional[list] = None,
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
    prefer_manifest:
        If True (default) on-disk files are enriched with manifest metadata when
        their ``doc_id`` is indexed, and a missing/empty ``manifest/`` is a
        configuration error (see Raises below). If False the manifest is
        ignored entirely (pure path-derived metadata) and the manifest-required
        check is disabled — a disk-only walk, Python API only (there is no CLI
        flag for it); mostly useful for testing.
    corpus:
        The vault corpus of the documents (:data:`DEFAULT_CORPUS`; the CLI
        passes ``--corpus``): the corpus every manifest row must name (a row
        without ``corpus`` is refused ``missing_corpus``, one naming another
        corpus ``corpus_conflict``, as the vault refuses them), and the
        ``corpus`` of every path-derived payload.
    refused:
        When given, receives ``(doc_id, reasons)`` (``reasons`` a tuple of
        ``vault_contract`` refusal reasons) for every manifest row the
        documents contract refuses, as they are met: those documents are
        skipped, never read, and the caller reports them. When ``None``
        (default) the source logs one warning naming the first
        :data:`REFUSED_EXAMPLES` with their reasons.

    Raises
    ------
    ValueError
        ``root`` is not configured (no argument and no ``CB_CORPUS_ROOT``),
        or ``corpus`` is not a corpus code (:func:`check_corpus`).
    FileNotFoundError
        ``<root>/raw`` does not exist, or (when ``prefer_manifest`` is True)
        ``<root>/manifest`` has no ``*.jsonl`` files.
    """
    check_corpus(corpus)
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
    bounded = year_min is not None or year_max is not None

    # Walk the disk for completeness; enrich from the per-bank manifest index.
    # No manifest at all is a configuration error (mis-pointed root, unsynced
    # share), not a corpus of undated documents — fail loudly, like raw/.
    if prefer_manifest:
        index = _load_manifest_index(require_manifest(Path(root)))
    else:
        index = {}
    drop_keys = dropped_extra_keys()
    # A caller collecting the refused rows reports them itself (the CLI: one
    # warning and the run report); for any other caller the source warns.
    warn = refused is None
    if refused is None:
        refused = []
    first_refused = len(refused)

    # The warning is given in a finally: a run that stops early (--limit, a
    # fatal error) still says which rows it skipped.
    try:
        for bank_dir in sorted(p for p in raw.iterdir() if p.is_dir()):
            bank_code = bank_dir.name
            if bank_set and bank_code.lower() not in bank_set:
                continue

            for dt_dir in sorted(p for p in bank_dir.iterdir() if p.is_dir()):
                doc_type = dt_dir.name.upper()
                group = doc_type[:1]  # the --groups filter
                if doctype_set and doc_type not in doctype_set:
                    continue
                if group_set and group not in group_set:
                    continue

                for year_dir in sorted(p for p in dt_dir.iterdir() if p.is_dir()):
                    year = _parse_year(year_dir.name)
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
                        # .pdf only: HTML is converted to .md/PDF before the
                        # engine; see issue #17. (.DS_Store, .html, etc. skipped.)
                        if ext != ".pdf":
                            continue

                        doc_id = f.stem  # == cb_corpus's stable sha1 doc_id
                        rec = index.get(doc_id)
                        if rec is not None:
                            reasons = _refusals(rec, corpus)
                            if reasons:
                                refused.append((doc_id, reasons))
                                continue  # the vault keeps no row for it
                        payload, dropped = _build_payload(
                            doc_id=doc_id, bank_code=bank_code, doc_type=doc_type, year=year,
                            rec=rec, corpus=corpus, drop_keys=drop_keys,
                        )
                        yield SourceItem(doc_id=doc_id, path=f, payload=payload,
                                         dropped_keys=dropped)
    finally:
        mine = refused[first_refused:]
        if warn and mine:
            logger.warning(
                "%d manifest row(s) the documents contract refuses (the vault keeps no row for "
                "them) skipped, never read: %s", len(mine),
                "; ".join(f"{doc_id}: {', '.join(reasons)}"
                          for doc_id, reasons in mine[:REFUSED_EXAMPLES]))


def _refusals(rec: dict, corpus: str) -> tuple[str, ...]:
    """The reasons the vault would refuse manifest row ``rec`` for
    (``vault_contract.check_row`` without vocabulary, ``transition=False``
    as the vault ingests); empty when it is accepted. ``corpus`` is a corpus
    code (:func:`check_corpus`, checked once per run)."""
    checked = check_row(rec, None, expected_corpus=corpus, transition=False)
    return tuple(r.reason for r in checked.refusals)
