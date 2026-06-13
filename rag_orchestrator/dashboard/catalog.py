"""Read layer for the dashboard — no database, just live reads.

Two sources of truth:

* **Corpus state** = the cb_corpus ``manifest.jsonl`` (one row per scraped
  document, with the real publication ``date``, title, url, sha256, path).
* **RAG state** = Qdrant, scrolled live to learn which ``doc_id`` are actually
  ingested and how many chunks each holds.

Everything is keyed by the cb_corpus ``doc_id`` (the stable sha1 hash), so the
two states join cleanly. Qdrant being unreachable degrades gracefully to the
corpus-only view.
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

from ..config import get_path

# ``CB_CORPUS_ROOT`` (in .env) points to the folder that holds the synced corpus.
ROOT_ENV_KEY = "CB_CORPUS_ROOT"
MANIFEST_NAME = "manifest.jsonl"


# ---------------------------------------------------------------------------
# Locating the manifest
# ---------------------------------------------------------------------------
def find_manifest() -> Optional[Path]:
    """Best-effort discovery of the cb_corpus ``manifest.jsonl``.

    Order: explicit ``CB_CORPUS_ROOT`` (``<root>/manifest.jsonl`` then
    ``<root>/data/manifest.jsonl``), then a walk up this repo's ancestors looking
    for a sibling ``cb_corpus/data/manifest.jsonl``.
    """
    root = get_path(ROOT_ENV_KEY)
    if root is not None:
        for cand in (root / MANIFEST_NAME, root / "data" / MANIFEST_NAME):
            if cand.is_file():
                return cand

    here = Path(__file__).resolve()
    for ancestor in here.parents:
        cand = ancestor / "cb_corpus" / "data" / MANIFEST_NAME
        if cand.is_file():
            return cand
    return None


def _resolve_file(manifest_dir: Path, local_path: str) -> Path:
    """Resolve a manifest ``local_path`` (e.g. ``data/raw/us/C1/...pdf``).

    Handles both on-disk layouts: the git repo (``cb_corpus/data`` + ``data/raw``)
    and a synced data folder (``<root>`` containing ``raw/`` directly). Returns
    the first candidate that exists, else the git-layout default for display.
    """
    lp = Path(local_path)
    candidates = [manifest_dir.parent / lp]  # git layout
    try:
        candidates.append(manifest_dir / lp.relative_to("data"))  # synced layout
    except ValueError:
        pass
    candidates.append(manifest_dir / lp)
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


# ---------------------------------------------------------------------------
# Corpus state (manifest)
# ---------------------------------------------------------------------------
DOC_GROUP_LABELS = {
    "A": "Monetary policy",
    "B": "Transcripts",
    "C": "Speeches & interviews",
    "D": "Research",
    "E": "Reports",
    "F": "Projections",
    "G": "Statistical / supervisory",
}


def _parse_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    try:
        return datetime.strptime(value[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def load_manifest(manifest_path: Optional[Path] = None) -> list[dict]:
    """Read the manifest into a list of enriched dict rows.

    Each row gains: ``pub_date`` (``date`` or ``None``), ``doc_group``,
    ``group_label``, ``abs_path`` (resolved absolute :class:`Path`), and
    ``source`` (defaults to ``"cb_corpus"``).
    """
    if manifest_path is None:
        manifest_path = find_manifest()
    if manifest_path is None or not Path(manifest_path).is_file():
        return []

    manifest_path = Path(manifest_path)
    manifest_dir = manifest_path.parent
    rows: list[dict] = []
    with manifest_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            doc_type = (rec.get("doc_type") or "").upper()
            group = doc_type[:1]
            rec["source"] = rec.get("source", "cb_corpus")
            rec["pub_date"] = _parse_date(rec.get("date"))
            rec["doc_group"] = group
            rec["group_label"] = DOC_GROUP_LABELS.get(group, "Other")
            rec["abs_path"] = _resolve_file(manifest_dir, rec.get("local_path", ""))
            rows.append(rec)
    return rows


def load_corpus(manifest_path: Optional[Path] = None) -> list[dict]:
    """Disk-truth corpus: every PDF under ``raw/``, enriched from the manifest.

    The disk is the source of truth for *completeness* (the manifest can lag
    behind downloads — e.g. it was reset while PDFs kept accumulating); the
    manifest supplies *rich metadata* (exact ``pub_date``, ``title``, ``url``)
    whenever an on-disk ``doc_id`` is indexed, else we fall back to what the path
    encodes (``bank``/``doctype``/``year``) and date the doc to ``<year>-01-01``.

    This mirrors the orchestrator's hybrid connector, so the dashboard's corpus
    count matches exactly what would be ingested. Falls back to the manifest-only
    view if the ``raw/`` tree can't be located.
    """
    if manifest_path is None:
        manifest_path = find_manifest()
    if manifest_path is None or not Path(manifest_path).is_file():
        return []
    manifest_path = Path(manifest_path)
    raw = manifest_path.parent / "raw"
    if not raw.is_dir():
        return load_manifest(manifest_path)  # no disk tree → manifest-only

    index: dict[str, dict] = {}
    with manifest_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            did = rec.get("doc_id")
            if did:
                index[did] = rec

    rows: list[dict] = []
    for f in raw.glob("*/*/*/*.pdf"):
        parts = f.relative_to(raw).parts  # bank/type/year/file.pdf
        if len(parts) < 4:
            continue
        bank, doc_type, ystr = parts[0], parts[1].upper(), parts[2]
        year = int(ystr) if ystr.isdigit() else None
        group = doc_type[:1]
        rec = index.get(f.stem)
        if rec is not None:
            rows.append({
                "doc_id": f.stem, "source": rec.get("source", "cb_corpus"),
                "bank_code": bank, "doc_type": doc_type, "doc_group": group,
                "doc_type_label": DOCTYPE_LABELS.get(doc_type, doc_type),
                "group_label": DOC_GROUP_LABELS.get(group, "Other"),
                "title": rec.get("title", ""), "pdf_url": rec.get("pdf_url", ""),
                "provenance": rec.get("provenance", ""),
                "pub_date": _parse_date(rec.get("date")),
                "year": rec.get("year", year), "metadata_source": "manifest",
                "abs_path": f,
            })
        else:
            rows.append({
                "doc_id": f.stem, "source": "cb_corpus",
                "bank_code": bank, "doc_type": doc_type, "doc_group": group,
                "doc_type_label": DOCTYPE_LABELS.get(doc_type, doc_type),
                "group_label": DOC_GROUP_LABELS.get(group, "Other"),
                "title": "", "pdf_url": "", "provenance": "disk",
                "pub_date": date(year, 1, 1) if year else None,
                "year": year, "metadata_source": "path",
                "abs_path": f,
            })
    return rows


# ---------------------------------------------------------------------------
# RAG state (Qdrant) — queried live, optional
# ---------------------------------------------------------------------------
@dataclass
class RagState:
    """What the RAG actually contains, learned by scrolling Qdrant."""

    reachable: bool
    collections: list[str]
    chunks_by_doc: dict[str, int]   # doc_id -> n_chunks
    total_points: int
    error: str = ""


def get_rag_state(
    collection: str,
    *,
    host: str = "localhost",
    port: int = 6333,
    timeout: float = 5.0,
) -> RagState:
    """Scroll a Qdrant collection and tally chunks per ``doc_id``.

    Never raises: an unreachable Qdrant or a missing collection returns a
    ``RagState`` with ``reachable=False`` / empty counts so the dashboard can
    still render the corpus-only view.
    """
    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:  # pragma: no cover
        return RagState(False, [], {}, 0, error=f"qdrant-client not installed: {exc}")

    try:
        client = QdrantClient(host=host, port=port, timeout=timeout)
        names = [c.name for c in client.get_collections().collections]
    except Exception as exc:  # noqa: BLE001 — Qdrant down is an expected state
        return RagState(False, [], {}, 0, error=str(exc))

    if collection not in names:
        return RagState(True, names, {}, 0, error=f"collection '{collection}' not found")

    chunks_by_doc: dict[str, int] = {}
    total = 0
    offset = None
    try:
        while True:
            points, offset = client.scroll(
                collection_name=collection,
                with_payload=["doc_id"],
                with_vectors=False,
                limit=1000,
                offset=offset,
            )
            for p in points:
                total += 1
                doc_id = (p.payload or {}).get("doc_id")
                if doc_id is not None:
                    chunks_by_doc[doc_id] = chunks_by_doc.get(doc_id, 0) + 1
            if offset is None:
                break
    except Exception as exc:  # noqa: BLE001
        return RagState(True, names, chunks_by_doc, total, error=str(exc))

    return RagState(True, names, chunks_by_doc, total)


# ---------------------------------------------------------------------------
# Join + as-of filtering
# ---------------------------------------------------------------------------
def build_documents(
    manifest_rows: Iterable[dict],
    rag: RagState,
    *,
    as_of: Optional[date] = None,
) -> list[dict]:
    """Join corpus rows with RAG state and apply the as-of (publication) filter.

    A document is kept when its ``pub_date`` is on/before ``as_of`` (undated docs
    are kept so they stay visible). Each output row gains ``in_rag`` (bool) and
    ``n_chunks`` (int) from the RAG state.
    """
    out: list[dict] = []
    for r in manifest_rows:
        pub = r.get("pub_date")
        if as_of is not None and pub is not None and pub > as_of:
            continue
        doc_id = r.get("doc_id")
        n = rag.chunks_by_doc.get(doc_id, 0)
        row = dict(r)
        row["in_rag"] = n > 0
        row["n_chunks"] = n
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# Inventory & quality control: coverage matrix, anomalies, cadence, fetch errors
# ---------------------------------------------------------------------------
# Known per-year expectations, mirrored from cb_corpus adapters' `expected_per_year`
# (the published meeting/release calendars). Anchors the anomaly check for the
# majors; everything else falls back to a data-driven baseline (median).
EXPECTED_PER_YEAR: dict[tuple[str, str], int] = {
    ("ecb", "A1"): 8, ("ecb", "A2"): 8, ("ecb", "A3"): 8, ("ecb", "E4"): 8,
    ("us", "A2"): 8, ("us", "A3"): 8, ("us", "F1"): 4,
    ("au", "A1"): 8, ("au", "A3"): 8, ("au", "E1"): 4,
}


def counts_by_bank_type_year(rows: Iterable[dict]) -> dict[tuple[str, str, int], int]:
    """Document counts keyed by (bank_code, doc_type, year)."""
    counts: dict[tuple[str, str, int], int] = defaultdict(int)
    for r in rows:
        y = r.get("year")
        if y is None:
            continue
        counts[(r.get("bank_code"), r.get("doc_type"), y)] += 1
    return dict(counts)


def coverage_matrix(rows: Iterable[dict], bank: str) -> dict:
    """Year × doc_type count grid for one bank (for a pivot table)."""
    grid: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    types: set[str] = set()
    for r in rows:
        if r.get("bank_code") != bank:
            continue
        y = r.get("year")
        if y is None:
            continue
        dt = r.get("doc_type")
        grid[y][dt] += 1
        types.add(dt)
    return {"grid": {y: dict(v) for y, v in grid.items()}, "types": sorted(types)}


# Human labels for cb_corpus doc-type codes (kept local; mirrors the taxonomy).
DOCTYPE_LABELS = {
    "A1": "Rate-decision press release", "A2": "Policy statement",
    "A3": "Minutes / accounts", "A4": "Voting record",
    "B1": "Press-conference transcript", "B2": "Opening remarks",
    "C1": "Speech", "C2": "Interview / op-ed / testimony",
    "D1": "Working paper", "D2": "Occasional / discussion paper",
    "D3": "Economic letter / blog", "E1": "Monetary policy / inflation report",
    "E2": "Financial stability report", "E3": "Annual report",
    "E4": "Economic / quarterly bulletin", "F1": "Staff projections / forecasts",
    "G1": "Regulatory notice", "G2": "Statistical release", "G3": "Supervisory report",
}


def type_summary(rows: Iterable[dict], bank: str) -> list[dict]:
    """Per doc_type inventory stats for a bank.

    Returns total, active-year span, **avg/yr** and **median/yr** (the corpus's
    real cadence), plus ``calendar_per_year`` — the bank's *published* meeting
    count (:data:`EXPECTED_PER_YEAR`) as a reference, or ``None`` if unknown.
    The corpus often stores several documents per meeting, so avg/median is the
    honest "how many do we actually have per year", and the calendar number is
    just context.
    """
    per_type_year: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        if r.get("bank_code") != bank:
            continue
        y = r.get("year")
        if y is None:
            continue
        per_type_year[r.get("doc_type")][y] += 1

    out: list[dict] = []
    for dt, yearly in per_type_year.items():
        counts = list(yearly.values())
        years = sorted(yearly)
        total = sum(counts)
        n = len(years)
        out.append({
            "doc_type": dt,
            "label": DOCTYPE_LABELS.get(dt, ""),
            "total": total,
            "years": f"{years[0]}–{years[-1]}" if years else "",
            "avg_per_year": round(total / n, 1) if n else 0,
            "median_per_year": int(statistics.median(counts)) if counts else 0,
            "calendar_per_year": EXPECTED_PER_YEAR.get((bank, dt)),
            "last_year": years[-1] if years else None,
        })
    out.sort(key=lambda r: r["doc_type"])
    return out


def anomalies(rows: Iterable[dict], *, current_year: int,
              lookback_years: int = 12, min_baseline: int = 4) -> list[dict]:
    """Flag (bank, type, year) cells that deviate from the expected count.

    `missing` (year with 0), `low` (< 50% of baseline), `exceptional` (> 150%).
    The baseline is the corpus's **own recent cadence** for that (bank, type) —
    the median of its non-zero yearly counts inside the lookback window — NOT the
    published meeting calendar: the corpus stores a varying number of documents
    per meeting (e.g. ~2 ECB accounts per meeting → 16/yr, not 8), so a meeting
    count would flag every normal year. To stay a short, actionable list the
    check is constrained to:

    * the type's **active span** (first→last year it published), so a type that
      didn't exist yet / was discontinued isn't flagged as "missing";
    * the **recent** ``lookback_years`` window, where gaps are actionable;
    * completed years only (the in-progress year is partial);
    * a baseline of at least ``min_baseline``/yr (skips low-volume noise).

    The published calendar (:data:`EXPECTED_PER_YEAR`) is surfaced separately in
    the UI as a reference, not used as the threshold here.
    """
    per_bt: dict[tuple[str, str], dict[int, int]] = defaultdict(dict)
    for (bank, dt, y), c in counts_by_bank_type_year(rows).items():
        per_bt[(bank, dt)][y] = c

    out: list[dict] = []
    for (bank, dt), yearly in per_bt.items():
        nonzero_years = [y for y, c in yearly.items() if c > 0]
        if not nonzero_years:
            continue
        span_lo, span_hi = min(nonzero_years), max(nonzero_years)
        lo = max(span_lo, current_year - lookback_years)
        hi = min(span_hi, current_year - 1)               # completed & within activity
        recent_nonzero = [yearly[y] for y in nonzero_years if lo <= y <= hi]
        if len(recent_nonzero) < 3:
            continue
        baseline = int(statistics.median(recent_nonzero))
        if baseline < min_baseline:
            continue
        basis = "recent-median"
        for y in range(lo, hi + 1):
            c = yearly.get(y, 0)
            if c == 0:
                flag = "missing"
            elif c < baseline * 0.5:
                flag = "low"
            elif c > baseline * 1.5:
                flag = "exceptional"
            else:
                continue
            out.append({"bank_code": bank, "doc_type": dt, "year": y, "count": c,
                        "expected": baseline, "basis": basis, "flag": flag})
    order = {"missing": 0, "low": 1, "exceptional": 2}
    out.sort(key=lambda r: (order.get(r["flag"], 9), r["bank_code"], r["doc_type"], r["year"]))
    return out


def upcoming(rows: Iterable[dict], *, today: date,
             horizon_days: int = 60) -> list[dict]:
    """Next-expected / overdue per recurring (bank, type), from publication cadence.

    Uses only real-dated (manifest-enriched) docs of calendar-known types so the
    `YYYY-01-01` path fallback never pollutes the interval estimate. Estimates
    the next release as last_date + median(recent intervals); `overdue` if we're
    >7 days past it, `soon` if it's within `horizon_days`.
    """
    series: dict[tuple[str, str], list[date]] = defaultdict(list)
    for r in rows:
        if r.get("metadata_source") != "manifest":
            continue
        pd_ = r.get("pub_date")
        key = (r.get("bank_code"), r.get("doc_type"))
        if pd_ is not None and key in EXPECTED_PER_YEAR:
            series[key].append(pd_)

    out: list[dict] = []
    for (bank, dt), dates in series.items():
        dates = sorted(set(dates))
        if len(dates) < 4:
            continue
        intervals = [(dates[i] - dates[i - 1]).days for i in range(1, len(dates))]
        med = statistics.median(intervals[-12:])          # recent cadence
        if med <= 0:
            continue
        nxt = dates[-1] + timedelta(days=med)
        days_until = (nxt - today).days
        status = ("overdue" if days_until < -7
                  else "soon" if days_until <= horizon_days else "on-track")
        out.append({"bank_code": bank, "doc_type": dt, "last": dates[-1],
                    "interval_days": round(med), "next_expected": nxt,
                    "days_until": days_until, "status": status})
    order = {"overdue": 0, "soon": 1, "on-track": 2}
    out.sort(key=lambda r: (order.get(r["status"], 9), r["days_until"]))
    return out


def load_discovery_errors(manifest_path: Optional[Path] = None) -> list[dict]:
    """Read cb_corpus's ``data/discovery_errors.jsonl`` (fetch-failure audit trail)."""
    if manifest_path is None:
        manifest_path = find_manifest()
    if manifest_path is None:
        return []
    path = Path(manifest_path).parent / "discovery_errors.jsonl"
    if not path.is_file():
        return []
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows
