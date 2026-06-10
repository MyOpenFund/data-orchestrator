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
import os
from dataclasses import dataclass
from datetime import date, datetime
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
# Append-only logs (queries / graphs) — state/*.jsonl, no DB
# ---------------------------------------------------------------------------
def _state_dir() -> Path:
    override = os.environ.get("RAGO_STATE_DIR")
    if override:
        return Path(override).expanduser()
    # repo root = two levels above this file (rag_orchestrator/dashboard/..)
    return Path(__file__).resolve().parents[2] / "state"


def log_jsonl(name: str, record: dict) -> None:
    """Append one record to ``state/<name>.jsonl`` (stamped with ``ts``)."""
    d = _state_dir()
    d.mkdir(parents=True, exist_ok=True)
    rec = {"ts": datetime.now().isoformat(timespec="seconds"), **record}
    with (d / f"{name}.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def load_jsonl(name: str) -> list[dict]:
    """Read ``state/<name>.jsonl`` (newest first); empty if it doesn't exist."""
    path = _state_dir() / f"{name}.jsonl"
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
    rows.reverse()
    return rows
