"""Vault selection source: the anti-join over documents/rag_ingestions.

"New document detection" is this query — everything the vault knows for a
corpus that the target collection has not ingested yet. Payloads are built
from the vault's normalized columns merged with the corpus-specific ``extra``
fields (columns always win). File paths join the per-machine corpus root
(routing) with the vault's relative ``local_path``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Optional, Sequence

from ..core import SourceItem
from ..routing import corpus_root

logger = logging.getLogger(__name__)

_PAYLOAD_COLUMNS = (
    "doc_id", "corpus", "source_code", "doc_type", "title", "date",
    "year", "language", "sha256", "provenance",
)

_SELECT_COLUMNS = _PAYLOAD_COLUMNS + ("local_path", "extra")


def build_selection_sql(
    corpus: str,
    collection: str,
    *,
    source_codes: Optional[Sequence[str]] = None,
    doctypes: Optional[Sequence[str]] = None,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    languages: Optional[Sequence[str]] = None,
) -> tuple[str, dict]:
    """The anti-join with optional filters. Returns (sql, named params)."""
    clauses = [
        "d.corpus = %(corpus)s",
        "d.deleted_at IS NULL",
        "r.doc_id IS NULL",
    ]
    params: dict = {"corpus": corpus, "collection": collection}
    if source_codes:
        clauses.append("d.source_code = ANY(%(source_codes)s)")
        params["source_codes"] = list(source_codes)
    if doctypes:
        clauses.append("d.doc_type = ANY(%(doctypes)s)")
        params["doctypes"] = list(doctypes)
    if year_min is not None:
        clauses.append("d.year >= %(year_min)s")
        params["year_min"] = year_min
    if year_max is not None:
        clauses.append("d.year <= %(year_max)s")
        params["year_max"] = year_max
    if languages:
        clauses.append("d.language = ANY(%(languages)s)")
        params["languages"] = list(languages)

    sql = (
        f"SELECT {', '.join('d.' + c for c in _SELECT_COLUMNS)}\n"
        "FROM documents d\n"
        "LEFT JOIN rag_ingestions r\n"
        "  ON r.doc_id = d.doc_id AND r.collection = %(collection)s\n"
        f"WHERE {' AND '.join(clauses)}\n"
        "ORDER BY d.doc_id"
    )
    return sql, params


def row_to_item(row: dict, root: Path) -> SourceItem | None:
    """Map one selection row to a SourceItem. None when local_path is NULL."""
    local_path = row.get("local_path")
    if not local_path:
        return None
    extra = row.get("extra") or {}
    payload = dict(extra)
    for col in _PAYLOAD_COLUMNS:
        value = row.get(col)
        if value is None:
            payload.pop(col, None)
            continue
        payload[col] = value.isoformat() if hasattr(value, "isoformat") else value
    return SourceItem(doc_id=row["doc_id"], path=root / local_path, payload=payload)


def iter_items(conn, corpus: str, collection: str, **filters) -> Iterator[SourceItem]:
    """Yield SourceItems for every vault doc not yet in ``collection``."""
    root = corpus_root(corpus)
    sql, params = build_selection_sql(corpus, collection, **filters)
    skipped = 0
    with conn.cursor() as cur:
        cur.execute(sql, params)
        columns = [d[0] for d in cur.description]
        for raw in cur:
            item = row_to_item(dict(zip(columns, raw)), root)
            if item is None:
                skipped += 1
                continue
            yield item
    if skipped:
        logger.warning("vault selection: %d row(s) without local_path skipped", skipped)
