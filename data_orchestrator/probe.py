"""Facts probe: fill documents.has_text_layer / page_count for a corpus.

Feeds the future facts-driven OCR policy. Only rows with
``has_text_layer IS NULL`` are probed, so the pass is resumable by
construction and re-runs converge to a no-op. The probe is the ONLY writer
of these two columns (the vault's manifest upsert deliberately never touches
them).

Each document's UPDATE is wrapped in its own SAVEPOINT so that one row
failing (a DB error, a trigger, a constraint) cannot poison the surrounding
batch's transaction: on Postgres, an uncaught error aborts the transaction
until a ROLLBACK (or ROLLBACK TO SAVEPOINT), so without this isolation every
statement after the failure would raise ``InFailedSqlTransaction`` and the
eventual ``commit()`` would silently discard the whole batch — including
already-"probed" successes that ``stats`` had already counted.
"""
from __future__ import annotations

import logging
from pathlib import Path

from .routing import resolve_local_path

logger = logging.getLogger(__name__)

SELECT_UNPROBED_SQL = """
SELECT doc_id, local_path FROM documents
WHERE corpus = %s AND has_text_layer IS NULL
  AND deleted_at IS NULL AND local_path IS NOT NULL
ORDER BY doc_id
"""

UPDATE_FACTS_SQL = """
UPDATE documents SET has_text_layer = %s, page_count = %s WHERE doc_id = %s
"""


def probe_file(path: Path) -> tuple[bool | None, int | None]:
    """(has_text_layer, page_count) for one file; (None, None) if unreadable."""
    if not path.exists():
        return None, None
    if path.suffix.lower() != ".pdf":
        # Non-PDF RAG-ready artifacts (.md, .txt) are text by definition.
        return True, None
    try:
        import fitz  # pymupdf, via the eigenmind dependency tree

        with fitz.open(path) as doc:
            page_count = doc.page_count
            has_text = any(page.get_text().strip() for page in doc)
        return has_text, page_count
    except Exception as exc:  # noqa: BLE001 — a corrupt PDF must not kill the pass
        logger.warning("probe failed for %s: %s", path, exc)
        return None, None


def run_probe(conn, corpus: str, *, limit: int | None = None, batch: int = 50) -> dict:
    """Probe every unprobed document of ``corpus``. Returns counters."""
    with conn.cursor() as cur:
        cur.execute(SELECT_UNPROBED_SQL, (corpus,))
        rows = cur.fetchall()
    if limit is not None:
        rows = rows[:limit]

    stats = {"probed": 0, "skipped": 0, "errors": 0}
    pending = 0
    for doc_id, local_path in rows:
        has_text, page_count = probe_file(resolve_local_path(corpus, local_path))
        if has_text is None and page_count is None:
            stats["skipped"] += 1
            continue
        try:
            with conn.cursor() as cur:
                cur.execute("SAVEPOINT probe_doc")
                cur.execute(UPDATE_FACTS_SQL, (has_text, page_count, doc_id))
                cur.execute("RELEASE SAVEPOINT probe_doc")
            stats["probed"] += 1
            pending += 1
            if pending >= batch:
                conn.commit()
                pending = 0
        except Exception as exc:  # noqa: BLE001 — one bad row must not poison the batch's transaction
            logger.warning("facts update failed for %s: %s", doc_id, exc)
            with conn.cursor() as cur:
                cur.execute("ROLLBACK TO SAVEPOINT probe_doc")
            stats["errors"] += 1
    conn.commit()
    logger.info("probe(%s): %s", corpus, stats)
    return stats
