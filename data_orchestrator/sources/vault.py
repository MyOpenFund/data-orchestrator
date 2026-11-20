"""Vault selection source: the anti-join over documents/rag_ingestions.

"New document detection" is this query — everything the vault knows for a
corpus that the target collection has not ingested yet, restricted to the
types the engine can read (``routing.supported_mime_types``). The engine
dispatches on the file extension, so a row is supported iff its
``local_path`` ends with an extension of a supported type AND its mime_type
is either unusable (NULL/blank) or itself supported; a supported mime with a
foreign or missing extension, or a NULL ``local_path``, is unsupported. The
candidates it leaves out are counted per source by ``count_unsupported``,
never silently. Payloads
are built by ``payload.row_to_payload``: the vault's normalized columns
(``document_url``, ``source_url`` and ``mime_type`` included) merged with the
corpus-specific ``extra`` fields (columns always win), without the keys of
``extra`` dropped by design (``DATA_ORCHESTRATOR_PAYLOAD_DROP_EXTRA_KEYS``,
default ``html_path``, ``alt_urls``, ``alt_source_urls``, ``local_path``;
left out silently, not counted). A key of ``extra`` that
the engine writes itself (``filename``, ``chunk_number``, ``ingestion_date``,
``text``, ``page``, and the derived ``date_start`` / ``date_end``) is left
out and counted, never sent: the engine refuses it, and one stray key must not
fail the document. ``date_precision``, ``date_kind``, ``entity_scheme`` and
``entity_value`` are columns since the documents contract (keys of ``extra``
before it, still read from there when the column is NULL); the engine derives
the interval from ``date`` and ``date_precision``. File paths join the
per-machine corpus root (routing) with the vault's ``local_path``, relative to
that root; a ``local_path`` the contract refuses (``routing.BadLocalPath``) is
skipped and counted, never read. A document is ``(corpus, doc_id)``: the
anti-join matches both.

Transition (2026-11 rename): the vault's ``pdf_url`` column became
``document_url``. :func:`iter_items` reads which of the two the vault has, once
per connection (``vault.document_url_column``), and selects it as
``document_url``, so a run against a vault that has not run the rename yet
still carries the address. To remove once every vault has run it.
"""
from __future__ import annotations

import logging
from typing import Iterator, Sequence

from ..core import SourceItem, source_label
from ..payload import PAYLOAD_COLUMNS, row_to_payload  # noqa: F401  (row_to_payload: re-exported)
from ..vault import DOCUMENT_URL_COLUMN, document_url_column, document_url_select
from ..routing import (
    BadLocalPath, resolve_local_path, supported_extensions, supported_mime_types, validate_mimes,
)

logger = logging.getLogger(__name__)

_PAYLOAD_COLUMNS = PAYLOAD_COLUMNS

#: How many doc_ids of rows refused for their local_path the warning names.
REFUSED_EXAMPLES = 10

_SELECT_COLUMNS = _PAYLOAD_COLUMNS + ("local_path", "extra")


# Whitespace stripped from both ends of mime_type and the end of local_path.
_WS = "E' \\t\\r\\n'"

# A vault mime_type normalised for comparison: lower-cased, parameters dropped
# ("text/plain; charset=utf-8" -> "text/plain"), spaces/tabs/CR/LF trimmed,
# blank treated as NULL.
_MIME_EXPR = f"nullif(btrim(split_part(lower(d.mime_type), ';', 1), {_WS}), '')"
_PATH_EXPR = f"lower(rtrim(d.local_path, {_WS}))"

# Supported = the path ends with a supported extension (the engine dispatches
# on it) AND the mime is unusable or supported. A supported mime on a foreign
# extension ("application/pdf" + ".html") or on no extension is unsupported.
# coalesce(..., false): a NULL local_path makes the whole test NULL, which
# would drop the row from both the selection and the unsupported count; with
# it, the selection and its NOT form an exact partition of the candidates.
_SUPPORTED_PREDICATE = (
    f"coalesce({_PATH_EXPR} LIKE ANY(%(supported_ext_patterns)s) AND "
    f"({_MIME_EXPR} IS NULL OR {_MIME_EXPR} = ANY(%(supported_mimes)s)), "
    "false)"
)


def _where(
    corpus: str,
    collection: str,
    *,
    supported_mimes: Sequence[str] | None = None,
    source_codes: Sequence[str] | None = None,
    doctypes: Sequence[str] | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
    languages: Sequence[str] | None = None,
) -> tuple[list[str], dict]:
    """Candidate clauses (without the supported-type term) and their params.

    ``supported_mimes`` None -> ``routing.supported_mime_types()``; an explicit
    list is validated by ``routing.validate_mimes`` (``ValueError``).
    """
    if supported_mimes is None:
        supported_mimes = supported_mime_types()
    else:
        supported_mimes = validate_mimes(supported_mimes, origin="supported_mimes")
    clauses = ["d.corpus = %(corpus)s", "d.deleted_at IS NULL", "r.doc_id IS NULL"]
    params: dict = {
        "corpus": corpus, "collection": collection,
        "supported_mimes": list(supported_mimes),
        "supported_ext_patterns": [f"%{ext}" for ext in supported_extensions(supported_mimes)],
    }
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
    return clauses, params


# A document is (corpus, doc_id) in the vault (documents contract): the
# anti-join matches both, so another corpus's doc_id never hides one.
_FROM = ("FROM documents d\n"
         "LEFT JOIN rag_ingestions r\n"
         "  ON r.corpus = d.corpus AND r.doc_id = d.doc_id AND r.collection = %(collection)s\n")


def _select_list(document_url_from: str) -> str:
    """``_SELECT_COLUMNS`` as the select list: every column under its own
    name, the document's address read from ``document_url_from`` and named
    ``document_url``."""
    return ", ".join(document_url_select(document_url_from) if c == DOCUMENT_URL_COLUMN
                     else f"d.{c}" for c in _SELECT_COLUMNS)


def build_selection_sql(corpus, collection, *, supported_mimes=None,
                        document_url_from: str = DOCUMENT_URL_COLUMN, **filters):
    """The anti-join restricted to supported types. Returns (sql, params).

    ``supported_mimes`` None -> ``routing.supported_mime_types()``.
    ``document_url_from``: the vault column of the document's address
    (``document_url``, or ``pdf_url`` on a vault before the rename), selected
    as ``document_url``.
    """
    clauses, params = _where(corpus, collection, supported_mimes=supported_mimes, **filters)
    clauses.append(_SUPPORTED_PREDICATE)
    sql = (f"SELECT {_select_list(document_url_from)}\n" + _FROM +
           f"WHERE {' AND '.join(clauses)}\nORDER BY d.doc_id")
    return sql, params


def build_unsupported_sql(corpus, collection, *, supported_mimes=None, **filters):
    """Same candidates as the selection, but the unsupported ones, per source_code."""
    clauses, params = _where(corpus, collection, supported_mimes=supported_mimes, **filters)
    clauses.append(f"NOT {_SUPPORTED_PREDICATE}")
    sql = ("SELECT d.source_code, count(*)\n" + _FROM +
           f"WHERE {' AND '.join(clauses)}\nGROUP BY d.source_code\nORDER BY d.source_code")
    return sql, params


def count_unsupported(conn, corpus, collection, **kwargs) -> dict[str, int]:
    """{source_code: n} of not-yet-ingested candidates the engine cannot read.

    A NULL source_code is reported under ``corpus`` — ``core.source_label``,
    the label ``core._bump`` files a NULL-source vault item under (its
    payload carries ``corpus``), so the run report's ``sources[]`` merges both
    counts into one entry.
    """
    sql, params = build_unsupported_sql(corpus, collection, **kwargs)
    out: dict[str, int] = {}
    with conn.cursor() as cur:
        cur.execute(sql, params)
        for code, n in cur.fetchall():
            key = source_label(code, corpus)
            out[key] = out.get(key, 0) + int(n)
    return out


def row_to_item(row: dict, corpus: str) -> SourceItem | None:
    """Map one selection row to a SourceItem. None when local_path is NULL;
    ``routing.BadLocalPath`` when the documents contract refuses it."""
    local_path = row.get("local_path")
    if local_path:
        # Same trailing-whitespace trim as the SQL predicate (_PATH_EXPR), so
        # the file opened is the one whose extension was matched.
        local_path = local_path.rstrip(" \t\r\n")
    if not local_path:
        # Defensive only: the selection's SQL predicate already excludes a
        # NULL/blank local_path, so this guard should never fire.
        return None
    payload, dropped = row_to_payload(row)
    return SourceItem(doc_id=row["doc_id"], path=resolve_local_path(corpus, local_path),
                      payload=payload, dropped_keys=dropped)


def iter_items(conn, corpus: str, collection: str, **filters) -> Iterator[SourceItem]:
    """Yield SourceItems for every vault doc not yet in ``collection``.

    The column of the document's address is read from the vault first
    (``vault.document_url_column``: ``document_url``, or ``pdf_url`` before
    the rename)."""
    sql, params = build_selection_sql(corpus, collection,
                                      document_url_from=document_url_column(conn), **filters)
    skipped = 0
    refused: list[str] = []
    with conn.cursor() as cur:
        cur.execute(sql, params)
        columns = [d[0] for d in cur.description]
        for raw in cur:
            row = dict(zip(columns, raw))
            try:
                item = row_to_item(row, corpus)
            except BadLocalPath:
                refused.append(row["doc_id"])
                continue
            if item is None:
                skipped += 1
                continue
            yield item
    if skipped:
        logger.warning("vault selection: %d row(s) without local_path skipped", skipped)
    if refused:
        logger.warning("vault selection: %d row(s) with a local_path the documents contract "
                       "refuses skipped, never read: %s", len(refused),
                       ", ".join(refused[:REFUSED_EXAMPLES]))
