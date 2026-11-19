"""``sync-metadata``: bring the date metadata of ingested documents in Qdrant in line with the vault.

The vault's ``documents`` row of a document can change after the document was
ingested (corrected dates: ``date``, ``year``, and in ``extra``
``date_precision`` / ``date_kind``). Ingestion never offers an ingested
document again (the anti-join on ``rag_ingestions``), so the payload of its
points keeps the old values. This module compares, for every document the
``rag_ingestions`` ledger records for a collection, the metadata ingestion
would build today from the vault row with what the points hold, and corrects
the points with rag-engine's ``QdrantStore.set_payload``: no vector is read,
sent or rewritten, no point is created, ``rag_ingestions`` is not written.

The keys it owns (:data:`OWNED_KEYS`): ``date``, ``year``, ``date_precision``,
``date_kind``, ``date_source`` (where the date was read: a fresh ingestion
copies it from ``extra``, so a synced point must carry it too) and the interval the engine derives from the first ones,
``date_start`` / ``date_end``. ``title`` only on request
(:data:`TITLE_KEY`, ``--include-title``). The links only on request
(``--include-links``): :data:`LINK_KEYS` (``document_url``, ``source_url``,
``mime_type``, set from the vault), :data:`LEGACY_LINK_KEYS` (``pdf_url``, the
former name of ``document_url``: an ingestion never sends it, so it must be
absent and is removed from the points ingested before the rename; one run
renames the key on every point) and the keys of ``extra`` dropped by
design (``payload.dropped_extra_keys``, ``DATA_ORCHESTRATOR_PAYLOAD_DROP_EXTRA_KEYS``:
an ingestion never sends them, so they must be absent and are removed from the
points ingested before they were dropped). Nothing else is compared or written.

The vault's column of the document's address is ``document_url``, or
``pdf_url`` on a vault that has not run the 2026-11 rename yet: it is read once
per connection (``vault.document_url_column``) and selected as
``document_url`` (transition only).

**The expected values come from the ingestion code path.** The vault row goes
through ``payload.row_to_payload`` (the metadata ``sources.vault.row_to_item``
hands the engine) and then through ``rag_engine.schema.build_payload`` (what the
engine writes on every point); the owned keys are read from that payload. A
key absent there (a NULL column, a key not in ``extra``) must be absent from
the points; a null stored in a point counts as absent (Qdrant drops a key set
to null, and a filter reads a null as a missing key).

**The write** is ``date_fields(date, date_precision, year=year)`` plus
``date_kind`` and ``date_source`` (``None``, which removes the key, when the
vault has none) and,
with ``--include-title``, ``title``, with ``--include-links`` the links and a
``None`` (a removal) for each key dropped by design; before sending it, it is checked to be
exactly the expected values (a ``None`` for each absent key).

**Every point is read**, not one per document. rag-engine copies the same
metadata into every point of an ingestion (``ingest_document`` passes one
``metadata`` mapping to ``build_payload`` for each chunk), so the points of a
document agree as long as nothing else wrote them; but an earlier ingestion
that produced more chunks, or a correction interrupted by a crash, could leave
points that disagree. The read is cheap (the owned keys and ``doc_id`` only,
no vector) and it is the guard: a document whose points disagree is reported
and corrected on all of its points (``set_payload`` filters on ``doc_id``).

Dry run by default. Idempotent: a second ``--write`` finds nothing to write.
Resumable: an interrupted run leaves every document either corrected or not,
and the next run takes the rest.
"""
from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from qdrant_client import models
from rag_engine.schema import build_payload, date_fields
from rag_engine.store import MissingIndexError, QdrantAuthError

from .payload import PAYLOAD_COLUMNS as _PAYLOAD_COLUMNS
from .payload import dropped_extra_keys, row_to_payload
from .vault import document_url_column, document_url_select

#: The payload keys sync-metadata compares and writes, in report order.
OWNED_KEYS: tuple[str, ...] = (
    "date", "year", "date_precision", "date_kind", "date_source", "date_start", "date_end",
)

#: Compared and written only with ``include_title`` (``--include-title``).
TITLE_KEY = "title"

#: Compared and written only with ``include_links`` (``--include-links``), with
#: :data:`LEGACY_LINK_KEYS` and the keys of ``extra`` dropped by design
#: (``payload.dropped_extra_keys``), which must be absent.
LINK_KEYS: tuple[str, ...] = ("document_url", "source_url", "mime_type")

#: Former names of :data:`LINK_KEYS` (``pdf_url``, now ``document_url``):
#: owned with ``include_links`` and expected absent (``payload.row_to_payload``
#: never carries them), so a ``--write`` removes them from the points while it
#: sets the new name. Transition only: to remove once no point carries one.
LEGACY_LINK_KEYS: tuple[str, ...] = ("pdf_url",)

#: Points read per scroll request (``--page-size``).
DEFAULT_PAGE_SIZE = 256
#: Documents per scroll filter (``--doc-batch``): one ``doc_id`` match-any per batch.
DEFAULT_DOC_BATCH = 100

# Actions, one per document.
UNCHANGED = "unchanged"
UPDATED = "updated"
UNDATED = "undated"
MISSING_IN_QDRANT = "missing-in-qdrant"
MISSING_IN_VAULT = "missing-in-vault"
ACTIONS = (UNCHANGED, UPDATED, UNDATED, MISSING_IN_QDRANT, MISSING_IN_VAULT)

#: One row per document and owned key that differs (one row, ``field`` empty,
#: for a document missing in Qdrant or in the vault). Values: empty when the key
#: is absent, a string as it is, anything else as JSON; when the points of the
#: document disagree, ``qdrant_value`` is the JSON list of their values.
CSV_COLUMNS = ("doc_id", "action", "field", "qdrant_value", "vault_value", "note")

#: ``{document_url}``: the select-list entry of the document's address
#: (``vault.document_url_select``), ``{doc_ids}`` / ``{limit}``: the filters.
LEDGER_SQL = (
    "SELECT r.doc_id AS ledger_doc_id, d.doc_id AS vault_doc_id, d.deleted_at, "
    + ", ".join("{document_url}" if c == "document_url" else f"d.{c}"
                for c in _PAYLOAD_COLUMNS if c != "doc_id")
    + ", d.extra\n"
    "FROM rag_ingestions r\n"
    "LEFT JOIN documents d ON d.doc_id = r.doc_id\n"
    "WHERE r.collection = %(collection)s AND r.corpus = %(corpus)s{doc_ids}\n"
    "ORDER BY r.doc_id{limit}"
)


class SyncError(RuntimeError):
    """sync-metadata cannot go on, for a reason it names (a fatal error of the run)."""


@dataclass
class SyncStats:
    documents: int = 0
    actions: dict = field(default_factory=lambda: {a: 0 for a in ACTIONS})
    #: Documents whose points need a write (UPDATED, and UNDATED with a difference).
    to_write: int = 0
    #: Documents actually written (``--write``).
    written: int = 0
    #: Points the write reached (``set_payload``'s count).
    points_written: int = 0
    points_read: int = 0
    #: Documents whose points do not all carry the same owned values.
    disagreeing: list = field(default_factory=list)
    #: Per owned key, the number of documents where it changes.
    fields: dict = field(default_factory=dict)
    #: Documents whose point count changed between the read and the write.
    count_mismatch: list = field(default_factory=list)
    csv_rows: int = 0
    #: The ledger doc_ids examined.
    seen_ids: set = field(default_factory=set)
    indexes_missing: list = field(default_factory=list)
    indexes_created: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Expected values: the ingestion code path
# ---------------------------------------------------------------------------
def owned_keys(include_title: bool = False, include_links: bool = False,
               drop_keys: Sequence[str] | None = None) -> tuple[str, ...]:
    """The keys compared and written: :data:`OWNED_KEYS`, then ``title`` on
    request, then on request :data:`LINK_KEYS`, :data:`LEGACY_LINK_KEYS` and
    the keys of ``extra`` dropped by design (``drop_keys``; None ->
    ``payload.dropped_extra_keys``).
    A key named twice is owned once, at its first place."""
    keys = OWNED_KEYS + ((TITLE_KEY,) if include_title else ())
    if include_links:
        if drop_keys is None:
            drop_keys = dropped_extra_keys()
        keys = keys + LINK_KEYS + LEGACY_LINK_KEYS + tuple(drop_keys)
    return tuple(dict.fromkeys(keys))


def expected_values(doc_id: str, row: Mapping, keys: Sequence[str]) -> dict:
    """The owned keys an ingestion of ``row`` today would write on every point
    (absent keys left out): ``row_to_payload`` then ``build_payload``."""
    metadata, _dropped = row_to_payload(dict(row, doc_id=doc_id))
    payload = build_payload(doc_id, metadata, filename="", chunk_number=0,
                            ingestion_date="", text="", page=None)
    return {k: payload[k] for k in keys if payload.get(k) is not None}


def write_fields(doc_id: str, row: Mapping, include_title: bool = False,
                 include_links: bool = False) -> dict:
    """What ``set_payload`` is given for ``row``: ``date_fields`` with ``year``,
    ``date_kind`` and ``date_source`` (``None`` removes them) and, on request,
    ``title``; with ``include_links``, ``document_url``, ``source_url``,
    ``mime_type`` (``None`` removes them) and ``None`` for ``pdf_url`` (the
    former name of ``document_url``) and for each key of ``extra`` dropped by
    design (which removes it).

    Raises :class:`SyncError` if, once the ``None`` values are left out, it is
    not exactly :func:`expected_values`: the write must give the points the
    payload an ingestion would.
    """
    metadata, _dropped = row_to_payload(dict(row, doc_id=doc_id))
    fields = date_fields(metadata.get("date"), metadata.get("date_precision"),
                         year=metadata.get("year"))
    fields["date_kind"] = metadata.get("date_kind")
    fields["date_source"] = metadata.get("date_source")
    if include_title:
        fields[TITLE_KEY] = metadata.get(TITLE_KEY)
    if include_links:
        for key in owned_keys(include_title, include_links):
            if key not in fields:  # LINK_KEYS, LEGACY_LINK_KEYS, the dropped keys (absent: None)
                fields[key] = metadata.get(key)
    written = {k: v for k, v in fields.items() if v is not None}
    expected = expected_values(doc_id, row, owned_keys(include_title, include_links))
    if written != expected:
        raise SyncError(
            f"document {doc_id!r}: the write {written!r} is not what an ingestion writes "
            f"{expected!r}; nothing was written for it")
    return fields


def point_values(payload: Mapping | None, keys: Sequence[str]) -> dict:
    """The owned keys a point carries (a stored null counts as absent)."""
    payload = payload or {}
    return {k: payload[k] for k in keys if payload.get(k) is not None}


# ---------------------------------------------------------------------------
# Vault: the ledger and the rows
# ---------------------------------------------------------------------------
def ledger_rows(conn, corpus: str, collection: str, *, doc_ids: Sequence[str] | None = None,
                limit: int | None = None) -> list[dict]:
    """The documents ``rag_ingestions`` records for ``collection`` and ``corpus``,
    with their vault row (all ``None`` when the row is gone), in ``doc_id`` order.
    The document's address is read from ``vault.document_url_column`` (``pdf_url``
    on a vault before the rename) and always named ``document_url``."""
    params: dict = {"collection": collection, "corpus": corpus}
    doc_filter = ""
    if doc_ids:
        doc_filter = " AND r.doc_id = ANY(%(doc_ids)s)"
        params["doc_ids"] = list(doc_ids)
    limit_sql = ""
    if limit is not None:
        limit_sql = "\nLIMIT %(limit)s"
        params["limit"] = int(limit)
    sql = LEDGER_SQL.format(document_url=document_url_select(document_url_column(conn)),
                            doc_ids=doc_filter, limit=limit_sql)
    with conn.cursor() as cur:
        cur.execute(sql, params)
        columns = [d[0] for d in cur.description]
        rows = [dict(zip(columns, raw)) for raw in cur.fetchall()]
    conn.rollback()  # read only: end the transaction, keep the session (and its lock)
    return rows


# ---------------------------------------------------------------------------
# Qdrant: reads that name what went wrong
# ---------------------------------------------------------------------------
def _qdrant_error(exc: BaseException, collection: str) -> None:
    """Raise rag-engine's own error for a refused key or a strict-mode refusal."""
    status = getattr(exc, "status_code", None)
    if status in (401, 403):
        raise QdrantAuthError(
            f"Qdrant refused the request (HTTP {status}): check QDRANT_API_KEY") from exc
    if status == 400:
        content = getattr(exc, "content", b"")
        text = content.decode("utf-8", "replace") if isinstance(content, bytes) else str(content)
        if "Index required but not found" in text:
            raise MissingIndexError(
                f"collection {collection!r} refuses a filter on a field without a payload "
                f"index (strict mode, HTTP 400: {text.strip()[:300]})") from exc


def payload_schema(store, collection: str) -> dict:
    """The payload indexes of ``collection``: {field: kind}. A missing collection is a SyncError."""
    try:
        if not store.client.collection_exists(collection_name=collection):
            raise SyncError(f"collection {collection!r} does not exist on this Qdrant")
        schema = store.client.get_collection(collection_name=collection).payload_schema or {}
    except SyncError:
        raise
    except Exception as exc:
        _qdrant_error(exc, collection)
        raise
    out = {}
    for name, info in schema.items():
        kind = getattr(info, "data_type", None)
        out[name] = getattr(kind, "value", kind)
    return out


def read_points(store, collection: str, doc_ids: Sequence[str], keys: Sequence[str], *,
                page_size: int) -> dict[str, list[dict]]:
    """{doc_id: [owned values of each point]} for ``doc_ids``: scroll pages of
    ``page_size`` points through a ``doc_id`` match-any filter, the owned keys
    and ``doc_id`` only, never a vector. A document without points is absent."""
    selector = models.Filter(must=[models.FieldCondition(
        key="doc_id", match=models.MatchAny(any=list(doc_ids)))])
    wanted = set(doc_ids)
    found: dict[str, list[dict]] = {}
    offset = None
    while True:
        try:
            points, offset = store.client.scroll(
                collection_name=collection, scroll_filter=selector, limit=page_size,
                offset=offset, with_payload=["doc_id", *keys], with_vectors=False,
            )
        except Exception as exc:
            _qdrant_error(exc, collection)
            raise
        for point in points:
            payload = point.payload or {}
            doc_id = payload.get("doc_id")
            if doc_id in wanted:
                found.setdefault(doc_id, []).append(point_values(payload, keys))
        if offset is None:
            return found


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
def _cell(value) -> str:
    """A CSV cell: empty for an absent key, a non-empty string as it is, any
    other value as JSON (``2015``, ``""``, ``["2014-01-01", "2016-01-01"]``)."""
    if value is _ABSENT:
        return ""
    if isinstance(value, str) and value:
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


_ABSENT = object()


def compare(doc_id: str, expected: Mapping, points: Sequence[Mapping], keys: Sequence[str]
            ) -> tuple[list[tuple[str, object, object]], bool]:
    """The keys that differ on at least one point, as ``(field, qdrant, vault)``,
    and whether the points disagree among themselves. ``qdrant`` is the value
    (``_ABSENT`` when absent) when the points agree on that key, else the list
    of their distinct values (absent as ``None``), sorted by their JSON form."""
    distinct = {json.dumps(p, sort_keys=True, default=str) for p in points}
    disagree = len(distinct) > 1
    diffs = []
    for key in keys:
        want = expected.get(key, _ABSENT)
        seen = [p.get(key, _ABSENT) for p in points]
        if all(v == want and type(v) is type(want) for v in seen):
            continue
        values = []
        for v in seen:
            if not any(v is u or (v == u and type(v) is type(u)) for u in values):
                values.append(v)
        got = values[0] if len(values) == 1 else sorted(
            (None if v is _ABSENT else v for v in values),
            key=lambda v: json.dumps(v, sort_keys=True, default=str))
        diffs.append((key, got, want))
    return diffs, disagree


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------
def _batches(items: Sequence, size: int) -> Iterable[Sequence]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def run_sync(
    conn,
    store,
    *,
    corpus: str,
    collection: str,
    csv_path: Path,
    write: bool = False,
    include_title: bool = False,
    include_links: bool = False,
    doc_ids: Sequence[str] | None = None,
    limit: int | None = None,
    page_size: int = DEFAULT_PAGE_SIZE,
    doc_batch: int = DEFAULT_DOC_BATCH,
    create_indexes: bool = True,
    on_progress: Callable[[SyncStats], None] | None = None,
    guard: Callable[[], None] | None = None,
    schema: Mapping | None = None,
) -> SyncStats:
    """Compare (and with ``write``, correct) the owned keys of every ledger document.

    ``conn`` is the vault connection (holding the collection lock), ``store`` a
    ``rag_engine.store.QdrantStore``. With ``write``, the payload indexes of
    rag-engine (``PAYLOAD_INDEXES``: on a collection indexed before payload
    schema v2, ``date_start`` and ``date_end``) are created first, unless
    ``create_indexes`` is False; the rag-engine README states that creating
    them before the points have their interval is safe. Without ``write``
    nothing is written anywhere but ``csv_path``.

    ``schema`` is the collection's :func:`payload_schema` when the caller has
    already read it (it is then not read again).

    ``guard()`` is called before each batch is read and before each document
    is written (the CLI checks there that the vault session still holds the
    collection lock); an exception it raises ends the run there, every
    document before it being either corrected or not.
    """
    for name, value in (("page_size", page_size), ("doc_batch", doc_batch)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be an integer >= 1, got {value!r}")
    keys = owned_keys(include_title, include_links)
    stats = SyncStats()

    from rag_engine.schema import PAYLOAD_INDEXES

    if schema is None:
        schema = payload_schema(store, collection)
    stats.indexes_missing = [f for f, _kind in PAYLOAD_INDEXES if f not in schema]
    if write and create_indexes and stats.indexes_missing:
        stats.indexes_created = store.ensure_payload_indexes(collection)

    rows = ledger_rows(conn, corpus, collection, doc_ids=doc_ids, limit=limit)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for batch in _batches(rows, doc_batch):
            if guard is not None:
                guard()
            ids = [r["ledger_doc_id"] for r in batch]
            points = read_points(store, collection, ids, keys, page_size=page_size)
            for row in batch:
                out = _one(row, points.get(row["ledger_doc_id"], []), keys, stats, store,
                           collection, write=write, include_title=include_title,
                           include_links=include_links, guard=guard)
                for rec in out:
                    writer.writerow(rec)
                stats.csv_rows += len(out)
                if on_progress is not None:
                    on_progress(stats)
            fh.flush()
    return stats


def _one(row, points, keys, stats: SyncStats, store, collection, *, write, include_title,
         include_links=False, guard=None):
    doc_id = row["ledger_doc_id"]
    stats.documents += 1
    stats.seen_ids.add(doc_id)
    stats.points_read += len(points)
    if row.get("vault_doc_id") is None or row.get("deleted_at") is not None:
        stats.actions[MISSING_IN_VAULT] += 1
        note = "deleted in the vault (deleted_at set)" if row.get("vault_doc_id") else \
            "no documents row"
        return [{"doc_id": doc_id, "action": MISSING_IN_VAULT, "field": "", "qdrant_value": "",
                 "vault_value": "", "note": f"{note}; {len(points)} point(s) left as they are"}]
    if not points:
        stats.actions[MISSING_IN_QDRANT] += 1
        return [{"doc_id": doc_id, "action": MISSING_IN_QDRANT, "field": "", "qdrant_value": "",
                 "vault_value": "", "note": "recorded in rag_ingestions, no point in the collection"}]
    vault_row = {k: v for k, v in row.items()
                 if k not in ("ledger_doc_id", "vault_doc_id", "deleted_at")}
    expected = expected_values(doc_id, vault_row, keys)
    diffs, disagree = compare(doc_id, expected, points, keys)
    undated = "date" not in expected
    action = UNDATED if undated else (UPDATED if diffs else UNCHANGED)
    stats.actions[action] += 1
    if disagree:
        stats.disagreeing.append(doc_id)
    if not diffs:
        return []
    fields = write_fields(doc_id, vault_row, include_title, include_links)  # checked first
    stats.to_write += 1
    for key, _got, _want in diffs:
        stats.fields[key] = stats.fields.get(key, 0) + 1
    note = f"points disagree ({len(points)} points)" if disagree else ""
    if write:
        if guard is not None:
            guard()
        reached = store.set_payload(collection, doc_id, fields)
        stats.written += 1
        stats.points_written += reached
        if reached != len(points):
            stats.count_mismatch.append((doc_id, len(points), reached))
            note = (note + "; " if note else "") + \
                f"{len(points)} point(s) read, {reached} written"
    return [{"doc_id": doc_id, "action": action, "field": key, "qdrant_value": _cell(got),
             "vault_value": _cell(want), "note": note} for key, got, want in diffs]


def print_summary(stats: SyncStats, *, write: bool, elapsed: float | None = None,
                  csv_path: Path | None = None, out=None) -> None:
    import sys

    out = out or sys.stdout
    print("─" * 60, file=out)
    print("Done." if write else "Done (dry run: nothing was written to Qdrant).", file=out)
    print(f"  documents          : {stats.documents}", file=out)
    for action in ACTIONS:
        print(f"  {action:<19}: {stats.actions[action]}", file=out)
    if write:
        print(f"  documents written  : {stats.written}", file=out)
        print(f"  points written     : {stats.points_written}", file=out)
    else:
        print(f"  documents to write : {stats.to_write}", file=out)
    print(f"  points read        : {stats.points_read}", file=out)
    print(f"  points disagreeing : {len(stats.disagreeing)} document(s)", file=out)
    if stats.fields:
        changed = ", ".join(f"{k} x{n}" for k, n in sorted(stats.fields.items(),
                                                            key=lambda kv: OWNED_KEYS.index(kv[0])
                                                            if kv[0] in OWNED_KEYS else 99))
        print(f"  keys changing      : {changed}", file=out)
    if elapsed is not None:
        print(f"  time               : {elapsed:.1f}s", file=out)
    if csv_path is not None:
        print(f"  CSV                : {csv_path} ({stats.csv_rows} row(s))", file=out)
    for doc_id in stats.disagreeing[:10]:
        print(f"warning: the points of {doc_id!r} do not all carry the same values; "
              + ("all of them were corrected" if write else "--write corrects all of them"),
              file=sys.stderr)
    for doc_id, read, reached in stats.count_mismatch[:10]:
        print(f"warning: {doc_id!r}: {read} point(s) read, {reached} written: was the document "
              "ingested meanwhile? Run sync-metadata again once no ingestion runs", file=sys.stderr)


def default_csv_path(state_dir: Path, collection: str) -> Path:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return state_dir / f"sync-metadata-{collection}-{stamp}.csv"
