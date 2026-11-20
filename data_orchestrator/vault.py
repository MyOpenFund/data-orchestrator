"""Vault (Postgres) plumbing: connection + the rag_ingestions-backed ledger.

The vault owns the schema (its ingestion service runs the DDL train); this
module only reads/writes through the documented contract. Write protocol:
``VaultLedger.mark`` is called by ``run_ingest`` strictly AFTER the document's
Qdrant points landed — a crash in between leaves the vault under-claiming,
which the next resume pass heals (deterministic point ids overwrite in place).
"""
from __future__ import annotations

import json
import os
import weakref

import psycopg2

from .config import load_dotenv
from .core import FatalRunError

# The vault's identity of a document is (corpus, doc_id) since the documents
# contract (vault 12ffd11): two corpora may both have a doc_id. rag_ingestions
# is keyed (corpus, doc_id, collection), its foreign key is (corpus, doc_id).
RESUME_SQL = "SELECT doc_id FROM rag_ingestions WHERE collection = %s AND corpus = %s"

# The `runs` columns this writer fills, in INSERT order. Single source of
# truth: the SQL below is built from it, and `insert_run_report` sweeps every
# report key that is NOT in it into `extra` (JSONB). Mirrors the vault
# ingester's `KNOWN_FIELDS` (`vault/ingestion/ingest_runs.py`). `corpus`
# (vault `runs.corpus`, issue #18) is the resolved `--corpus`, the same value
# as `rag_ingestions.corpus`.
_RUN_COLUMNS = (
    "run_id", "tool", "command", "started_at", "finished_at",
    "outcome", "exit_code", "totals", "sources", "corpus",
)

# Columns whose report value must be JSON-encoded before it hits the driver.
_JSON_COLUMNS = frozenset({"totals", "sources"})

INSERT_RUN_SQL = """
INSERT INTO runs ({columns}, extra)
VALUES ({placeholders}, %s)
ON CONFLICT (run_id) DO NOTHING
""".format(
    columns=", ".join(_RUN_COLUMNS),
    placeholders=", ".join(["%s"] * len(_RUN_COLUMNS)),
)

UPSERT_INGESTION_SQL = """
INSERT INTO rag_ingestions (
    doc_id, collection, corpus, source_code,
    embedding_model, embedding_version, chunk_count
) VALUES (%s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (corpus, doc_id, collection) DO UPDATE SET
    source_code = EXCLUDED.source_code,
    embedding_model = EXCLUDED.embedding_model,
    embedding_version = EXCLUDED.embedding_version,
    chunk_count = EXCLUDED.chunk_count,
    ingested_at = now();
"""


# TCP keepalives of the vault connection (libpq ``keepalives_idle``,
# ``keepalives_interval``, ``keepalives_count``): a run holds its session, and
# the collection lock with it, for hours while it mostly talks to Qdrant; a
# peer that went away (a laptop asleep, a dropped tailnet link) is then
# noticed by the operating system within idle + interval x count seconds
# instead of never. Environment variable -> (libpq parameter, default).
KEEPALIVE_ENV = {
    "DATA_ORCHESTRATOR_PG_KEEPALIVES_IDLE": ("keepalives_idle", 30),
    "DATA_ORCHESTRATOR_PG_KEEPALIVES_INTERVAL": ("keepalives_interval", 10),
    "DATA_ORCHESTRATOR_PG_KEEPALIVES_COUNT": ("keepalives_count", 3),
}


# Keepalives only probe an IDLE connection. A request sent into a dead link
# (a partition: data sent and never acknowledged) waits instead for TCP's
# retransmissions, about 15 minutes on Linux, and a lock check or a ledger
# write would hang that long. libpq's ``tcp_user_timeout`` bounds it: the
# connection is closed once data stays unacknowledged that many milliseconds.
# libpq applies it only where the system has TCP_USER_TIMEOUT (Linux); on
# macOS it is accepted and has no effect, and the keepalives and the
# system's own retransmission limit apply. 0 leaves the system default.
# A server-side statement_timeout cannot help here: on a dead link the
# server never receives the statement.
TCP_USER_TIMEOUT_ENV = "DATA_ORCHESTRATOR_PG_TCP_USER_TIMEOUT_MS"
DEFAULT_TCP_USER_TIMEOUT_MS = 30000


def keepalive_options(environ=None) -> dict:
    """The connection parameters :func:`connect` passes to libpq for a dead
    peer: the keepalives (always on) and ``tcp_user_timeout``.

    A keepalive value is a whole number of seconds (or probes) >= 1, the
    timeout a whole number of milliseconds >= 0, else ``ValueError`` naming
    the variable; unset is the default.
    """
    environ = os.environ if environ is None else environ
    options = {"keepalives": 1}
    for env, (param, default) in KEEPALIVE_ENV.items():
        raw = environ.get(env)
        if raw is None:
            options[param] = default
            continue
        text = raw.strip()
        if not (text.isascii() and text.isdigit() and int(text) >= 1):
            raise ValueError(f"{env}={raw!r} is not a whole number >= 1")
        options[param] = int(text)
    raw = environ.get(TCP_USER_TIMEOUT_ENV)
    if raw is None:
        options["tcp_user_timeout"] = DEFAULT_TCP_USER_TIMEOUT_MS
    else:
        text = raw.strip()
        if not (text.isascii() and text.isdigit()):
            raise ValueError(f"{TCP_USER_TIMEOUT_ENV}={raw!r} is not a whole number of "
                             "milliseconds >= 0")
        options["tcp_user_timeout"] = int(text)
    return options


def connect():
    """Connect to the vault Postgres (DATABASE_URL from .env / environment),
    with TCP keepalives on (:func:`keepalive_options`).

    Runs ``config.load_dotenv()`` first (a no-op once loaded): a library
    caller whose ``DATA_ORCHESTRATOR_DOTENV`` names a missing file gets
    ``FileNotFoundError`` here.
    """
    load_dotenv()
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set (.env or environment). Set it, or run "
            "with --no-vault to use the local file ledger without vault state."
        )
    return psycopg2.connect(url, application_name=application_name(), **keepalive_options())


def application_name() -> str:
    """``data-orchestrator@<host>``: how this machine's sessions read in
    ``pg_stat_activity`` (and in a refusal naming the holder of a lock)."""
    import socket

    return f"data-orchestrator@{socket.gethostname()}"[:63]


# ---------------------------------------------------------------------------
# Transition: the column of the document's address, ``document_url`` or ``pdf_url``
# ---------------------------------------------------------------------------
#: The column of the document's own address. The vault renamed ``pdf_url`` to
#: ``document_url`` (2026-11, its gated DDL train); until every vault this
#: package talks to has run that DDL, the selection reads whichever exists.
DOCUMENT_URL_COLUMN = "document_url"
#: Its former name, read when the vault has not renamed it yet. Transition
#: only: to remove with :func:`document_url_column` once every vault has run
#: the rename.
LEGACY_DOCUMENT_URL_COLUMN = "pdf_url"

DOCUMENT_URL_COLUMN_SQL = (
    "SELECT column_name FROM information_schema.columns\n"
    "WHERE table_name = 'documents' AND table_schema = ANY (current_schemas(false))\n"
    "  AND column_name IN (%(new)s, %(old)s)"
)

# conn -> the column found on it: the schema is read once per connection.
_DOCUMENT_URL_COLUMNS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def document_url_column(conn) -> str:
    """The name of the ``documents`` column holding the document's address on
    the vault behind ``conn``: ``document_url``, else (a vault that has not run
    the rename yet) ``pdf_url``. Neither found (no ``documents`` table in the
    search path) -> ``document_url``, so the selection fails with Postgres's
    own error naming it. Read once per connection (``information_schema``),
    then remembered for that connection.
    """
    try:
        return _DOCUMENT_URL_COLUMNS[conn]
    except (KeyError, TypeError):
        pass
    with conn.cursor() as cur:
        cur.execute(DOCUMENT_URL_COLUMN_SQL,
                    {"new": DOCUMENT_URL_COLUMN, "old": LEGACY_DOCUMENT_URL_COLUMN})
        found = {row[0] for row in cur.fetchall()}
    column = (LEGACY_DOCUMENT_URL_COLUMN
              if LEGACY_DOCUMENT_URL_COLUMN in found and DOCUMENT_URL_COLUMN not in found
              else DOCUMENT_URL_COLUMN)
    try:
        _DOCUMENT_URL_COLUMNS[conn] = column
    except TypeError:  # a connection without weak references: read again next time
        pass
    return column


def document_url_select(column: str, table: str = "d") -> str:
    """The select-list entry of the document's address, always named
    ``document_url``: ``d.document_url``, or ``d.pdf_url AS document_url``."""
    if column == DOCUMENT_URL_COLUMN:
        return f"{table}.{DOCUMENT_URL_COLUMN}"
    if column == LEGACY_DOCUMENT_URL_COLUMN:
        return f"{table}.{LEGACY_DOCUMENT_URL_COLUMN} AS {DOCUMENT_URL_COLUMN}"
    raise ValueError(f"not a column of the document's address: {column!r}")


def insert_run_report(conn, report: dict) -> None:
    """Write one run-report row to the vault's ``runs`` table.

    Best-effort by contract: the caller (cli._build_report's consumers)
    tolerates a raising insert with a warning rather than letting it mask
    the run's own outcome.

    Report keys outside ``_RUN_COLUMNS`` (notably ``fatal``, set by
    ``cli._fatal_report``, and the vault source's ``unsupported_by_source``)
    are swept into ``extra``; an empty sweep writes SQL
    NULL rather than ``'{}'``, so ``extra IS NULL`` means the same thing here
    as in the vault ingester.

    A report missing one of ``_RUN_COLUMNS`` (e.g. no ``totals``/``sources``
    on an early failure) writes SQL NULL for that column, same as an absent
    key does in the vault ingester — it never raises ``KeyError``.
    """
    extra = {k: v for k, v in report.items() if k not in _RUN_COLUMNS}

    def _param(c):
        v = report.get(c)
        return json.dumps(v) if c in _JSON_COLUMNS and v is not None else v

    params = tuple(_param(c) for c in _RUN_COLUMNS)
    with conn.cursor() as cur:
        cur.execute(INSERT_RUN_SQL, params + (
            json.dumps(extra, default=str) if extra else None,
        ))
    conn.commit()


#: SQLSTATE invalid_column_reference: Postgres's answer to an ON CONFLICT
#: naming columns no unique index covers.
ON_CONFLICT_WITHOUT_KEY = "42P10"


class VaultLedger:
    """``Ledger``-compatible resume/record state backed by ``rag_ingestions``,
    for one corpus and one collection: a document is ``(corpus, doc_id)``."""

    def __init__(self, conn, collection: str, corpus: str,
                 embedding_model: str, embedding_version: str):
        self._conn = conn
        self.collection = collection
        self.corpus = corpus
        self.embedding_model = embedding_model
        self.embedding_version = embedding_version
        with conn.cursor() as cur:
            cur.execute(RESUME_SQL, (collection, corpus))
            self._done = {row[0] for row in cur.fetchall()}

    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self._done

    def __len__(self) -> int:
        return len(self._done)

    def mark(self, doc_id: str, chunks: int, payload: dict | None = None) -> None:
        payload = payload or {}
        source_code = payload.get("source_code")
        try:
            with self._conn.cursor() as cur:
                cur.execute(UPSERT_INGESTION_SQL, (
                    doc_id, self.collection, self.corpus, source_code,
                    self.embedding_model, self.embedding_version, chunks,
                ))
            self._conn.commit()
        except Exception as exc:
            if getattr(self._conn, "closed", 0):
                # The session itself is gone (and the collection lock with
                # it): every later document would fail the same way.
                raise VaultSessionLost(
                    f"the vault session was lost while recording document {doc_id!r} "
                    f"({type(exc).__name__}: {str(exc).strip()}); its points are written, "
                    "the next run writes them again") from exc
            if getattr(exc, "pgcode", None) == ON_CONFLICT_WITHOUT_KEY:
                # No unique key (corpus, doc_id, collection): a vault before the
                # documents contract. Every document would fail the same way.
                self._conn.rollback()
                raise VaultSchemaTooOld(
                    "rag_ingestions has no key (corpus, doc_id, collection): this vault "
                    "predates the documents contract (vault 12ffd11). Deploy the vault and "
                    f"run its ingestion first; document {doc_id!r} has its points written, "
                    "the next run writes them again") from exc
            # A raising execute (e.g. an FK violation for a doc_id unknown to
            # `documents`) leaves this shared connection in aborted-transaction
            # state: every later statement on it would raise
            # InFailedSqlTransaction until the abort is cleared, silently
            # blocking convergence for every document processed after this
            # one. Roll back so the connection is healthy again, then
            # propagate — run_ingest's per-document isolation still counts
            # this one as a docs_error and moves on cleanly.
            self._conn.rollback()
            raise
        self._done.add(doc_id)


# ---------------------------------------------------------------------------
# Collection lock: no sync-metadata alongside an ingestion of the same
# collection (rag-engine's set_payload must not run beside an ingestion of the
# same document: a batch written after the correction puts the old metadata
# back). A Postgres session-level advisory lock on (COLLECTION_LOCK_CLASS,
# hashtext(collection)), held by the vault connection of the run and released
# when it closes (or when the process dies). Ingestion runs and sync-metadata
# dry runs take it SHARED, so they never exclude each other; sync-metadata
# --write takes it EXCLUSIVE.
# Diagnosis: SELECT * FROM pg_locks WHERE locktype = 'advisory' AND classid = 7468;
# ---------------------------------------------------------------------------
COLLECTION_LOCK_CLASS = 7468

_LOCK_SQL = {
    False: "SELECT pg_try_advisory_lock_shared(%s, hashtext(%s))",
    True: "SELECT pg_try_advisory_lock(%s, hashtext(%s))",
}


class CollectionBusyError(RuntimeError):
    """The collection lock is held by a run that excludes this one."""


class VaultSchemaTooOld(FatalRunError):
    """The vault predates the documents contract: ``rag_ingestions`` has no
    key ``(corpus, doc_id, collection)``. The run stops; deploy the vault."""


class VaultSessionLost(FatalRunError):
    """The vault session of the run is gone, or no longer holds the collection
    lock (``pg_terminate_backend``, a dropped network link, a server restart):
    another run may now take the lock. The run stops; it is resumable."""


_HOLDS_SQL = (
    "SELECT count(*) FROM pg_locks WHERE locktype = 'advisory' AND granted "
    "AND pid = pg_backend_pid() AND classid = %s "
    "AND objid = (hashtext(%s)::bigint & 4294967295)::oid AND objsubid = 2 AND mode = %s"
)


def check_collection_lock(conn, collection: str, *, exclusive: bool, holder: str) -> None:
    """Check, on the run's own session, that it is alive and still holds the
    collection lock taken by :func:`lock_collection`; raise
    :class:`VaultSessionLost` otherwise. One small query; the transaction it
    opens is ended at once (the lock is a session lock, it stays)."""
    mode = "ExclusiveLock" if exclusive else "ShareLock"
    try:
        with conn.cursor() as cur:
            cur.execute(_HOLDS_SQL, (COLLECTION_LOCK_CLASS, collection, mode))
            (held,) = cur.fetchone()
        conn.rollback()
    except Exception as exc:  # noqa: BLE001 — any failure of this query means the session is unusable
        raise VaultSessionLost(
            f"{holder}: the vault session holding the lock of collection {collection!r} "
            f"is gone ({type(exc).__name__}: {str(exc).strip()}); the run stops here and "
            "can be run again") from exc
    if not held:
        raise VaultSessionLost(
            f"{holder}: the vault session no longer holds the lock of collection "
            f"{collection!r}; the run stops here and can be run again")


def lock_collection(conn, collection: str, *, exclusive: bool, holder: str) -> None:
    """Take the collection lock on ``conn`` without waiting, or raise
    :class:`CollectionBusyError`. ``holder`` names the run asking, for the message.

    The lock lives as long as the session: ``conn.close()`` releases it, a
    commit or a rollback does not.
    """
    with conn.cursor() as cur:
        cur.execute(_LOCK_SQL[exclusive], (COLLECTION_LOCK_CLASS, collection))
        (acquired,) = cur.fetchone()
    conn.commit()
    if acquired:
        return
    if exclusive:
        other = "an ingestion (or another sync-metadata run)"
    else:
        other = "a sync-metadata --write run"
    holders, pids = _describe_holders(conn, collection)
    raise CollectionBusyError(
        f"{holder}: {other} holds collection {collection!r} (vault advisory lock "
        f"{COLLECTION_LOCK_CLASS}{'; ' + holders if holders else ''}); wait for it to end, then "
        "run again. If no such run is alive any more (a machine asleep, a lost network: the "
        "server keeps a dead session, and its lock, until its own TCP keepalive ends it), "
        "check that its process is gone, then end the session with the vault's admin role: "
        + (" ".join(f"SELECT pg_terminate_backend({pid});" for pid in pids)
           or "SELECT pg_terminate_backend(<pid>);"))


_HOLDERS_SQL = (
    "SELECT l.pid, l.mode, a.usename, a.application_name, host(a.client_addr), a.backend_start "
    "FROM pg_locks l LEFT JOIN pg_stat_activity a ON a.pid = l.pid "
    "WHERE l.locktype = 'advisory' AND l.granted AND l.classid = %s "
    "AND l.objid = (hashtext(%s)::bigint & 4294967295)::oid AND l.objsubid = 2 "
    "AND l.database = (SELECT oid FROM pg_database WHERE datname = current_database()) "
    "AND l.pid <> pg_backend_pid() ORDER BY l.pid"
)


def lock_holders(conn, collection: str) -> list[tuple]:
    """The other sessions of this database holding the collection lock:
    ``(pid, mode, user, application, client address, session start)``. A role
    that may not read another role's session details gets ``None`` for them
    (``pg_stat_activity`` hides them); ``[]`` when the query itself fails."""
    try:
        with conn.cursor() as cur:
            cur.execute(_HOLDERS_SQL, (COLLECTION_LOCK_CLASS, collection))
            rows = list(cur.fetchall())
        conn.rollback()
        return rows
    except Exception:  # noqa: BLE001 — a description only: never mask the refusal
        try:
            conn.rollback()
        except Exception:  # noqa: BLE001
            pass
        return []


def _describe_holders(conn, collection: str) -> tuple[str, list]:
    """The holders of the collection lock, as text for a refusal, and their pids."""
    rows = lock_holders(conn, collection)
    try:
        return _holder_text(rows), [row[0] for row in rows]
    except Exception:  # noqa: BLE001 — a description only: never mask the refusal
        return "", []


def _holder_text(rows) -> str:
    """A role without ``pg_read_all_stats`` sees another role's user and
    application, but not its client address nor its start (NULL)."""
    parts = []
    for pid, mode, user, app, client, started in rows:
        details = [mode]
        if user is not None:
            details.append(f"user {user}")
        if app:
            details.append(f"application {app}")
        if started is None:
            details.append("its client and start are not visible to this role")
        else:
            when = started.isoformat(timespec="seconds") if hasattr(started, "isoformat") else started
            details.append(f"client {client or 'local socket'}, session started {when}")
        parts.append(f"held by pid {pid} ({', '.join(details)})")
    return "; ".join(parts)
