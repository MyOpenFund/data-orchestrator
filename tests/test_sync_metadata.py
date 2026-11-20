"""sync-metadata against fakes: a scripted vault connection and qdrant-client's
local mode behind a real ``rag_engine.store.QdrantStore``."""
from __future__ import annotations

import csv
import datetime
import json

import pytest
from qdrant_client import QdrantClient, models
from rag_engine.config import Settings
from rag_engine.schema import build_payload, point_id
from rag_engine.store import MissingIndexError, QdrantAuthError, QdrantStore

from data_orchestrator import cli
from data_orchestrator import sync_metadata as sm

# qdrant-client's local mode keeps no payload index and warns when asked for one
# (a --write creates the engine's indexes first).
pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect:UserWarning")

COLLECTION = "c-test"
CORPUS = "central-bank"
KEYS = sm.OWNED_KEYS


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class LedgerCursor:
    def __init__(self, conn):
        self._conn = conn
        self._rows = []
        self.description = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, sql, params=None):
        self._conn.executed.append((" ".join(sql.split()), params))
        if "pg_try_advisory_lock" in sql:
            self._rows = [(self._conn.lock_free,)]
            return
        if "pg_stat_activity" in sql:
            self._rows = []
            return
        if "FROM pg_locks" in sql:
            self._rows = [(int(self._conn.lock_held),)]
            return
        if "information_schema.columns" in sql:
            self._rows = [(c,) for c in self._conn.columns
                          if c in (params["new"], params["old"])]
            return
        assert "FROM rag_ingestions r" in sql
        assert params["collection"] == COLLECTION and params["corpus"] == CORPUS
        rows = sorted(self._conn.ledger, key=lambda r: r["ledger_doc_id"])
        if "doc_ids" in params:
            rows = [r for r in rows if r["ledger_doc_id"] in params["doc_ids"]]
        if "limit" in params:
            rows = rows[:params["limit"]]
        columns = [c.strip().split(" AS ")[-1].split(".")[-1]
                   for c in sql.split("SELECT", 1)[1].split("FROM rag_ingestions")[0].split(",")]
        self.description = [(c,) for c in columns]
        self._rows = [tuple(r.get(c) for c in columns) for r in rows]

    def fetchall(self):
        return list(self._rows)

    def fetchone(self):
        return self._rows[0]


class LedgerConn:
    """The vault: ``ledger`` holds one dict per rag_ingestions row, joined
    with its documents row (``vault_doc_id`` None when the row is gone)."""

    def __init__(self, ledger=(), lock_free=True, columns=("document_url",)):
        self.ledger = list(ledger)
        #: The columns of ``documents`` that information_schema reports (the
        #: document's address: ``document_url``, or ``pdf_url`` before the rename).
        self.columns = tuple(columns)
        self.lock_free = lock_free
        self.lock_held = True
        self.executed = []
        self.closed = False

    def cursor(self):
        return LedgerCursor(self)

    def commit(self):
        pass

    def rollback(self):
        pass

    def close(self):
        self.closed = True


def vault_row(doc_id, *, date=datetime.date(2016, 1, 1), year=2016, extra=None, title="T",
              deleted=False, gone=False, **cols):
    row = {"ledger_doc_id": doc_id, "vault_doc_id": None if gone else doc_id,
           "deleted_at": datetime.datetime(2026, 1, 1) if deleted else None,
           "corpus": CORPUS, "source_code": "gb", "doc_type": "A3", "title": title,
           "date": date, "year": year, "language": "en", "sha256": "ff",
           "provenance": "manifest", "extra": extra}
    row.update(cols)
    return row


def make_store():
    client = QdrantClient(":memory:")
    store = QdrantStore(client=client, settings=Settings())
    store.ensure_collection(COLLECTION, 4)
    return store


def v1_payload(doc_id, chunk, *, date="2016-01-01", year=2016, title="T", **extra):
    """A point as ingestion wrote it before schema v2 (no interval)."""
    meta = {"corpus": CORPUS, "source_code": "gb", "doc_type": "A3", "title": title,
            "date": date, "year": year, "language": "en", "sha256": "ff",
            "provenance": "manifest", **extra}
    meta = {k: v for k, v in meta.items() if v is not None}
    payload = build_payload(doc_id, meta, filename=f"{doc_id}.pdf", chunk_number=chunk,
                            ingestion_date="2026-10-01T00:00:00", text=f"text {chunk}",
                            page=chunk)
    payload.pop("date_start", None)
    payload.pop("date_end", None)
    return payload


def seed(store, doc_id, n=3, **kw):
    points = []
    for chunk in range(n):
        payload = v1_payload(doc_id, chunk, **kw)
        vec = [0.1 * (chunk + 1), 0.2, -0.3, 0.4 + len(doc_id) / 100]
        points.append(models.PointStruct(id=point_id(doc_id, chunk, chunk), vector=vec,
                                         payload=payload))
    store.client.upsert(COLLECTION, points=points, wait=True)


def snapshot(store):
    points, _ = store.client.scroll(COLLECTION, limit=10_000, with_payload=True, with_vectors=True)
    return {p.id: (dict(p.payload), list(p.vector)) for p in points}


def run(conn, store, tmp_path, **kw):
    path = tmp_path / kw.pop("csv_name", "out.csv")
    stats = sm.run_sync(conn, store, corpus=CORPUS, collection=COLLECTION, csv_path=path, **kw)
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return stats, rows


# ---------------------------------------------------------------------------
# Expected values: the ingestion code path
# ---------------------------------------------------------------------------
def test_expected_values_are_what_ingestion_writes():
    row = vault_row("d", date=datetime.date(2016, 7, 1), year=2016,
                    extra={"date_precision": "month", "date_kind": "meeting",
                           "date_start": "1999-01-01T00:00:00Z", "date_source": "title"})
    assert sm.expected_values("d", row, KEYS) == {
        "date": "2016-07-01", "year": 2016, "date_precision": "month", "date_kind": "meeting",
        "date_source": "title",
        "date_start": "2016-07-01T00:00:00Z", "date_end": "2016-07-31T00:00:00Z"}


@pytest.mark.parametrize("date,year,extra", [
    (datetime.date(2016, 7, 13), 2016, {"date_precision": "day", "date_kind": "meeting"}),
    (datetime.date(2016, 7, 1), 2016, {"date_precision": "month"}),
    (datetime.date(2016, 1, 1), 2016, {"date_precision": "year", "date_kind": "period"}),
    (datetime.date(2015, 12, 17), 2015, None),
    (datetime.date(2016, 7, 13), 2016, {"date_precision": "garbage"}),
    (datetime.date(2016, 7, 13), 2016, {"date_precision": 7, "date_kind": ["x"]}),
    (None, None, {"date_precision": "month", "date_kind": "meeting"}),
    (None, 2016, None),
])
def test_the_write_is_exactly_the_expected_values(date, year, extra):
    row = vault_row("d", date=date, year=year, extra=extra)
    fields = sm.write_fields("d", row)
    assert set(fields) == {"date", "date_precision", "date_start", "date_end", "year", "date_kind",
                           "date_source"}
    assert {k: v for k, v in fields.items() if v is not None} == sm.expected_values("d", row, KEYS)


def test_a_garbage_precision_is_read_as_a_day():
    row = vault_row("d", date=datetime.date(2016, 7, 13), extra={"date_precision": "Month"})
    exp = sm.expected_values("d", row, KEYS)
    assert exp["date_precision"] == "Month"
    assert exp["date_start"] == exp["date_end"] == "2016-07-13T00:00:00Z"


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------
def test_dry_run_reports_and_writes_nothing(tmp_path):
    store = make_store()
    seed(store, "gb-1")  # placeholder 2016-01-01, v1
    conn = LedgerConn([vault_row("gb-1", date=datetime.date(2015, 12, 17), year=2015,
                                 extra={"date_precision": "day", "date_kind": "meeting"})])
    before = snapshot(store)
    stats, rows = run(conn, store, tmp_path)
    assert snapshot(store) == before
    assert stats.actions[sm.UPDATED] == 1 and stats.to_write == 1 and stats.written == 0
    assert stats.points_read == 3
    assert {r["field"]: (r["qdrant_value"], r["vault_value"]) for r in rows} == {
        "date": ("2016-01-01", "2015-12-17"),
        "year": ("2016", "2015"),
        "date_precision": ("", "day"),
        "date_kind": ("", "meeting"),
        "date_start": ("", "2015-12-17T00:00:00Z"),
        "date_end": ("", "2015-12-17T00:00:00Z"),
    }
    assert {r["action"] for r in rows} == {sm.UPDATED}
    # The vault is only read: the column of the document's address, then the
    # ledger (the CLI takes the lock).
    assert [sql.split()[0] for sql, _ in conn.executed] == ["SELECT", "SELECT"]
    assert "information_schema.columns" in conn.executed[0][0]


def test_write_changes_only_the_owned_keys_and_a_second_run_writes_nothing(tmp_path):
    store = make_store()
    seed(store, "gb-1")
    seed(store, "gb-2", n=2, date="2010-03-04", year=2010)
    conn = LedgerConn([
        vault_row("gb-1", date=datetime.date(2015, 12, 17), year=2015,
                  extra={"date_precision": "day", "date_kind": "meeting"}),
        vault_row("gb-2", date=datetime.date(2010, 3, 4), year=2010),
    ])
    before = snapshot(store)
    stats, _rows = run(conn, store, tmp_path, write=True)
    assert stats.written == 2 and stats.points_written == 5
    after = snapshot(store)
    assert set(after) == set(before)
    for pid, (payload, vector) in after.items():
        old_payload, old_vector = before[pid]
        assert vector == old_vector
        changed = {k for k in set(payload) | set(old_payload) if payload.get(k) != old_payload.get(k)}
        assert changed <= set(KEYS)
    gb1 = [p for p, _ in after.values() if p["doc_id"] == "gb-1"]
    assert all(p["date"] == "2015-12-17" and p["year"] == 2015 and p["date_kind"] == "meeting"
               and p["date_start"] == p["date_end"] == "2015-12-17T00:00:00Z" for p in gb1)
    gb2 = [p for p, _ in after.values() if p["doc_id"] == "gb-2"]
    assert all("date_precision" not in p and "date_kind" not in p
               and p["date_start"] == "2010-03-04T00:00:00Z" for p in gb2)

    # A write never writes the vault (rag_ingestions included): reads only.
    assert all(sql.startswith("SELECT") for sql, _ in conn.executed)

    again, rows = run(conn, store, tmp_path, write=True, csv_name="again.csv")
    assert again.written == 0 and again.to_write == 0 and rows == []
    assert again.actions[sm.UNCHANGED] == 2
    assert snapshot(store) == after


def test_points_that_disagree_are_reported_and_all_corrected(tmp_path):
    store = make_store()
    seed(store, "gb-1", n=3, date="2016-01-01")
    # One point from an older ingestion with another date.
    store.client.set_payload(COLLECTION, payload={"date": "2014-01-01", "year": 2014},
                             points=[point_id("gb-1", 2, 2)], wait=True)
    conn = LedgerConn([vault_row("gb-1", date=datetime.date(2016, 1, 1), year=2016)])
    stats, rows = run(conn, store, tmp_path)
    assert stats.disagreeing == ["gb-1"]
    date_row = [r for r in rows if r["field"] == "date"][0]
    assert json.loads(date_row["qdrant_value"]) == ["2014-01-01", "2016-01-01"]
    assert "points disagree" in date_row["note"]
    stats, _ = run(conn, store, tmp_path, write=True, csv_name="w.csv")
    assert stats.points_written == 3
    payloads = [p for p, _ in snapshot(store).values()]
    assert {(p["date"], p["year"], p["date_end"]) for p in payloads} == {
        ("2016-01-01", 2016, "2016-01-01T00:00:00Z")}
    again, _ = run(conn, store, tmp_path, csv_name="again.csv")
    assert again.disagreeing == [] and again.to_write == 0


def test_a_document_gone_from_the_vault_is_reported_and_left_alone(tmp_path):
    store = make_store()
    seed(store, "gone")
    seed(store, "deleted")
    conn = LedgerConn([vault_row("gone", gone=True), vault_row("deleted", deleted=True,
                                                               date=datetime.date(2000, 1, 1))])
    before = snapshot(store)
    stats, rows = run(conn, store, tmp_path, write=True)
    assert stats.actions[sm.MISSING_IN_VAULT] == 2 and stats.written == 0
    assert snapshot(store) == before
    assert {r["doc_id"]: r["action"] for r in rows} == {"gone": sm.MISSING_IN_VAULT,
                                                        "deleted": sm.MISSING_IN_VAULT}


def test_a_ledger_document_without_points_is_missing_in_qdrant(tmp_path):
    store = make_store()
    conn = LedgerConn([vault_row("nowhere")])
    stats, rows = run(conn, store, tmp_path, write=True)
    assert stats.actions[sm.MISSING_IN_QDRANT] == 1 and stats.written == 0
    assert rows[0]["action"] == sm.MISSING_IN_QDRANT


def test_an_undated_vault_row_removes_the_date_and_the_interval(tmp_path):
    store = make_store()
    seed(store, "jp-1", date="2026-01-01", year=2026)  # a stale placeholder
    conn = LedgerConn([vault_row("jp-1", date=None, year=2026,
                                 extra={"date_precision": "garbage", "date_kind": "meeting"})])
    stats, rows = run(conn, store, tmp_path, write=True)
    assert stats.actions[sm.UNDATED] == 1 and stats.written == 1
    assert {r["field"] for r in rows} == {"date", "date_precision", "date_kind"}
    for payload, _ in snapshot(store).values():
        assert "date" not in payload and "date_start" not in payload and "date_end" not in payload
        assert payload["date_precision"] == "garbage" and payload["year"] == 2026
    again, _ = run(conn, store, tmp_path, csv_name="again.csv")
    assert again.actions[sm.UNDATED] == 1 and again.to_write == 0


def test_a_year_removed_in_the_vault_is_removed_from_the_points(tmp_path):
    store = make_store()
    seed(store, "d", date="2016-01-01", year=2016)
    conn = LedgerConn([vault_row("d", date=None, year=None)])
    run(conn, store, tmp_path, write=True)
    assert all("year" not in p for p, _ in snapshot(store).values())


def test_title_is_compared_only_on_request(tmp_path):
    store = make_store()
    seed(store, "d", title="Old title")
    conn = LedgerConn([vault_row("d", title="New title", extra={"date_precision": "day"})])
    stats, rows = run(conn, store, tmp_path, write=True)
    assert "title" not in {r["field"] for r in rows}
    assert all(p["title"] == "Old title" for p, _ in snapshot(store).values())
    stats, rows = run(conn, store, tmp_path, write=True, include_title=True, csv_name="t.csv")
    assert [r["field"] for r in rows] == ["title"]
    assert all(p["title"] == "New title" for p, _ in snapshot(store).values())


@pytest.mark.parametrize("page_size,doc_batch", [(1, 1), (2, 3), (256, 100)])
def test_paging_does_not_change_the_answer(tmp_path, page_size, doc_batch):
    store = make_store()
    ledger = []
    for i in range(7):
        seed(store, f"d{i}", n=i % 3 + 1)
        ledger.append(vault_row(f"d{i}", date=datetime.date(2016, 1, 1 + i),
                                extra={"date_precision": "day"}))
    ledger.append(vault_row("absent"))
    stats, rows = run(LedgerConn(ledger), store, tmp_path, page_size=page_size,
                      doc_batch=doc_batch)
    assert stats.points_read == sum(i % 3 + 1 for i in range(7))
    assert stats.actions[sm.UPDATED] == 7 and stats.actions[sm.MISSING_IN_QDRANT] == 1
    assert len(rows) == 1 + 3 + 6 * 4  # absent; d0 keeps its date: 3 keys; the others 4
    by_doc = {}
    for r in rows:
        by_doc.setdefault(r["doc_id"], set()).add(r["field"])
    assert by_doc["d0"] == {"date_precision", "date_start", "date_end"}
    assert by_doc["d3"] == {"date", "date_precision", "date_start", "date_end"}


def test_limit_and_doc_ids_narrow_the_ledger(tmp_path):
    store = make_store()
    ledger = [vault_row(f"d{i}") for i in range(5)]
    for i in range(5):
        seed(store, f"d{i}", n=1)
    stats, _ = run(LedgerConn(ledger), store, tmp_path, limit=2)
    assert stats.seen_ids == {"d0", "d1"}
    stats, _ = run(LedgerConn(ledger), store, tmp_path, doc_ids=["d3", "zz"], csv_name="b.csv")
    assert stats.seen_ids == {"d3"}


def test_write_creates_the_missing_indexes_first_and_a_dry_run_never(tmp_path):
    calls = []

    class Recording(QdrantStore):
        def ensure_payload_indexes(self, collection, indexes=None):
            calls.append(collection)
            return ["date_start", "date_end"]

    store = Recording(client=QdrantClient(":memory:"), settings=Settings())
    store.ensure_collection(COLLECTION, 4)
    stats, _ = run(LedgerConn([]), store, tmp_path)
    assert calls == [] and "date_start" in stats.indexes_missing
    stats, _ = run(LedgerConn([]), store, tmp_path, write=True, csv_name="w.csv")
    assert calls == [COLLECTION] and stats.indexes_created == ["date_start", "date_end"]


def test_a_missing_collection_is_a_sync_error(tmp_path):
    store = QdrantStore(client=QdrantClient(":memory:"), settings=Settings())
    with pytest.raises(sm.SyncError, match="does not exist"):
        run(LedgerConn([]), store, tmp_path)


def test_a_write_that_would_not_be_what_ingestion_writes_is_refused(monkeypatch):
    monkeypatch.setattr(sm, "date_fields", lambda d, p, year=None: {
        "date": d, "date_precision": p, "date_start": None, "date_end": None, "year": year})
    with pytest.raises(sm.SyncError, match="not what an ingestion writes"):
        sm.write_fields("d", vault_row("d", date=datetime.date(2016, 1, 1)))


class _HTTPError(Exception):
    def __init__(self, status, content=b""):
        super().__init__(f"HTTP {status}")
        self.status_code, self.content = status, content


def test_qdrant_errors_are_named():
    with pytest.raises(QdrantAuthError):
        sm._qdrant_error(_HTTPError(403), "c")
    body = (b'{"status":{"error":"Bad request: Index required but not found for \\"doc_id\\" '
            b'of one of the following types: [keyword]"}}')
    with pytest.raises(MissingIndexError, match="strict mode"):
        sm._qdrant_error(_HTTPError(400, body), "c")
    sm._qdrant_error(_HTTPError(400, b"other"), "c")  # anything else: the caller re-raises
    sm._qdrant_error(_HTTPError(500), "c")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
@pytest.fixture
def cli_env(monkeypatch, tmp_path):
    import data_orchestrator.core as core
    import data_orchestrator.vault as vault_mod

    monkeypatch.setenv("DATA_ORCHESTRATOR_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("DATA_ORCHESTRATOR_EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
    store = make_store()
    conn = LedgerConn()
    monkeypatch.setattr(vault_mod, "connect", lambda: conn)
    monkeypatch.setattr(core, "build_store", lambda settings: store)
    return store, conn


def test_cli_dry_run_prints_counts_and_the_csv(cli_env, tmp_path, capsys):
    store, conn = cli_env
    seed(store, "gb-1")
    conn.ledger = [vault_row("gb-1", date=datetime.date(2015, 12, 17), year=2015,
                             extra={"date_precision": "day", "date_kind": "meeting"})]
    out_csv = tmp_path / "x.csv"
    rc = cli.main(["sync-metadata", "--collection", COLLECTION, "--csv", str(out_csv)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "dry run: nothing is written to Qdrant" in out
    assert "  updated            : 1" in out and "  documents to write : 1" in out
    assert f"CSV                : {out_csv} (6 row(s))" in out
    assert "payload indexes missing" in out
    assert conn.closed


def test_cli_default_csv_goes_to_the_state_dir(cli_env, tmp_path, capsys):
    rc = cli.main(["sync-metadata", "--collection", COLLECTION])
    assert rc == 0
    (path,) = (tmp_path / "state").glob(f"sync-metadata-{COLLECTION}-*.csv")
    assert path.read_text(encoding="utf-8").startswith("doc_id,action,field,")


def test_cli_refuses_to_run_beside_an_ingestion(cli_env, capsys):
    _store, conn = cli_env
    conn.lock_free = False
    rc = cli.main(["sync-metadata", "--collection", COLLECTION, "--write"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "CollectionBusyError" in err and "an ingestion" in err


def test_an_ingestion_is_refused_while_sync_metadata_runs(monkeypatch, tmp_path, capsys):
    from tests.test_cli_args import _fake_vault

    conn, reports = _fake_vault(monkeypatch, tmp_path, [])
    conn.lock_free = False
    rc = cli.main(["vault"])
    assert rc == 1
    assert "a sync-metadata --write run holds collection" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [
    ["vault", "--write"], ["cb_corpus", "--doc-ids", "a"], ["probe", "--include-title"],
    ["vault", "--create-indexes"],
])
def test_sync_flags_are_refused_elsewhere(argv, capsys):
    assert cli.main(argv) == 2
    assert "only apply to sync-metadata" in capsys.readouterr().err


@pytest.mark.parametrize("extra", [
    ["--count-only"], ["--no-vault"], ["--source-codes", "gb"], ["--year-min", "2000"],
    ["--create-indexes", "--write"], ["--page-size", "0"], ["--doc-batch", "0"],
    ["--doc-ids", ","], ["--limit", "0"],
])
def test_sync_metadata_usage_errors(extra, cli_env, capsys):
    assert cli.main(["sync-metadata", *extra]) == 2


def test_cli_create_indexes(cli_env, capsys, monkeypatch):
    store, _conn = cli_env
    monkeypatch.setattr(store, "ensure_payload_indexes", lambda c: ["date_start", "date_end"])
    assert cli.main(["sync-metadata", "--collection", COLLECTION, "--create-indexes"]) == 0
    assert "Payload indexes created on 'c-test': date_start, date_end" in capsys.readouterr().out


def test_cli_a_refused_key_says_which_key_is_needed(cli_env, capsys, monkeypatch):
    store, _conn = cli_env
    seed(store, "d")
    _conn.ledger = [vault_row("d", date=datetime.date(2015, 1, 1))]

    def refuse(*a, **k):
        raise QdrantAuthError("Qdrant refused the request (HTTP 403): check QDRANT_API_KEY")

    monkeypatch.setattr(store, "set_payload", refuse)
    monkeypatch.setattr(store, "ensure_payload_indexes", lambda c: [])
    assert cli.main(["sync-metadata", "--collection", COLLECTION, "--write"]) == 1
    err = capsys.readouterr().err
    assert "QdrantAuthError" in err and "a read-only key is refused" in err


def test_cli_missing_index_gives_the_remedy(cli_env, capsys, monkeypatch):
    def strict(*a, **k):
        raise MissingIndexError("collection 'c-test' refuses a filter on 'doc_id'")

    store, conn = cli_env
    conn.ledger = [vault_row("d")]
    monkeypatch.setattr(sm, "read_points", strict)
    assert cli.main(["sync-metadata", "--collection", COLLECTION]) == 1
    err = capsys.readouterr().err
    assert "MissingIndexError" in err
    assert f"sync-metadata --create-indexes --collection {COLLECTION}" in err


@pytest.mark.parametrize("extra", [[], ["--no-resume"]])
def test_a_disk_source_ingestion_in_vault_mode_takes_the_lock(monkeypatch, tmp_path, capsys,
                                                               extra):
    """With or without its resume ledger, a disk-source run in vault mode
    writes the points of documents sync-metadata may be correcting: it must
    take the shared lock (``--no-resume`` rewrites the payloads of ingested
    documents)."""
    from tests.test_cli_args import _fake_vault

    conn, _reports = _fake_vault(monkeypatch, tmp_path, [])
    (tmp_path / "raw").mkdir()
    (tmp_path / "manifest").mkdir()
    (tmp_path / "manifest" / "us.jsonl").write_text("", encoding="utf-8")
    conn.lock_free = False
    rc = cli.main(["cb_corpus", *extra])
    assert rc == 1
    assert "a sync-metadata --write run holds collection" in capsys.readouterr().err
    locks = [p for sql, p in conn.executed if "pg_try_advisory_lock_shared" in sql]
    assert locks and locks[0][1] == "central-bank-e5b-v1"


def test_cli_stops_when_the_vault_session_loses_the_lock(cli_env, capsys):
    """A lost lock (the session terminated, the network gone) stops the run
    before the next write instead of letting it write unlocked."""
    store, conn = cli_env
    for i in range(3):
        seed(store, f"d{i}", n=1)
    conn.ledger = [vault_row(f"d{i}", date=datetime.date(2015, 1, 1 + i)) for i in range(3)]
    writes = []
    real = store.set_payload

    def counting(collection, doc_id, fields):
        writes.append(doc_id)
        conn.lock_held = False  # the session is lost right after the first write
        return real(collection, doc_id, fields)

    store.set_payload = counting
    rc = cli.main(["sync-metadata", "--collection", COLLECTION, "--write", "--doc-batch", "1"])
    err = capsys.readouterr().err
    assert rc == 1 and writes == ["d0"]
    assert "VaultSessionLost" in err and "no longer holds the lock" in err
    assert "can be run again" in err


def test_run_sync_checks_the_guard_before_each_batch_and_each_write(tmp_path):
    store = make_store()
    for i in range(3):
        seed(store, f"d{i}", n=1)
    conn = LedgerConn([vault_row(f"d{i}", date=datetime.date(2015, 1, 1 + i)) for i in range(3)])
    calls = []
    run(conn, store, tmp_path, write=True, doc_batch=2, guard=lambda: calls.append(1))
    assert len(calls) == 2 + 3  # two batches, three writes


def test_date_source_is_owned_so_a_synced_point_equals_a_fresh_ingestion(tmp_path):
    """A fresh ingestion copies ``date_source`` from ``extra``: after a sync the
    date keys of a point are exactly those a new ingestion of the row writes."""
    store = make_store()
    seed(store, "d", n=2, date="2016-01-01", year=2016, date_source="placeholder")
    extra = {"date_precision": "day", "date_kind": "meeting", "date_source": "title"}
    row = vault_row("d", date=datetime.date(2016, 7, 13), year=2016, extra=extra)
    stats, rows = run(LedgerConn([row]), store, tmp_path, write=True)
    assert ("date_source", "placeholder", "title") in {
        (r["field"], r["qdrant_value"], r["vault_value"]) for r in rows}
    fresh = build_payload("d", sm.row_to_payload(row)[0], filename="x", chunk_number=0,
                          ingestion_date="t", text="t", page=0)
    for payload, _ in snapshot(store).values():
        assert {k: payload.get(k) for k in sm.OWNED_KEYS} == {k: fresh.get(k) for k in sm.OWNED_KEYS}
        assert payload["date_source"] == "title"
    # A vault row that drops date_source removes it from the points.
    row2 = vault_row("d", date=datetime.date(2016, 7, 13), year=2016,
                     extra={"date_precision": "day", "date_kind": "meeting"})
    run(LedgerConn([row2]), store, tmp_path, write=True, csv_name="b.csv")
    assert all("date_source" not in p for p, _ in snapshot(store).values())


@pytest.mark.parametrize("write,expected", [(False, "pg_try_advisory_lock_shared"),
                                            (True, "pg_try_advisory_lock(")])
def test_a_dry_run_takes_the_shared_lock_a_write_the_exclusive_one(cli_env, write, expected):
    """A dry run only reads: it must not stop an ingestion from starting (the
    shared lock is compatible with an ingestion's), but it does not run beside
    a --write (which holds the exclusive one)."""
    _store, conn = cli_env
    assert cli.main(["sync-metadata", "--collection", COLLECTION,
                     *(["--write"] if write else [])]) == 0
    (lock_sql,) = [sql for sql, _ in conn.executed if "pg_try_advisory_lock" in sql]
    assert expected in lock_sql
    modes = {p[2] for sql, p in conn.executed if "FROM pg_locks" in sql}
    assert modes <= {"ExclusiveLock" if write else "ShareLock"}


def test_a_dry_run_refused_beside_a_write_says_so(cli_env, capsys):
    _store, conn = cli_env
    conn.lock_free = False
    assert cli.main(["sync-metadata", "--collection", COLLECTION]) == 1
    assert "a sync-metadata --write run holds collection" in capsys.readouterr().err


@pytest.mark.parametrize("stored,expected", [(2016.0, 2016), (True, 1), ("2016", 2016)])
def test_a_value_of_another_type_is_a_difference(stored, expected):
    """2016.0 == 2016 and True == 1 in Python, but a fresh ingestion writes the
    vault's type: the comparison is on the value and its type."""
    diffs, disagree = sm.compare("d", {"year": expected}, [{"year": stored}], ("year",))
    assert diffs == [("year", stored, expected)] and not disagree


def test_an_explicit_null_in_extra_and_a_missing_key_are_the_same(tmp_path):
    """Ingestion leaves ``"date_kind": null`` of ``extra`` out; a point
    written before it did holds a stored null. set_payload cannot store one
    (Qdrant drops a key set to null), and every filter reads a null as a
    missing key: both count as absent, so neither a point holding the null nor
    one without the key is reported or written."""
    store = make_store()
    seed(store, "with-null", n=1, date="2016-07-13", year=2016, date_kind=None)
    store.client.set_payload(COLLECTION, payload={"date_precision": "day",
                                                  "date_start": "2016-07-13T00:00:00Z",
                                                  "date_end": "2016-07-13T00:00:00Z"},
                             points=[point_id("with-null", 0, 0)], wait=True)
    store.client.overwrite_payload(  # the null an ingestion writes
        COLLECTION, payload={**snapshot(store)[point_id("with-null", 0, 0)][0], "date_kind": None},
        points=[point_id("with-null", 0, 0)], wait=True)
    assert snapshot(store)[point_id("with-null", 0, 0)][0]["date_kind"] is None
    row = vault_row("with-null", date=datetime.date(2016, 7, 13), year=2016,
                    extra={"date_precision": "day", "date_kind": None})
    stats, rows = run(LedgerConn([row]), store, tmp_path, write=True)
    assert stats.actions[sm.UNCHANGED] == 1 and stats.written == 0 and rows == []


def test_the_payload_schema_is_read_once_per_run(tmp_path, monkeypatch):
    store = make_store()
    reads = []
    real = sm.payload_schema
    monkeypatch.setattr(sm, "payload_schema", lambda *a: reads.append(1) or real(*a))
    schema = sm.payload_schema(store, COLLECTION)
    reads.clear()
    run(LedgerConn([]), store, tmp_path, schema=schema)
    assert reads == []


# ---------------------------------------------------------------------------
# --include-links: document_url, source_url, mime_type from the vault, and
# pdf_url (the former name of document_url) and the keys of extra dropped by
# design removed from the points that still carry them.
# ---------------------------------------------------------------------------
LINKS = {"document_url": "https://x.org/d.pdf", "source_url": "https://x.org/d",
         "mime_type": "application/pdf"}


def test_the_links_are_owned_only_on_request():
    assert not set(sm.LINK_KEYS) & set(sm.owned_keys())
    keys = sm.owned_keys(include_links=True)
    assert keys[:len(sm.OWNED_KEYS)] == sm.OWNED_KEYS
    assert set(keys) == set(sm.OWNED_KEYS) | {"document_url", "source_url", "mime_type",
                                              "pdf_url", "html_path", "alt_urls",
                                              "alt_source_urls", "local_path"}
    assert len(keys) == len(set(keys))


def test_the_owned_dropped_keys_follow_the_setting(monkeypatch):
    monkeypatch.setenv("DATA_ORCHESTRATOR_PAYLOAD_DROP_EXTRA_KEYS", "html_path,date_source")
    keys = sm.owned_keys(include_links=True)
    assert "html_path" in keys and "alt_urls" not in keys
    assert keys.count("date_source") == 1  # an owned date key named again is owned once


def test_the_links_write_is_exactly_the_expected_values():
    row = vault_row("d", extra={"html_path": "/Users/x/d.html", "alt_urls": ["a"],
                                "date_precision": "day"}, document_url=LINKS["document_url"],
                    source_url=None, mime_type="application/pdf")
    fields = sm.write_fields("d", row, include_links=True)
    assert fields["document_url"] == LINKS["document_url"]
    assert fields["mime_type"] == "application/pdf"
    assert fields["source_url"] is None  # a NULL column removes the key
    assert "pdf_url" in fields and fields["pdf_url"] is None  # the former name: removed
    assert all(fields[k] is None for k in ("html_path", "alt_urls", "alt_source_urls",
                                           "local_path"))
    keys = sm.owned_keys(include_links=True)
    assert {k: v for k, v in fields.items() if v is not None} == sm.expected_values("d", row, keys)


def test_include_links_sets_the_links_and_removes_html_path_then_finds_nothing(tmp_path):
    store = make_store()
    # A point ingested before the links were sent, carrying the producer's path.
    seed(store, "d", n=2, html_path="/Users/someone/cb/d.html", alt_urls=["https://alt"],
         date_precision="day")
    store.client.set_payload(COLLECTION, payload={"date_start": "2016-01-01T00:00:00Z",
                                                  "date_end": "2016-01-01T00:00:00Z"},
                             points=[point_id("d", c, c) for c in range(2)], wait=True)
    row = vault_row("d", extra={"date_precision": "day", "html_path": "/Users/someone/cb/d.html",
                                "alt_urls": ["https://alt"]}, **LINKS)
    conn = LedgerConn([row])
    before = snapshot(store)

    # Without the flag: nothing to do (the date keys already agree).
    stats, rows = run(conn, store, tmp_path)
    assert stats.to_write == 0 and rows == []

    stats, rows = run(conn, store, tmp_path, include_links=True, csv_name="dry.csv")
    assert snapshot(store) == before  # a dry run writes nothing
    assert {r["field"]: (r["qdrant_value"], r["vault_value"]) for r in rows} == {
        "document_url": ("", LINKS["document_url"]),
        "source_url": ("", LINKS["source_url"]),
        "mime_type": ("", "application/pdf"),
        "html_path": ("/Users/someone/cb/d.html", ""),
        "alt_urls": ('["https://alt"]', ""),
    }
    assert stats.fields["html_path"] == 1

    stats, _ = run(conn, store, tmp_path, include_links=True, write=True, csv_name="w.csv")
    assert stats.written == 1 and stats.points_written == 2
    after = snapshot(store)
    for pid, (payload, vector) in after.items():
        assert vector == before[pid][1]
        assert "html_path" not in payload and "alt_urls" not in payload
        assert {k: payload[k] for k in LINKS} == LINKS
        old = before[pid][0]
        changed = {k for k in set(payload) | set(old) if payload.get(k) != old.get(k)}
        assert changed == {"document_url", "source_url", "mime_type", "html_path", "alt_urls"}

    again, rows = run(conn, store, tmp_path, include_links=True, write=True, csv_name="a.csv")
    assert again.to_write == 0 and again.written == 0 and rows == []
    assert snapshot(store) == after


def test_include_links_removes_a_link_the_vault_no_longer_has(tmp_path):
    store = make_store()
    seed(store, "d", n=1, document_url="https://old.org/d.pdf")
    row = vault_row("d", date=datetime.date(2016, 1, 1), document_url=None)
    run(LedgerConn([row]), store, tmp_path, include_links=True, write=True)
    assert all("document_url" not in p for p, _ in snapshot(store).values())


def test_cli_include_links(cli_env, tmp_path, capsys):
    store, conn = cli_env
    seed(store, "gb-1", html_path="/x.html")
    conn.ledger = [vault_row("gb-1", extra={"html_path": "/x.html"}, **LINKS)]
    out_csv = tmp_path / "l.csv"
    rc = cli.main(["sync-metadata", "--collection", COLLECTION, "--include-links",
                   "--csv", str(out_csv)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "document_url, source_url, mime_type, pdf_url, html_path" in out
    with out_csv.open(encoding="utf-8") as fh:
        fields = {r["field"] for r in csv.DictReader(fh)}
    assert {"document_url", "source_url", "mime_type", "html_path"} <= fields


def test_include_links_is_refused_elsewhere_and_with_create_indexes(cli_env, capsys):
    assert cli.main(["vault", "--include-links"]) == 2
    assert "only apply to sync-metadata" in capsys.readouterr().err
    assert cli.main(["sync-metadata", "--create-indexes", "--include-links"]) == 2


def test_include_links_removes_a_blank_link_and_never_writes_one(tmp_path):
    """2,294 central-bank rows of the vault hold ``source_url = ''``: no link,
    so the write removes the key and a point carrying ``""`` is corrected."""
    row = vault_row("d", date=datetime.date(2016, 1, 1), document_url="https://x.org/d.pdf",
                    source_url="", mime_type="  ")
    fields = sm.write_fields("d", row, include_links=True)
    assert fields["source_url"] is None and fields["mime_type"] is None
    keys = sm.owned_keys(include_links=True)
    assert "source_url" not in sm.expected_values("d", row, keys)
    store = make_store()
    seed(store, "d", n=1, source_url="", document_url="https://x.org/d.pdf")
    stats, _ = run(LedgerConn([row]), store, tmp_path, include_links=True, write=True)
    assert stats.written == 1
    assert all("source_url" not in p for p, _ in snapshot(store).values())


# ---------------------------------------------------------------------------
# The pdf_url -> document_url rename (2026-11): one --include-links run renames
# the key on the points; the ledger reads either column of the vault.
# ---------------------------------------------------------------------------
def _aligned_before_the_rename(store, doc_id, n):
    """Points as production holds them before the rename: the date set of an
    ingestion, and the links set by an earlier --include-links under ``pdf_url``."""
    seed(store, doc_id, n=n, date_precision="day", pdf_url=f"https://x.org/{doc_id}.pdf",
         source_url=f"https://x.org/{doc_id}", mime_type="application/pdf")
    store.client.set_payload(COLLECTION, payload={"date_start": "2016-01-01T00:00:00Z",
                                                  "date_end": "2016-01-01T00:00:00Z"},
                             points=[point_id(doc_id, c, c) for c in range(n)], wait=True)


def _renamed_row(doc_id):
    return vault_row(doc_id, extra={"date_precision": "day"},
                     document_url=f"https://x.org/{doc_id}.pdf",
                     source_url=f"https://x.org/{doc_id}", mime_type="application/pdf")


def test_include_links_renames_pdf_url_on_every_point_once(tmp_path):
    store = make_store()
    for doc_id, n in (("a", 3), ("b", 1), ("c", 2)):
        _aligned_before_the_rename(store, doc_id, n)
    conn = LedgerConn([_renamed_row(d) for d in ("a", "b", "c")])
    before = snapshot(store)

    # Without --include-links the dates already agree: nothing to do.
    stats, rows = run(conn, store, tmp_path)
    assert stats.to_write == 0 and rows == []

    # The dry run shows the rename: document_url set, pdf_url removed, per document.
    stats, rows = run(conn, store, tmp_path, include_links=True, csv_name="dry.csv")
    assert snapshot(store) == before
    assert stats.to_write == 3 and stats.fields == {"document_url": 3, "pdf_url": 3}
    assert {(r["doc_id"], r["field"], r["qdrant_value"], r["vault_value"]) for r in rows} == {
        (d, k, q, v) for d in ("a", "b", "c") for k, q, v in (
            ("document_url", "", f"https://x.org/{d}.pdf"),
            ("pdf_url", f"https://x.org/{d}.pdf", ""))}

    # The write: every ledger document once, every point reached.
    stats, _ = run(conn, store, tmp_path, include_links=True, write=True, csv_name="w.csv")
    assert stats.written == 3 and stats.points_written == 6 and not stats.count_mismatch
    after = snapshot(store)
    for pid, (payload, vector) in after.items():
        assert vector == before[pid][1]
        old = before[pid][0]
        assert "pdf_url" not in payload and payload["document_url"] == old["pdf_url"]
        changed = {k for k in set(payload) | set(old) if payload.get(k) != old.get(k)}
        assert changed == {"document_url", "pdf_url"}

    # Idempotent: a second run finds nothing.
    again, rows = run(conn, store, tmp_path, include_links=True, write=True, csv_name="a.csv")
    assert again.to_write == 0 and again.written == 0 and rows == []
    assert snapshot(store) == after


def test_a_vault_before_the_rename_is_read_through_pdf_url(tmp_path):
    """The orchestrator may run before the vault's DDL renamed the column: the
    ledger then selects ``pdf_url AS document_url`` and the answer is the same."""
    store = make_store()
    _aligned_before_the_rename(store, "a", 2)
    old_vault = LedgerConn([_renamed_row("a")], columns=("pdf_url",))
    stats, rows = run(old_vault, store, tmp_path, include_links=True, write=True)
    assert "d.pdf_url AS document_url" in old_vault.executed[1][0]
    assert stats.written == 1 and {r["field"] for r in rows} == {"document_url", "pdf_url"}
    assert all(p["document_url"] == "https://x.org/a.pdf" and "pdf_url" not in p
               for p, _ in snapshot(store).values())

    new_vault = LedgerConn([_renamed_row("a")], columns=("document_url",))
    again, _ = run(new_vault, store, tmp_path, include_links=True, csv_name="n.csv")
    assert "d.document_url," in new_vault.executed[1][0]
    assert again.to_write == 0


def test_the_column_is_read_once_per_connection():
    conn = LedgerConn([_renamed_row("a")], columns=("pdf_url", "document_url"))
    sm.ledger_rows(conn, CORPUS, COLLECTION)
    rows = sm.ledger_rows(conn, CORPUS, COLLECTION)
    reads = [sql for sql, _ in conn.executed if "information_schema" in sql]
    assert len(reads) == 1
    # Both columns (a vault mid-DDL, or a copy of the old one): the new one wins.
    assert "d.document_url," in conn.executed[-1][0]
    assert rows[0]["document_url"] == "https://x.org/a.pdf"


# ---------------------------------------------------------------------------
# The documents contract: date_precision and date_kind leave extra for columns
# ---------------------------------------------------------------------------
def _ingested_before_the_contract(store, doc_id, row, n=2):
    """Points as an ingestion wrote them from ``row`` (interval included)."""
    from data_orchestrator.payload import row_to_payload

    meta, _ = row_to_payload(dict(row, doc_id=doc_id))
    points = [models.PointStruct(
        id=point_id(doc_id, c, c), vector=[0.1 * (c + 1), 0.2, -0.3, 0.4],
        payload=build_payload(doc_id, meta, filename=f"{doc_id}.pdf", chunk_number=c,
                              ingestion_date="2026-10-01T00:00:00", text=f"t{c}", page=c))
        for c in range(n)]
    store.client.upsert(COLLECTION, points=points, wait=True)


@pytest.mark.parametrize("include_links", [False, True])
def test_after_the_vault_moves_the_date_keys_to_columns_a_write_changes_no_point(
        tmp_path, include_links):
    """The points were ingested from a vault row with ``date_precision`` and
    ``date_kind`` in ``extra``; the vault ingestion on the contract moves both
    to their columns and out of ``extra``. sync-metadata must read them from the
    columns: nothing differs, and a --write strips no date key from any point
    (its self-check cannot catch a payload rule that forgets a column, since
    both sides come from the same ``row_to_payload``: hence this test)."""
    store = make_store()
    before = vault_row("d", date=datetime.date(2016, 7, 1),
                       extra={"date_precision": "month", "date_kind": "meeting",
                              "date_source": "title"})
    _ingested_before_the_contract(store, "d", before)
    seeded = snapshot(store)
    assert all(p["date_precision"] == "month" and p["date_kind"] == "meeting"
               and p["date_end"] == "2016-07-31T00:00:00Z" for p, _v in seeded.values())
    after = vault_row("d", date=datetime.date(2016, 7, 1), extra={"date_source": "title"},
                      date_precision="month", date_kind="meeting", entity_scheme=None,
                      entity_value=None)
    stats, rows = run(LedgerConn([after]), store, tmp_path, write=True,
                      include_links=include_links)
    assert stats.actions[sm.UNCHANGED] == 1 and stats.to_write == 0 and rows == []
    assert snapshot(store) == seeded


def test_the_ledger_query_selects_the_contract_columns():
    conn = LedgerConn([])
    sm.ledger_rows(conn, CORPUS, COLLECTION)
    sql = [s for s, _p in conn.executed if "FROM rag_ingestions r" in s][0]
    for column in ("date_precision", "date_kind", "entity_scheme", "entity_value"):
        assert f"d.{column}" in sql


def test_an_undated_row_that_lost_its_precision_is_one_difference(tmp_path):
    """One of the 7 undated central-bank rows with a precision: its points carry
    ``date_precision: month`` (from the vault's extra before the contract); on
    the contract the vault drops a precision without a date, so the dry run
    reports that one key, nothing else (SYNC-METADATA.md)."""
    store = make_store()
    _ingested_before_the_contract(store, "u", vault_row("u", date=None, year=None,
                                                        extra={"date_precision": "month",
                                                               "date_source": "repec"}))
    after = vault_row("u", date=None, year=None, extra={"date_source": "repec"},
                      date_precision=None, date_kind=None)
    stats, rows = run(LedgerConn([after]), store, tmp_path)
    assert stats.actions[sm.UNDATED] == 1 and stats.to_write == 1
    assert [(r["field"], r["qdrant_value"], r["vault_value"]) for r in rows] == [
        ("date_precision", "month", "")]
