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
        if "FROM pg_locks" in sql:
            self._rows = [(int(self._conn.lock_held),)]
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

    def __init__(self, ledger=(), lock_free=True):
        self.ledger = list(ledger)
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
    assert set(fields) == {"date", "date_precision", "date_start", "date_end", "year", "date_kind"}
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
    # The vault is only read: one query of the ledger (the CLI takes the lock).
    assert [sql.split()[0] for sql, _ in conn.executed] == ["SELECT"]


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
    assert "a sync-metadata run holds collection" in capsys.readouterr().err


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
    assert "a sync-metadata run holds collection" in capsys.readouterr().err
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
