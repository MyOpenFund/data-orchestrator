"""The vault session that holds the collection lock: keepalives, the lock
re-checked during a run, and a lost session fatal for the run."""
import psycopg2
import pytest

from data_orchestrator import vault
from data_orchestrator.core import FatalRunError, run_ingest
from tests.test_core_engine import FakeEmbedder, FakeStore, RecordingLedger, _md_item


class _Cur:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        self.conn.executed.append((" ".join(sql.split()), params))
        if self.conn.dead:
            self.conn.closed = 2
            raise psycopg2.OperationalError("server closed the connection unexpectedly")

    def fetchone(self):
        return (self.conn.held,)


class _Conn:
    def __init__(self, held=1, dead=False):
        self.held, self.dead, self.closed = held, dead, 0
        self.executed = []

    def cursor(self):
        return _Cur(self)

    def commit(self):
        pass

    def rollback(self):
        if self.closed:
            raise psycopg2.InterfaceError("connection already closed")


def test_keepalives_have_defaults_and_can_be_set():
    assert vault.keepalive_options({}) == {
        "keepalives": 1, "keepalives_idle": 30, "keepalives_interval": 10, "keepalives_count": 3,
        "tcp_user_timeout": 30000}
    assert vault.keepalive_options({
        "DATA_ORCHESTRATOR_PG_KEEPALIVES_IDLE": "60",
        "DATA_ORCHESTRATOR_PG_KEEPALIVES_INTERVAL": " 5 ",
        "DATA_ORCHESTRATOR_PG_KEEPALIVES_COUNT": "4"})["keepalives_idle"] == 60


@pytest.mark.parametrize("value", ["0", "-1", "x", "1.5", ""])
def test_a_bad_keepalive_value_is_refused(value):
    with pytest.raises(ValueError, match="DATA_ORCHESTRATOR_PG_KEEPALIVES_COUNT"):
        vault.keepalive_options({"DATA_ORCHESTRATOR_PG_KEEPALIVES_COUNT": value})


def test_connect_turns_the_keepalives_on(monkeypatch):
    seen = {}
    monkeypatch.setenv("DATABASE_URL", "postgresql://u@h/db")
    monkeypatch.setattr(vault.psycopg2, "connect", lambda url, **kw: seen.update(url=url, **kw))
    vault.connect()
    assert seen["url"] == "postgresql://u@h/db"
    assert seen["keepalives"] == 1 and seen["keepalives_idle"] == 30


def test_the_lock_check_passes_while_the_lock_is_held():
    conn = _Conn(held=1)
    vault.check_collection_lock(conn, "c", exclusive=True, holder="sync-metadata")
    sql, params = conn.executed[0]
    assert "pg_locks" in sql and "pg_backend_pid()" in sql
    assert params == (vault.COLLECTION_LOCK_CLASS, "c", "ExclusiveLock")


def test_a_lock_no_longer_held_is_a_lost_session():
    with pytest.raises(vault.VaultSessionLost, match="no longer holds"):
        vault.check_collection_lock(_Conn(held=0), "c", exclusive=False, holder="vault ingestion")


def test_a_dead_vault_session_is_a_lost_session():
    with pytest.raises(vault.VaultSessionLost, match="vault session"):
        vault.check_collection_lock(_Conn(dead=True), "c", exclusive=True, holder="sync-metadata")
    assert issubclass(vault.VaultSessionLost, FatalRunError)


def test_a_ledger_write_on_a_dead_session_is_fatal_not_a_document_error(tmp_path):
    class Ledger(vault.VaultLedger):
        def __init__(self, conn):
            self._conn, self.collection, self.corpus = conn, "c", "central-bank"
            self.embedding_model, self.embedding_version, self._done = "m", "v", set()

    conn = _Conn(dead=True)
    items = [_md_item(tmp_path, doc_id=f"d{i}") for i in range(3)]
    store = FakeStore([])
    with pytest.raises(vault.VaultSessionLost):
        run_ingest(items, collection="c", ledger=Ledger(conn), store=store,
                   embedder=FakeEmbedder())
    assert {p.payload["doc_id"] for p in store.client.points} == {"d0"}  # stopped at once


def test_a_guard_that_fails_stops_the_run_before_the_next_document(tmp_path):
    calls = []

    def guard():
        calls.append(1)
        if len(calls) == 2:
            raise vault.VaultSessionLost("gone")

    log = []
    items = [_md_item(tmp_path, doc_id=f"d{i}") for i in range(3)]
    store = FakeStore(log)
    with pytest.raises(vault.VaultSessionLost) as exc:
        run_ingest(items, collection="c", ledger=RecordingLedger(log), store=store,
                   embedder=FakeEmbedder(), guard=guard)
    assert {p.payload["doc_id"] for p in store.client.points} == {"d0"}
    assert exc.value.ingest_stats.docs_ingested == 1


def test_a_bad_keepalive_setting_is_a_usage_error_before_any_work(monkeypatch, capsys):
    from data_orchestrator import cli

    monkeypatch.setenv("DATA_ORCHESTRATOR_PG_KEEPALIVES_IDLE", "0")
    monkeypatch.setattr(vault, "connect", lambda: pytest.fail("connected"))
    assert cli.main(["sync-metadata"]) == 2
    assert "DATA_ORCHESTRATOR_PG_KEEPALIVES_IDLE" in capsys.readouterr().err


def test_the_tcp_user_timeout_bounds_a_request_sent_into_a_dead_link():
    """Keepalives only probe an idle connection: a request sent into a dead
    link waits for TCP retransmissions (about 15 min on Linux) unless
    tcp_user_timeout bounds it. 0 leaves the system default."""
    assert vault.keepalive_options({"DATA_ORCHESTRATOR_PG_TCP_USER_TIMEOUT_MS": "10000"})[
        "tcp_user_timeout"] == 10000
    assert vault.keepalive_options({"DATA_ORCHESTRATOR_PG_TCP_USER_TIMEOUT_MS": "0"})[
        "tcp_user_timeout"] == 0
    with pytest.raises(ValueError, match="DATA_ORCHESTRATOR_PG_TCP_USER_TIMEOUT_MS"):
        vault.keepalive_options({"DATA_ORCHESTRATOR_PG_TCP_USER_TIMEOUT_MS": "-5"})


def test_connect_sets_the_tcp_user_timeout(monkeypatch):
    seen = {}
    monkeypatch.setenv("DATABASE_URL", "postgresql://u@h/db")
    monkeypatch.setattr(vault.psycopg2, "connect", lambda url, **kw: seen.update(kw))
    vault.connect()
    assert seen["tcp_user_timeout"] == 30000


class _BusyConn:
    """The lock is taken; the holder query answers ``holders`` (or fails)."""

    def __init__(self, holders, fail=False):
        self.holders, self.fail, self.executed = holders, fail, []

    def cursor(self):
        conn = self

        class C:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def execute(self, sql, params=None):
                conn.executed.append(" ".join(sql.split()))
                if "pg_stat_activity" in sql and conn.fail:
                    raise psycopg2.errors.InsufficientPrivilege("permission denied")
                self.sql = sql

            def fetchone(self):
                return (False,)

            def fetchall(self):
                return conn.holders

        return C()

    def commit(self):
        pass

    def rollback(self):
        pass


def test_a_refusal_names_the_holder_and_how_to_clear_a_stale_lock():
    import datetime as dt

    conn = _BusyConn([(4242, "ShareLock", "vault_orchestrator", "data-orchestrator@mac",
                       "100.64.0.7", dt.datetime(2026, 11, 18, 22, 1, 2, tzinfo=dt.timezone.utc))])
    with pytest.raises(vault.CollectionBusyError) as exc:
        vault.lock_collection(conn, "c", exclusive=True, holder="sync-metadata")
    msg = str(exc.value)
    assert "pid 4242" in msg and "client 100.64.0.7" in msg
    assert "application data-orchestrator@mac" in msg and "2026-11-18" in msg
    assert "SELECT pg_terminate_backend(4242)" in msg
    sql = [s for s in conn.executed if "pg_stat_activity" in s][0]
    assert "current_database()" in sql and "pg_backend_pid()" in sql


def test_a_refusal_when_the_role_cannot_see_the_holder_s_details():
    conn = _BusyConn([(4242, "ShareLock", None, None, None, None)])
    with pytest.raises(vault.CollectionBusyError, match=r"held by pid 4242 \(ShareLock, its client and start are not visible to this role\)"):
        vault.lock_collection(conn, "c", exclusive=True, holder="sync-metadata")
    failing = _BusyConn([], fail=True)
    with pytest.raises(vault.CollectionBusyError, match="vault advisory lock 7468"):
        vault.lock_collection(failing, "c", exclusive=True, holder="sync-metadata")


def test_connect_names_the_session(monkeypatch):
    seen = {}
    monkeypatch.setenv("DATABASE_URL", "postgresql://u@h/db")
    monkeypatch.setattr(vault.psycopg2, "connect", lambda url, **kw: seen.update(kw))
    vault.connect()
    assert seen["application_name"].startswith("data-orchestrator@")
    assert len(seen["application_name"]) <= 63


@pytest.mark.parametrize("argv", [["vault"], ["cb_corpus"], ["cb_corpus", "--no-resume"]])
def test_every_vault_mode_ingestion_checks_the_lock_before_each_document(
        monkeypatch, tmp_path, argv):
    """The guard handed to run_ingest is the lock check of the run's own
    session, in shared mode: removing that wiring must fail here."""
    from data_orchestrator import cli
    from data_orchestrator.core import IngestStats
    from tests.test_cli_args import _fake_vault

    conn, _reports = _fake_vault(monkeypatch, tmp_path, [])
    (tmp_path / "raw").mkdir()
    (tmp_path / "manifest").mkdir()
    (tmp_path / "manifest" / "us.jsonl").write_text("", encoding="utf-8")
    seen = {}

    def fake_run_ingest(items, **kwargs):
        list(items)
        seen["guard"] = kwargs.get("guard")
        return IngestStats()

    monkeypatch.setattr(cli, "run_ingest", fake_run_ingest)
    assert cli.main(argv) == 0
    assert callable(seen["guard"])
    before = len(conn.executed)
    seen["guard"]()
    (sql, params), = conn.executed[before:]
    assert "FROM pg_locks" in sql and params[2] == "ShareLock"
