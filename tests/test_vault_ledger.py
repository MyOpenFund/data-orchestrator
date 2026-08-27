"""VaultLedger unit tests against a scripted fake psycopg2 connection."""
import pytest

from rag_orchestrator.vault import VaultLedger, connect


class FakeCursor:
    def __init__(self, conn):
        self._conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, sql, params=None):
        self._conn.executed.append((" ".join(sql.split()), params))

    def fetchall(self):
        return [(d,) for d in self._conn.resume_rows]


class FakeConn:
    def __init__(self, resume_rows=()):
        self.resume_rows = list(resume_rows)
        self.executed = []
        self.commits = 0

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        self.commits += 1


def _ledger(conn):
    return VaultLedger(
        conn, collection="central-bank-e5b-v1", corpus="central-bank",
        embedding_model="intfloat/multilingual-e5-base",
        embedding_version="e5-prefixes-v1",
    )


def test_resume_set_loaded_at_construction():
    conn = FakeConn(resume_rows=["d1", "d2"])
    led = _ledger(conn)
    assert "d1" in led and "d2" in led and "d3" not in led
    assert len(led) == 2
    sql, params = conn.executed[0]
    assert "FROM rag_ingestions" in sql and params == ("central-bank-e5b-v1",)


def test_mark_upserts_with_metadata_and_commits():
    conn = FakeConn()
    led = _ledger(conn)
    led.mark("d9", 42, payload={"bank_code": "us", "title": "x"})
    sql, params = conn.executed[-1]
    assert "INSERT INTO rag_ingestions" in sql
    assert "ON CONFLICT (doc_id, collection) DO UPDATE" in sql
    assert params == (
        "d9", "central-bank-e5b-v1", "central-bank", "us",
        "intfloat/multilingual-e5-base", "e5-prefixes-v1", 42,
    )
    assert conn.commits == 1
    assert "d9" in led  # in-memory set updated too


def test_mark_source_code_precedence():
    conn = FakeConn()
    led = _ledger(conn)
    led.mark("d1", 1, payload={"source_code": "ecb", "bank_code": "wrong"})
    assert conn.executed[-1][1][3] == "ecb"
    led.mark("d2", 1, payload=None)
    assert conn.executed[-1][1][3] is None


def test_connect_requires_database_url(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(RuntimeError):
        connect()


def test_cli_closes_vault_connection_on_run_failure(monkeypatch, tmp_path):
    """Verify vault connection is closed even if run_ingest raises."""
    from rag_orchestrator import cli
    import rag_orchestrator.vault as vault_mod

    class ClosableConn(FakeConn):
        def __init__(self):
            super().__init__()
            self.closed = False

        def close(self):
            self.closed = True

    conn = ClosableConn()
    monkeypatch.setattr(vault_mod, "connect", lambda: conn)

    def boom(*args, **kwargs):
        raise RuntimeError("engine init failed")

    monkeypatch.setattr(cli, "run_ingest", boom)
    monkeypatch.setenv("CB_CORPUS_ROOT", str(tmp_path))

    with pytest.raises(RuntimeError, match="engine init failed"):
        cli.main(["cb_corpus", "--root", str(tmp_path)])

    assert conn.closed
