"""VaultLedger unit tests against a scripted fake psycopg2 connection."""
import pytest

from data_orchestrator.vault import VaultLedger, connect


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

    def fetchone(self):  # vault.lock_collection: the lock is free
        return (True,)


class FakeConn:
    def __init__(self, resume_rows=()):
        self.resume_rows = list(resume_rows)
        self.executed = []
        self.commits = 0
        self.rollbacks = 0

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


class FailingCursor(FakeCursor):
    """A cursor whose execute() raises on the INSERT (the upsert), simulating
    e.g. an FK violation, but behaves normally for the constructor's resume
    SELECT so a VaultLedger can still be built against it."""

    def __init__(self, conn, exc):
        super().__init__(conn)
        self._exc = exc

    def execute(self, sql, params=None):
        self._conn.executed.append((" ".join(sql.split()), params))
        if "INSERT INTO" in sql:
            raise self._exc


class FailOnceConn(FakeConn):
    """Raises ``fail_exc`` from the cursor's INSERT execute(), tracking rollback()."""

    def __init__(self, resume_rows=(), fail_exc=None):
        super().__init__(resume_rows)
        self._fail_exc = fail_exc

    def cursor(self):
        if self._fail_exc is not None:
            return FailingCursor(self, self._fail_exc)
        return FakeCursor(self)


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
    assert "FROM rag_ingestions" in sql and params == ("central-bank-e5b-v1", "central-bank")
    assert "WHERE collection = %s AND corpus = %s" in sql  # a document is (corpus, doc_id)


def test_mark_upserts_with_metadata_and_commits():
    conn = FakeConn()
    led = _ledger(conn)
    led.mark("d9", 42, payload={"source_code": "us", "title": "x"})
    sql, params = conn.executed[-1]
    assert "INSERT INTO rag_ingestions" in sql
    assert "ON CONFLICT (corpus, doc_id, collection) DO UPDATE" in sql
    assert "corpus = EXCLUDED.corpus" not in sql  # part of the key
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
    led.mark("d3", 1, payload={"bank_code": "us"})  # the legacy key is not read
    assert conn.executed[-1][1][3] is None


def test_mark_rolls_back_and_reraises_on_upsert_failure():
    """A raising execute (e.g. an FK violation) must leave the shared
    connection healthy for the *next* mark() call — not aborted — or every
    later mark on this connection raises InFailedSqlTransaction regardless
    of that document's own health."""
    conn = FailOnceConn(fail_exc=RuntimeError("FK violation"))
    led = _ledger(conn)
    with pytest.raises(RuntimeError, match="FK violation"):
        led.mark("poison", 5, payload={"source_code": "us"})
    assert conn.rollbacks == 1
    assert conn.commits == 0
    assert "poison" not in led  # in-memory resume set untouched by the failure


def test_a_vault_before_the_documents_contract_stops_the_run_at_the_first_document():
    """Against a vault without the key (corpus, doc_id, collection), Postgres
    refuses the ON CONFLICT (SQLSTATE 42P10): every document would fail the
    same way after its points were written, so the run stops, saying why."""
    from data_orchestrator.core import FatalRunError
    from data_orchestrator.vault import VaultSchemaTooOld

    class NoMatchingKey(Exception):
        pgcode = "42P10"

    conn = FailOnceConn(fail_exc=NoMatchingKey("no unique or exclusion constraint"))
    led = _ledger(conn)
    with pytest.raises(VaultSchemaTooOld, match="predates the documents contract") as info:
        led.mark("d1", 5, payload={"source_code": "us"})
    assert isinstance(info.value, FatalRunError)
    assert conn.rollbacks == 1 and "d1" not in led


def test_connect_requires_database_url(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(RuntimeError):
        connect()


def test_cli_closes_vault_connection_on_run_failure(monkeypatch, tmp_path):
    """Verify vault connection is closed even if run_ingest raises.

    Per the run-report doctrine, a fatal exception no longer propagates out
    of cli.main: it is caught, turned into a best-effort "failed" report
    (exit 1), and the vault connection is still closed via the same
    try/finally as every other path.
    """
    from data_orchestrator import cli
    import data_orchestrator.vault as vault_mod

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
    (tmp_path / "raw").mkdir()
    (tmp_path / "manifest").mkdir()
    (tmp_path / "manifest" / "us.jsonl").write_text("", encoding="utf-8")

    rc = cli.main(["cb_corpus", "--root", str(tmp_path)])

    assert rc == 1
    assert conn.closed
    # Best-effort failed report was written to the (fake) vault connection.
    sql, params = conn.executed[-1]
    assert "INSERT INTO runs" in sql
    assert params[5] == "failed" and params[6] == 1


# ---------------------------------------------------------------------------
# Preflight: the vault is on the documents contract
# ---------------------------------------------------------------------------
class _CatalogConn(FakeConn):
    def __init__(self, columns, keyed):
        super().__init__()
        self.answer = (list(columns), keyed)

    def cursor(self):
        conn = self

        class _Cursor(FakeCursor):
            def fetchone(self):
                return conn.answer

        return _Cursor(self)


ALL_COLUMNS = ("date_kind", "date_precision", "entity_scheme", "entity_value")


@pytest.mark.real_contract_preflight
def test_the_preflight_passes_on_a_vault_on_the_contract():
    from data_orchestrator.vault import require_contract_schema

    conn = _CatalogConn(ALL_COLUMNS, True)
    require_contract_schema(conn)
    sql, params = conn.executed[0]
    assert "information_schema.columns" in sql and "pg_index" in sql
    assert params["key"] == ["collection", "corpus", "doc_id"]
    assert conn.rollbacks == 1  # read only: the transaction is ended


@pytest.mark.real_contract_preflight
@pytest.mark.parametrize("columns,keyed,match", [
    ((), False, "documents has no column date_precision, date_kind, entity_scheme, "
                "entity_value; rag_ingestions has no key"),
    (ALL_COLUMNS, False, "^rag_ingestions has no key"),
    (("date_precision",), True, "no column date_kind, entity_scheme, entity_value:"),
])
def test_the_preflight_refuses_a_vault_before_the_contract(columns, keyed, match):
    from data_orchestrator.vault import VaultSchemaTooOld, require_contract_schema

    with pytest.raises(VaultSchemaTooOld, match=match) as info:
        require_contract_schema(_CatalogConn(columns, keyed))
    assert "nothing was written" in str(info.value)


@pytest.mark.parametrize("argv", [["vault"], ["vault", "--count-only"], ["cb_corpus"],
                                  ["sync-metadata"]], ids=" ".join)
def test_every_vault_run_refuses_an_old_vault_before_any_qdrant_write(
        monkeypatch, tmp_path, capsys, argv):
    """The vault source, its count, the cb_corpus source with the vault
    ledger and sync-metadata all stop on the preflight, before the engine or
    Qdrant is touched."""
    from data_orchestrator import cli
    import data_orchestrator.vault as vault_mod
    from data_orchestrator.vault import VaultSchemaTooOld

    class Closable(FakeConn):
        def close(self):
            pass

    def old_vault(conn):
        raise VaultSchemaTooOld("rag_ingestions has no key (corpus, doc_id, collection)")

    monkeypatch.setattr(vault_mod, "connect", lambda: Closable())
    monkeypatch.setattr(vault_mod, "require_contract_schema", old_vault)
    monkeypatch.setattr(cli, "run_ingest", lambda *a, **k: pytest.fail("run_ingest called"))
    import data_orchestrator.core as core
    monkeypatch.setattr(core, "build_store", lambda *a, **k: _NoQdrant())
    monkeypatch.setenv("CB_CORPUS_ROOT", str(tmp_path))
    (tmp_path / "raw").mkdir()
    (tmp_path / "manifest").mkdir()
    (tmp_path / "manifest" / "us.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setenv("DATA_ORCHESTRATOR_STATE_DIR", str(tmp_path / "state"))
    assert cli.main(argv) == 1
    assert "VaultSchemaTooOld: rag_ingestions has no key" in capsys.readouterr().err


class _NoQdrant:
    host, port = "qdrant.test", 6433

    def __getattr__(self, name):
        pytest.fail(f"Qdrant touched: {name}")
