import json

from data_orchestrator.cli import _build_report
from data_orchestrator.core import IngestStats


def _stats(**over):
    s = IngestStats()
    for k, v in over.items():
        setattr(s, k, v)
    return s


def test_report_shape_and_ok_outcome():
    s = _stats(docs_seen=5, docs_ingested=4, docs_error=1,
               by_source={"us": {"docs_seen": 5, "docs_new": 4, "docs_failed": 1}})
    rep = _build_report("vault", s, "2026-08-31T00:00:00+00:00")
    assert rep["tool"] == "data-orchestrator" and rep["command"] == "vault"
    assert rep["outcome"] == "ok" and rep["exit_code"] == 0
    assert rep["totals"] == {"docs_seen": 5, "docs_new": 4, "docs_failed": 1,
                             "docs_path_metadata": 0}
    assert rep["sources"][0]["source_code"] == "us"
    assert set(rep) >= {"run_id", "started_at", "finished_at"}


def test_all_failed_is_degraded_exit_three():
    s = _stats(docs_seen=3, docs_ingested=0, docs_error=3,
               by_source={"us": {"docs_seen": 3, "docs_new": 0, "docs_failed": 3}})
    rep = _build_report("vault", s, "2026-08-31T00:00:00+00:00")
    assert rep["outcome"] == "degraded" and rep["exit_code"] == 3


def test_run_ingest_populates_by_source(tmp_path):
    # reuse the fakes from test_core_engine
    from tests.test_core_engine import FakeEmbedder, FakeStore, RecordingLedger, _md_item
    from data_orchestrator.core import run_ingest

    log = []
    items = [_md_item(tmp_path, doc_id="d1", payload={"bank_code": "us"}),
             _md_item(tmp_path, doc_id="d2", payload={"source_code": "ecb"})]
    stats = run_ingest(items, collection="c", ledger=RecordingLedger(log),
                       store=FakeStore(log), embedder=FakeEmbedder())
    assert stats.by_source["us"]["docs_new"] == 1
    assert stats.by_source["ecb"]["docs_new"] == 1


def test_insert_run_report_sql_contract():
    from data_orchestrator.vault import insert_run_report, _RUN_COLUMNS, _JSON_COLUMNS
    from tests.test_vault_ledger import FakeConn

    report = {"run_id": "r1", "tool": "data-orchestrator",
              "command": "vault", "started_at": "s", "finished_at": "f",
              "outcome": "ok", "exit_code": 0,
              "totals": {}, "sources": []}
    conn = FakeConn()
    insert_run_report(conn, report)
    sql, params = conn.executed[-1]
    assert "INSERT INTO runs" in sql and "ON CONFLICT (run_id) DO NOTHING" in sql

    # Params must line up positionally with _RUN_COLUMNS regardless of
    # column order — a full round-trip decode, not just a spot check.
    by_column = dict(zip(_RUN_COLUMNS, params))
    assert by_column["started_at"] == "s"
    decoded = {
        c: json.loads(v) if c in _JSON_COLUMNS else v
        for c, v in by_column.items()
    }
    assert decoded == report
    assert conn.commits == 1


def test_cb_corpus_no_resume_vault_mode_still_writes_report(monkeypatch, tmp_path):
    """cb_corpus + --no-resume, vault mode (no --no-vault): the resume-ledger
    vault_conn stays None because --no-resume skips the "if not
    args.no_resume" block entirely, so the success-path report write must
    open its own short-lived connection rather than silently doing nothing
    (finding 2 of the fix wave)."""
    from data_orchestrator import cli
    from data_orchestrator.core import IngestStats
    import data_orchestrator.vault as vault_mod
    from tests.test_vault_ledger import FakeConn

    monkeypatch.setenv("CB_CORPUS_ROOT", str(tmp_path))
    monkeypatch.setenv("DATA_ORCHESTRATOR_EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
    (tmp_path / "raw").mkdir()
    (tmp_path / "manifest").mkdir()
    (tmp_path / "manifest" / "us.jsonl").write_text("", encoding="utf-8")

    def fake_run_ingest(items, *, collection, **kwargs):
        return IngestStats(docs_seen=1, docs_ingested=1,
                            by_source={"us": {"docs_seen": 1, "docs_new": 1, "docs_failed": 0}})

    monkeypatch.setattr(cli, "run_ingest", fake_run_ingest)

    class ClosableFakeConn(FakeConn):
        def close(self):
            pass

    conn = ClosableFakeConn()
    monkeypatch.setattr(vault_mod, "connect", lambda: conn)

    rc = cli.main(["cb_corpus", "--no-resume"])

    assert rc == 0
    inserts = [sql for sql, _params in conn.executed if "INSERT INTO runs" in sql]
    assert inserts, "run report was never inserted for --no-resume vault mode"


def test_no_vault_report_append_failure_does_not_mask_clean_run(
    monkeypatch, tmp_path, capsys
):
    """MINOR (item 4 of the fix wave): on the --no-vault success path,
    _append_report_jsonl() was unguarded — a failed append cascaded into the
    fatal handler and flipped a clean run's exit code, contradicting the
    "report write never masks the run's outcome" contract that already
    applies to every vault insert. It must be warn-only, same as those."""
    from data_orchestrator import cli
    from data_orchestrator.core import IngestStats

    monkeypatch.setenv("CB_CORPUS_ROOT", str(tmp_path))
    monkeypatch.setenv("DATA_ORCHESTRATOR_EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
    (tmp_path / "raw").mkdir()
    (tmp_path / "manifest").mkdir()
    (tmp_path / "manifest" / "us.jsonl").write_text("", encoding="utf-8")

    def fake_run_ingest(items, *, collection, **kwargs):
        return IngestStats(docs_seen=1, docs_ingested=1,
                            by_source={"us": {"docs_seen": 1, "docs_new": 1, "docs_failed": 0}})

    def boom(report):
        raise OSError("disk full")

    monkeypatch.setattr(cli, "run_ingest", fake_run_ingest)
    monkeypatch.setattr(cli, "_append_report_jsonl", boom)

    rc = cli.main(["cb_corpus", "--no-vault"])

    assert rc == 0
    assert "warning" in capsys.readouterr().err.lower()


def test_report_totals_carry_the_path_metadata_counter():
    """runs.totals is the vault's only trace of how many documents went in with
    path-derived (year-only) metadata; the key is always present so a Metabase
    card can chart it without null-handling, and it never affects the outcome."""
    s = _stats(docs_seen=5, docs_ingested=5, docs_path_metadata=5,
               by_source={"us": {"docs_seen": 5, "docs_new": 5, "docs_failed": 0}})
    rep = _build_report("cb_corpus", s, "2026-09-03T00:00:00+00:00")
    assert rep["totals"]["docs_path_metadata"] == 5
    assert rep["outcome"] == "ok" and rep["exit_code"] == 0


def test_insert_run_report_sweeps_non_column_fields_into_extra():
    """R4: `_fatal_report` puts the exception text under an ``error`` key, which
    is not a `runs` column — before this, the vault path silently dropped it and
    the message survived only in the ``--no-vault`` JSONL. Non-column keys go to
    ``extra`` (JSONB), the same rule the vault ingester's ``parse_run_line``
    applies to the other writer of this table."""
    from data_orchestrator.cli import _fatal_report
    from data_orchestrator.vault import insert_run_report
    from tests.test_vault_ledger import FakeConn

    rep = _fatal_report("vault", "2026-09-04T00:00:00+00:00",
                        RuntimeError("vault unreachable"))
    conn = FakeConn()
    insert_run_report(conn, rep)

    sql, params = conn.executed[-1]
    assert ("run_id, tool, command, started_at, finished_at, outcome, "
            "exit_code, totals, sources, extra") in sql
    assert sql.count("%s") == 10 and len(params) == 10
    extra = json.loads(params[-1])
    assert extra == {"error": "RuntimeError: vault unreachable"}


def test_insert_run_report_extra_is_null_when_no_extra_fields():
    """An empty sweep must be SQL NULL, not the string ``'{}'`` — matching the
    ingester, so `extra IS NULL` means the same thing for both writers."""
    from data_orchestrator.vault import insert_run_report
    from tests.test_vault_ledger import FakeConn

    conn = FakeConn()
    insert_run_report(conn, {"run_id": "r1", "tool": "data-orchestrator",
                             "command": "vault", "started_at": "s", "finished_at": "f",
                             "outcome": "ok", "exit_code": 0,
                             "totals": {}, "sources": []})
    _sql, params = conn.executed[-1]
    assert params[-1] is None


def test_insert_run_report_absent_columns_become_sql_null():
    """A report missing ``totals``/``sources`` (e.g. an early-failure report)
    must not raise KeyError: absent columns map to SQL NULL, same as the
    vault ingester's absent-key rule."""
    from data_orchestrator.vault import insert_run_report, _RUN_COLUMNS
    from tests.test_vault_ledger import FakeConn

    conn = FakeConn()
    report = {"run_id": "r1", "tool": "data-orchestrator",
              "command": "vault", "started_at": "s", "finished_at": "f",
              "outcome": "error", "exit_code": 1}
    insert_run_report(conn, report)

    _sql, params = conn.executed[-1]
    by_column = dict(zip(_RUN_COLUMNS, params))
    assert by_column["totals"] is None
    assert by_column["sources"] is None
