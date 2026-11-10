import json

from data_orchestrator.cli import _build_report
from data_orchestrator.core import IngestStats
from tests.test_company_corpus import fake_company_corpus  # noqa: F401 — pytest fixture


def _stats(**over):
    s = IngestStats()
    for k, v in over.items():
        setattr(s, k, v)
    return s


def test_report_shape_and_ok_outcome():
    s = _stats(docs_seen=5, docs_ingested=4, docs_error=1,
               by_source={"us": {"docs_seen": 5, "docs_new": 4, "docs_failed": 1}})
    # 1 failure in 5 is 20 %, above the 0.10 default: an explicit ratio keeps
    # this a shape test (the ratio rule has its own tests below).
    rep = _build_report("vault", s, "2026-08-31T00:00:00+00:00",
                        corpus="central-bank", ratio=0.5)
    assert rep["tool"] == "data-orchestrator" and rep["command"] == "vault"
    assert rep["corpus"] == "central-bank"
    assert "docs_unsupported" not in rep["totals"]
    assert rep["outcome"] == "ok" and rep["exit_code"] == 0
    assert rep["totals"] == {"docs_seen": 5, "docs_new": 4, "docs_failed": 1,
                             "docs_path_metadata": 0}
    assert rep["sources"][0]["source_code"] == "us"
    assert set(rep) >= {"run_id", "started_at", "finished_at"}


def test_all_failed_is_degraded_exit_three():
    s = _stats(docs_seen=3, docs_ingested=0, docs_error=3,
               by_source={"us": {"docs_seen": 3, "docs_new": 0, "docs_failed": 3}})
    rep = _build_report("vault", s, "2026-08-31T00:00:00+00:00", corpus="central-bank")
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
              "command": "vault", "corpus": "central-bank", "started_at": "s", "finished_at": "f",
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
    rep = _build_report("cb_corpus", s, "2026-09-03T00:00:00+00:00", corpus="central-bank")
    assert rep["totals"]["docs_path_metadata"] == 5
    assert rep["outcome"] == "ok" and rep["exit_code"] == 0


def test_insert_run_report_sweeps_non_column_fields_into_extra():
    """R4: `_fatal_report` puts the exception text under a ``fatal`` key, which
    is not a `runs` column — before this, the vault path silently dropped it and
    the message survived only in the ``--no-vault`` JSONL. Non-column keys go to
    ``extra`` (JSONB), the same rule the vault ingester's ``parse_run_line``
    applies to the other writer of this table."""
    from data_orchestrator.cli import _fatal_report
    from data_orchestrator.vault import insert_run_report, _RUN_COLUMNS
    from tests.test_vault_ledger import FakeConn

    rep = _fatal_report("vault", "2026-09-04T00:00:00+00:00",
                        RuntimeError("vault unreachable"), corpus="central-bank")
    conn = FakeConn()
    insert_run_report(conn, rep)

    sql, params = conn.executed[-1]
    assert ("run_id, tool, command, started_at, finished_at, outcome, "
            "exit_code, totals, sources, corpus, extra") in sql
    assert sql.count("%s") == 11 and len(params) == 11
    extra = json.loads(params[-1])
    assert extra == {"fatal": "RuntimeError: vault unreachable"}
    assert dict(zip(_RUN_COLUMNS, params))["corpus"] == "central-bank"


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


def test_failure_ratio_above_threshold_degrades_a_run_that_ingested_something():
    s = _stats(docs_seen=10, docs_ingested=8, docs_error=2)
    rep = _build_report("vault", s, "t", corpus="central-bank", ratio=0.10)
    assert rep["outcome"] == "degraded" and rep["exit_code"] == 3


def test_unsupported_counts_land_in_totals_and_per_source_without_degrading():
    s = _stats(docs_seen=1, docs_ingested=1,
               by_source={"us": {"docs_seen": 1, "docs_new": 1, "docs_failed": 0}})
    rep = _build_report("vault", s, "t", corpus="central-bank",
                        unsupported={"ecb": 32, "us": 1})
    assert rep["outcome"] == "ok"
    assert rep["totals"]["docs_unsupported"] == 33
    by_code = {e["source_code"]: e for e in rep["sources"]}
    assert by_code["ecb"] == {"source_code": "ecb", "docs_seen": 0, "docs_new": 0,
                              "docs_failed": 0, "docs_unsupported": 32}
    assert by_code["us"]["docs_unsupported"] == 1 and by_code["us"]["docs_new"] == 1


def test_unsupported_empty_dict_still_writes_a_zero_total():
    rep = _build_report("vault", _stats(), "t", corpus="central-bank", unsupported={})
    assert rep["totals"]["docs_unsupported"] == 0


def test_build_report_does_not_mutate_the_stats_breakdown():
    s = _stats(by_source={"us": {"docs_seen": 1, "docs_new": 1, "docs_failed": 0}})
    _build_report("vault", s, "t", corpus="central-bank", unsupported={"us": 2})
    assert s.by_source["us"] == {"docs_seen": 1, "docs_new": 1, "docs_failed": 0}


def test_fatal_reason_is_truncated_to_the_shared_cap():
    from data_orchestrator.cli import FATAL_MAX_CHARS, _fatal_report
    rep = _fatal_report("vault", "t", RuntimeError("x" * 2000), corpus="company")
    assert len(rep["fatal"]) == FATAL_MAX_CHARS and "error" not in rep
    assert rep["corpus"] == "company" and rep["exit_code"] == 1


def _capture_no_vault_report(monkeypatch):
    """Fake the engine and capture the --no-vault report append."""
    from data_orchestrator import cli
    from data_orchestrator.core import IngestStats

    def fake_run_ingest(items, *, collection, **kwargs):
        return IngestStats(docs_seen=1, docs_ingested=1,
                           by_source={"us": {"docs_seen": 1, "docs_new": 1, "docs_failed": 0}})

    captured: list[dict] = []
    monkeypatch.setattr(cli, "run_ingest", fake_run_ingest)
    monkeypatch.setattr(cli, "_append_report_jsonl", captured.append)
    return captured


def test_cb_corpus_report_carries_its_implied_corpus(monkeypatch, tmp_path):
    from data_orchestrator import cli

    monkeypatch.setenv("CB_CORPUS_ROOT", str(tmp_path))
    monkeypatch.setenv("DATA_ORCHESTRATOR_EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
    monkeypatch.setenv("DATA_ORCHESTRATOR_STATE_DIR", str(tmp_path / "state"))
    (tmp_path / "raw").mkdir()
    (tmp_path / "manifest").mkdir()
    (tmp_path / "manifest" / "us.jsonl").write_text("", encoding="utf-8")
    captured = _capture_no_vault_report(monkeypatch)

    assert cli.main(["cb_corpus", "--no-vault"]) == 0
    assert [r["corpus"] for r in captured] == ["central-bank"]


def test_company_report_carries_its_implied_corpus(monkeypatch, tmp_path,
                                                   fake_company_corpus):
    from data_orchestrator import cli

    monkeypatch.setenv("DATA_ORCHESTRATOR_EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
    monkeypatch.setenv("DATA_ORCHESTRATOR_STATE_DIR", str(tmp_path / "state"))
    captured = _capture_no_vault_report(monkeypatch)

    assert cli.main(["company", "--no-vault", "--ciks", "320193"]) == 0
    assert [r["corpus"] for r in captured] == ["company"]


def test_fatal_report_from_cli_carries_the_corpus(monkeypatch, tmp_path,
                                                  fake_company_corpus):
    from data_orchestrator import cli

    monkeypatch.setenv("DATA_ORCHESTRATOR_EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
    monkeypatch.setenv("DATA_ORCHESTRATOR_STATE_DIR", str(tmp_path / "state"))

    def boom(*args, **kwargs):
        raise RuntimeError("engine init failed")

    captured: list[dict] = []
    monkeypatch.setattr(cli, "run_ingest", boom)
    monkeypatch.setattr(cli, "_append_report_jsonl", captured.append)

    assert cli.main(["company", "--no-vault"]) == 1
    assert len(captured) == 1
    assert captured[0]["corpus"] == "company"
    assert captured[0]["fatal"] == "RuntimeError: engine init failed"
