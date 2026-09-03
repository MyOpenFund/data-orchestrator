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
    assert rep["totals"] == {"docs_seen": 5, "docs_new": 4, "docs_failed": 1}
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
    from data_orchestrator.vault import insert_run_report
    from tests.test_vault_ledger import FakeConn

    conn = FakeConn()
    insert_run_report(conn, {"run_id": "r1", "tool": "data-orchestrator",
                             "command": "vault", "started_at": "s", "finished_at": "f",
                             "outcome": "ok", "exit_code": 0,
                             "totals": {}, "sources": []})
    sql, params = conn.executed[-1]
    assert "INSERT INTO runs" in sql and "ON CONFLICT (run_id) DO NOTHING" in sql
    assert params[0] == "r1"
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
