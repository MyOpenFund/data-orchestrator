"""CLI argument-plumbing tests (no engine, no DB — parser-level behavior)."""
import pytest

from data_orchestrator import cli


def test_vault_source_is_a_choice():
    with pytest.raises(SystemExit):  # argparse errors exit(2) on bad choice
        cli.main(["not-a-source", "--count-only"])


def test_vault_rejects_no_vault(capsys):
    rc = cli.main(["vault", "--no-vault"])
    assert rc == 2
    assert "--no-vault" in capsys.readouterr().err


def test_default_collection_for_vault_source(monkeypatch, tmp_path):
    seen = {}

    def fake_run(args, collection):
        seen["collection"] = collection
        return 0

    monkeypatch.setattr(cli, "_run_vault_source", fake_run)
    monkeypatch.setenv("DATA_ORCHESTRATOR_EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
    monkeypatch.setenv("CB_CORPUS_ROOT", str(tmp_path))
    cli.main(["vault", "--corpus", "central-bank"])
    assert seen["collection"] == "central-bank-e5b-v1"


def test_default_collection_for_cb_corpus_source_matches_vault_routing(monkeypatch, tmp_path):
    """The cb_corpus fallback must never default to the legacy 384-d
    'cb_corpus' collection name — it uses the same routing.collection_name
    resolution as the vault source (item 5 of the fix wave)."""
    seen = {}

    def fake_run_ingest(items, *, collection, **kwargs):
        seen["collection"] = collection
        from data_orchestrator.core import IngestStats
        return IngestStats()

    monkeypatch.setattr(cli, "run_ingest", fake_run_ingest)
    monkeypatch.setenv("DATA_ORCHESTRATOR_EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
    monkeypatch.setenv("CB_CORPUS_ROOT", str(tmp_path))
    (tmp_path / "raw").mkdir()
    (tmp_path / "manifest").mkdir()
    (tmp_path / "manifest" / "us.jsonl").write_text("", encoding="utf-8")
    cli.main(["cb_corpus", "--no-vault"])
    assert seen["collection"] == "central-bank-e5b-v1"
    assert "cb_corpus" not in seen["collection"]


def test_vault_rejects_no_resume(capsys):
    """The vault source's resume IS the documents/rag_ingestions anti-join —
    --no-resume there is an incoherent combo (item 4 of the fix wave): reject
    it with a clear error instead of silently ignoring the ledger while the
    anti-join still filters."""
    rc = cli.main(["vault", "--no-resume"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "--no-resume" in err
    assert "vault" in err.lower()


def test_vault_no_resume_rejected_even_with_count_only(capsys):
    rc = cli.main(["vault", "--no-resume", "--count-only"])
    assert rc == 2
    assert "--no-resume" in capsys.readouterr().err


def test_vault_source_connect_failure_returns_fatal_report_not_exception(
    monkeypatch, capsys, tmp_path
):
    """A vault.connect() failure inside _run_vault_source (e.g. DATABASE_URL
    unset) must not escape cli.main() as an uncaught exception — it must
    produce the best-effort failed/1 report, same as every other fatal path
    (finding 1 of the fix wave)."""
    import data_orchestrator.vault as vault_mod

    monkeypatch.setenv("CB_CORPUS_ROOT", str(tmp_path))

    def boom():
        raise RuntimeError("DATABASE_URL is not set")

    monkeypatch.setattr(vault_mod, "connect", boom)

    rc = cli.main(["vault", "--corpus", "central-bank"])

    assert rc == 1
    err = capsys.readouterr().err
    assert "error:" in err or "fatal:" in err
    assert "DATABASE_URL" in err


def test_vault_missing_root_fails_before_any_engine_work(monkeypatch, capsys):
    monkeypatch.delenv("CB_CORPUS_ROOT", raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgresql://nope")
    called = {"connect": False}
    import data_orchestrator.vault as vault_mod
    monkeypatch.setattr(vault_mod, "connect", lambda: called.__setitem__("connect", True))
    rc = cli.main(["vault", "--corpus", "central-bank"])
    assert rc == 1
    assert called["connect"] is False  # failed before touching the DB/engine
    assert "CB_CORPUS_ROOT" in capsys.readouterr().err


def test_progress_every_zero_is_rejected(capsys):
    rc = cli.main(["cb_corpus", "--progress-every", "0"])
    assert rc == 2
    assert "progress-every" in capsys.readouterr().err


def test_cb_corpus_missing_root_fails_before_any_engine_work(monkeypatch, capsys):
    """issue #7's own scenario: a missing/misconfigured CB_CORPUS_ROOT must
    fail before the vault connection or the ingest engine are ever touched.
    cb_corpus's iter_items() is a generator, so without this eager
    pre-flight the root is only validated at its first next() — by then the
    ledger/vault connection is already open and an orphan collection may
    already have been created (item 1 of the fix wave)."""
    monkeypatch.delenv("CB_CORPUS_ROOT", raising=False)
    called = {"connect": False, "run_ingest": False}
    import data_orchestrator.vault as vault_mod

    def fake_connect():
        called["connect"] = True

    def fake_run_ingest(*args, **kwargs):
        called["run_ingest"] = True
        from data_orchestrator.core import IngestStats
        return IngestStats()

    monkeypatch.setattr(vault_mod, "connect", fake_connect)
    monkeypatch.setattr(cli, "run_ingest", fake_run_ingest)

    rc = cli.main(["cb_corpus"])

    assert rc == 1
    assert called["connect"] is False
    assert called["run_ingest"] is False
    assert "CB_CORPUS_ROOT" in capsys.readouterr().err


def test_cb_corpus_explicit_missing_root_fails_before_any_engine_work(
    monkeypatch, tmp_path, capsys
):
    """An explicit --root overrides the env lookup entirely (see
    _build_cb_corpus_items), so a nonexistent --root needs its own eager
    existence check rather than falling through to routing.corpus_root()
    (which would happily resolve a *different*, possibly unset, env var)."""
    missing = tmp_path / "does-not-exist"
    called = {"connect": False, "run_ingest": False}
    import data_orchestrator.vault as vault_mod

    def fake_connect():
        called["connect"] = True

    def fake_run_ingest(*args, **kwargs):
        called["run_ingest"] = True
        from data_orchestrator.core import IngestStats
        return IngestStats()

    monkeypatch.setattr(vault_mod, "connect", fake_connect)
    monkeypatch.setattr(cli, "run_ingest", fake_run_ingest)

    rc = cli.main(["cb_corpus", "--root", str(missing)])

    assert rc == 1
    assert called["connect"] is False
    assert called["run_ingest"] is False
    assert str(missing) in capsys.readouterr().err


def _cb_corpus_root_without_manifest(tmp_path):
    """A --root with a raw/ tree but no manifest/ at all — the layout
    _build_cb_corpus_items always hands to iter_items() with the default
    prefer_manifest=True, so it must fail the manifest requirement."""
    root = tmp_path / "corpus"
    f = root / "raw" / "us" / "C1" / "2015" / "a.pdf"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(b"%PDF-1.4 stub")
    return root


def test_cb_corpus_missing_manifest_fails_before_any_engine_work(
    monkeypatch, tmp_path, capsys
):
    """iter_items() is a generator, so today the missing-manifest
    FileNotFoundError only fires at the first next() — inside run_ingest,
    after the ledger/vault connection and engine are already spun up (issue
    #7: an orphan Qdrant collection may have been created). The CLI must
    catch this eagerly, exactly like the missing-root check just above."""
    root = _cb_corpus_root_without_manifest(tmp_path)
    called = {"connect": False, "run_ingest": False}
    import data_orchestrator.vault as vault_mod

    def fake_connect():
        called["connect"] = True

    def fake_run_ingest(*args, **kwargs):
        called["run_ingest"] = True
        from data_orchestrator.core import IngestStats
        return IngestStats()

    monkeypatch.setattr(vault_mod, "connect", fake_connect)
    monkeypatch.setattr(cli, "run_ingest", fake_run_ingest)

    rc = cli.main(["cb_corpus", "--root", str(root), "--no-vault"])

    assert rc == 1
    assert called["connect"] is False
    assert called["run_ingest"] is False
    err = capsys.readouterr().err
    assert "error:" in err
    assert "manifest" in err


def test_cb_corpus_count_only_missing_manifest_fails_cleanly(tmp_path, capsys):
    """The --count-only loop (cli.py ~341-355) runs outside the run_ingest
    try/except, so before the pre-flight it would let the FileNotFoundError
    escape as a raw traceback. The eager pre-flight must catch it first."""
    root = _cb_corpus_root_without_manifest(tmp_path)

    rc = cli.main(["cb_corpus", "--root", str(root), "--no-vault", "--count-only"])

    assert rc == 1
    err = capsys.readouterr().err
    assert "error:" in err
    assert "manifest" in err
    assert "Traceback" not in err


def test_unknown_corpus_error_message_lists_known_corpora(capsys):
    """MINOR (item 2 of the fix wave): an unknown --corpus must not surface
    as a bare KeyError repr — catch it separately and derive the known list
    from routing.ROUTING rather than hardcoding it."""
    from data_orchestrator.routing import ROUTING

    rc = cli.main(["vault", "--corpus", "not-a-real-corpus"])

    assert rc == 1
    err = capsys.readouterr().err
    known = ", ".join(sorted(ROUTING))
    assert f"error: unknown corpus 'not-a-real-corpus' (known: {known})" in err


def test_summary_warns_on_stderr_when_path_metadata_was_ingested(capsys):
    """Path-derived metadata is a designed fallback, but it must never be
    silent: the summary names the count and the remedy on stderr."""
    from data_orchestrator.cli import _print_summary
    from data_orchestrator.core import IngestStats

    _print_summary(IngestStats(docs_seen=3, docs_ingested=3, docs_path_metadata=2))
    captured = capsys.readouterr()
    assert "path-metadata  : 2" in captured.out
    assert "warning" in captured.err and "2 document(s)" in captured.err
    assert "reindex-from-disk" in captured.err
    assert "--no-resume" in captured.err


def test_summary_is_silent_on_stderr_when_no_path_metadata(capsys):
    from data_orchestrator.cli import _print_summary
    from data_orchestrator.core import IngestStats

    _print_summary(IngestStats(docs_seen=3, docs_ingested=3))
    captured = capsys.readouterr()
    assert "path-metadata  : 0" in captured.out
    assert captured.err == ""


# ---------------------------------------------------------------------------
# Task 6 — CLI wiring: validations, --include-html removal, unsupported count
# ---------------------------------------------------------------------------

# Every source and mode an operator-error check must precede.
_ALL_SOURCE_ARGVS = [
    ["vault"],
    ["vault", "--count-only"],
    ["probe"],
    ["cb_corpus", "--no-vault"],
    ["company", "--no-vault"],
]


def _forbid_any_work(monkeypatch):
    import data_orchestrator.vault as vault_mod
    monkeypatch.setattr(vault_mod, "connect", lambda: pytest.fail("connected"))
    monkeypatch.setattr(cli, "run_ingest",
                        lambda *a, **k: pytest.fail("run_ingest called"))


@pytest.mark.parametrize("argv", _ALL_SOURCE_ARGVS, ids=" ".join)
@pytest.mark.parametrize("value", ["0", "-3"])
def test_limit_below_one_is_rejected(monkeypatch, capsys, argv, value):
    _forbid_any_work(monkeypatch)
    rc = cli.main(argv + ["--limit", value])
    assert rc == 2 and "--limit" in capsys.readouterr().err


def test_include_html_no_longer_exists():
    with pytest.raises(SystemExit) as exc:
        cli.main(["cb_corpus", "--include-html"])
    assert exc.value.code == 2


@pytest.mark.parametrize("argv", _ALL_SOURCE_ARGVS, ids=" ".join)
@pytest.mark.parametrize("key,value", [
    ("DATA_ORCHESTRATOR_DEGRADED_FAILURE_RATIO", "2"),
    ("DATA_ORCHESTRATOR_SUPPORTED_MIME_TYPES", "text/html"),
])
def test_bad_policy_env_fails_before_any_work(monkeypatch, capsys, argv, key, value):
    monkeypatch.setenv(key, value)
    _forbid_any_work(monkeypatch)
    rc = cli.main(argv)
    assert rc == 2 and key in capsys.readouterr().err


def test_vault_no_resume_message_explains_the_fresh_collection_rule(capsys):
    assert cli.main(["vault", "--no-resume"]) == 2
    err = capsys.readouterr().err
    assert "fresh --collection" in err and "stale chunks" in err


class _DispatchCursor:
    """Cursor answering by SQL text: resume ledger, unsupported count, selection."""

    def __init__(self, conn):
        self._conn = conn
        self._rows: list = []
        self.description = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, sql, params=None):
        if self._conn.aborted:  # psycopg: "current transaction is aborted"
            raise RuntimeError("InFailedSqlTransaction: current transaction is aborted")
        self._conn.executed.append((sql, params))
        if "FROM rag_ingestions WHERE collection" in sql:
            self._rows = []                       # resume ledger: nothing done yet
        elif "GROUP BY d.source_code" in sql:
            self._rows = list(self._conn.unsupported)
        elif "INSERT INTO runs" in sql:
            if self._conn.insert_error is not None:
                raise self._conn.insert_error
            self._conn.inserted.append(params)
        else:                                     # the selection
            self._conn.selection_params = params
            self.description = [(c,) for c in self._conn.selection_columns]
            self._rows = list(self._conn.selection_rows)

    def fetchall(self):
        return list(self._rows)

    def __iter__(self):
        return iter(self._rows)


class _DispatchConn:
    def __init__(self, unsupported, selection_rows=(),
                 selection_columns=("doc_id", "local_path")):
        self.unsupported = unsupported
        self.selection_rows = list(selection_rows)
        self.selection_columns = tuple(selection_columns)
        self.executed: list = []
        self.selection_params = None
        self.inserted: list = []      # INSERT INTO runs params (real insert_run_report)
        self.insert_error = None      # raised by every INSERT INTO runs when set
        self.aborted = False          # a failed statement poisons the transaction
        self.rollbacks = 0
        self.closed = False

    def cursor(self):
        return _DispatchCursor(self)

    def commit(self):
        pass

    def rollback(self):
        self.rollbacks += 1
        self.aborted = False

    def close(self):
        self.closed = True


def _fake_vault(monkeypatch, tmp_path, unsupported, **conn_kwargs):
    import data_orchestrator.vault as vault_mod
    from data_orchestrator.core import IngestStats

    monkeypatch.setenv("CB_CORPUS_ROOT", str(tmp_path))
    monkeypatch.setenv("DATA_ORCHESTRATOR_EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
    conn = _DispatchConn(unsupported, **conn_kwargs)
    monkeypatch.setattr(vault_mod, "connect", lambda: conn)
    reports: list[dict] = []
    monkeypatch.setattr(vault_mod, "insert_run_report", lambda _c, rep: reports.append(rep))

    def fake_run_ingest(items, **kwargs):
        list(items)  # drain: the selection query runs
        return IngestStats()

    monkeypatch.setattr(cli, "run_ingest", fake_run_ingest)
    return conn, reports


def test_vault_run_reports_the_unsupported_count(monkeypatch, tmp_path, capsys):
    from data_orchestrator.routing import DEFAULT_SUPPORTED_MIME_TYPES

    conn, reports = _fake_vault(monkeypatch, tmp_path, [("ecb", 32)])
    rc = cli.main(["vault"])
    assert rc == 0
    assert len(reports) == 1
    rep = reports[0]
    assert rep["totals"]["docs_unsupported"] == 32
    assert rep["corpus"] == "central-bank"
    assert {e["source_code"]: e["docs_unsupported"] for e in rep["sources"]} == {"ecb": 32}
    assert conn.selection_params["supported_mimes"] == list(DEFAULT_SUPPORTED_MIME_TYPES)
    assert "Skipping 32 document(s)" in capsys.readouterr().out
    assert conn.closed


def test_vault_run_uses_the_env_narrowed_mimes_for_both_queries(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_ORCHESTRATOR_SUPPORTED_MIME_TYPES", "application/pdf")
    conn, _reports = _fake_vault(monkeypatch, tmp_path, [])
    assert cli.main(["vault", "--source-codes", "ecb", "--doctypes", "C1,A3",
                     "--languages", "en,fr", "--year-min", "2015",
                     "--year-max", "2020"]) == 0
    unsupported_params = [p for s, p in conn.executed if "GROUP BY d.source_code" in s]
    assert len(unsupported_params) == 1
    assert unsupported_params[0] == conn.selection_params  # one filter set, both queries
    params = conn.selection_params
    assert params["supported_mimes"] == ["application/pdf"]
    assert params["source_codes"] == ["ecb"]
    assert params["doctypes"] == ["C1", "A3"]
    assert params["languages"] == ["en", "fr"]
    assert (params["year_min"], params["year_max"]) == (2015, 2020)


def test_vault_count_only_prints_the_unsupported_split(monkeypatch, tmp_path, capsys):
    conn, reports = _fake_vault(monkeypatch, tmp_path, [("ecb", 32), ("fr", 3)])
    rc = cli.main(["vault", "--count-only"])
    assert rc == 0 and reports == []
    out = capsys.readouterr().out
    assert "Matching documents: 0" in out
    assert "Unsupported (not offered to the engine): 35" in out
    assert "by source_code: {'ecb': 32, 'fr': 3}" in out
    assert conn.selection_params["supported_mimes"]


def test_vault_count_only_labels_null_source_codes_unknown(monkeypatch, tmp_path, capsys):
    """A NULL source_code reads ``unknown`` in BOTH breakdowns — the same label
    count_unsupported (and core._bump's last fallback) uses."""
    rows = [("d1", None, "C1", "a.pdf"), ("d2", "ecb", None, "b.pdf")]
    conn, _reports = _fake_vault(
        monkeypatch, tmp_path, [(None, 3)], selection_rows=rows,
        selection_columns=("doc_id", "source_code", "doc_type", "local_path"),
    )
    assert cli.main(["vault", "--count-only"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert "Matching documents: 2" in lines
    by_source = [line for line in lines if line.startswith("  by source_code:")]
    assert by_source == ["  by source_code: {'unknown': 1, 'ecb': 1}",
                         "  by source_code: {'unknown': 3}"]
    assert "  by doc_type: {'C1': 1, 'unknown': 1}" in lines
    assert not any("<unknown>" in line for line in lines)


def test_vault_run_mixes_new_and_unsupported_on_one_source(monkeypatch, tmp_path):
    """One selected row and an unsupported count on the same source_code land
    in ONE sources[] entry carrying both counters."""
    from data_orchestrator.core import IngestStats, _bump

    (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4 stub")
    conn, reports = _fake_vault(
        monkeypatch, tmp_path, [("ecb", 2)],
        selection_rows=[("d1", "ecb", "a.pdf")],
        selection_columns=("doc_id", "source_code", "local_path"),
    )
    seen_items = []

    def fake_run_ingest(items, **kwargs):
        stats = IngestStats()
        for item in items:
            seen_items.append(item)
            stats.docs_seen += 1
            stats.docs_ingested += 1
            _bump(stats, item, new=1)
        return stats

    monkeypatch.setattr(cli, "run_ingest", fake_run_ingest)
    assert cli.main(["vault"]) == 0
    assert [(i.doc_id, i.path) for i in seen_items] == [("d1", tmp_path / "a.pdf")]
    (rep,) = reports
    assert rep["sources"] == [{"source_code": "ecb", "docs_seen": 1, "docs_new": 1,
                               "docs_failed": 0, "docs_unsupported": 2}]
    assert rep["totals"]["docs_new"] == 1 and rep["totals"]["docs_unsupported"] == 2


# ---------------------------------------------------------------------------
# The failure-ratio env reaches the report through cli.main (every source).
# ---------------------------------------------------------------------------
def _one_failure_in_a_hundred(items, **kwargs):
    from data_orchestrator.core import IngestStats
    return IngestStats(docs_seen=100, docs_ingested=99, docs_error=1,
                       by_source={"us": {"docs_seen": 100, "docs_new": 99, "docs_failed": 1}})


@pytest.mark.parametrize("ratio,rc,outcome", [("0", 3, "degraded"), (None, 0, "ok")])
def test_failure_ratio_env_reaches_the_cb_corpus_report(monkeypatch, tmp_path,
                                                        ratio, rc, outcome):
    if ratio is not None:
        monkeypatch.setenv("DATA_ORCHESTRATOR_DEGRADED_FAILURE_RATIO", ratio)
    monkeypatch.setenv("DATA_ORCHESTRATOR_EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
    captured: list[dict] = []
    monkeypatch.setattr(cli, "run_ingest", _one_failure_in_a_hundred)
    monkeypatch.setattr(cli, "_append_report_jsonl", captured.append)
    assert cli.main(["cb_corpus", "--no-vault",
                     "--root", str(_minimal_cb_corpus(tmp_path))]) == rc
    (rep,) = captured
    assert (rep["outcome"], rep["exit_code"]) == (outcome, rc)


@pytest.mark.parametrize("ratio,rc,outcome", [("0", 3, "degraded"), (None, 0, "ok")])
def test_failure_ratio_env_reaches_the_vault_report(monkeypatch, tmp_path,
                                                    ratio, rc, outcome):
    if ratio is not None:
        monkeypatch.setenv("DATA_ORCHESTRATOR_DEGRADED_FAILURE_RATIO", ratio)
    conn, reports = _fake_vault(monkeypatch, tmp_path, [])
    monkeypatch.setattr(cli, "run_ingest", _one_failure_in_a_hundred)
    assert cli.main(["vault"]) == rc
    (rep,) = reports
    assert (rep["outcome"], rep["exit_code"]) == (outcome, rc)
    assert conn.closed


# ---------------------------------------------------------------------------
# Fatal paths: the known unsupported count survives, and the fatal report is
# written on a rolled-back connection (or the failure is said, never silent).
# ---------------------------------------------------------------------------
def _engine_crash(conn=None):
    def boom(items, **kwargs):
        if conn is not None:
            conn.aborted = True  # the failure left the transaction aborted
        raise RuntimeError("engine init failed")
    return boom


def test_vault_fatal_report_keeps_the_known_unsupported_count(monkeypatch, tmp_path, capsys):
    conn, reports = _fake_vault(monkeypatch, tmp_path, [("ecb", 4), ("fr", 1)])
    monkeypatch.setattr(cli, "run_ingest", _engine_crash())
    assert cli.main(["vault"]) == 1
    (rep,) = reports
    assert rep["outcome"] == "failed" and rep["exit_code"] == 1
    assert rep["corpus"] == "central-bank"
    assert rep["fatal"] == "RuntimeError: engine init failed" and "error" not in rep
    assert rep["totals"]["docs_unsupported"] == 5
    assert {e["source_code"]: e["docs_unsupported"] for e in rep["sources"]} == {"ecb": 4, "fr": 1}
    assert "error: fatal: RuntimeError: engine init failed" in capsys.readouterr().err
    assert conn.closed


def test_vault_fatal_report_without_a_known_unsupported_count_omits_it(monkeypatch, tmp_path):
    import data_orchestrator.sources.vault as vault_source
    conn, reports = _fake_vault(monkeypatch, tmp_path, [])

    def count_boom(*args, **kwargs):
        raise RuntimeError("count failed")

    monkeypatch.setattr(vault_source, "count_unsupported", count_boom)
    assert cli.main(["vault"]) == 1
    (rep,) = reports
    assert "docs_unsupported" not in rep["totals"]


def _real_insert_fake_vault(monkeypatch, tmp_path):
    """_fake_vault, but the real insert_run_report runs against the fake conn."""
    import data_orchestrator.vault as vault_mod
    real_insert = vault_mod.insert_run_report
    conn, _reports = _fake_vault(monkeypatch, tmp_path, [])
    monkeypatch.setattr(vault_mod, "insert_run_report", real_insert)
    return conn


def _disk_vault_argv(tmp_path):
    return ["cb_corpus", "--root", str(_minimal_cb_corpus(tmp_path))]


@pytest.mark.parametrize("argv_of", [lambda tmp: ["vault"], _disk_vault_argv],
                         ids=["vault", "cb_corpus-vault-ledger"])
def test_fatal_report_is_inserted_after_a_rollback(monkeypatch, tmp_path, capsys, argv_of):
    conn = _real_insert_fake_vault(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "run_ingest", _engine_crash(conn))
    assert cli.main(argv_of(tmp_path)) == 1
    assert conn.rollbacks == 1
    assert len(conn.inserted) == 1
    assert "failed to write run report" not in capsys.readouterr().err
    assert conn.closed


@pytest.mark.parametrize("argv_of", [lambda tmp: ["vault"], _disk_vault_argv],
                         ids=["vault", "cb_corpus-vault-ledger"])
def test_a_failed_fatal_report_insert_is_said(monkeypatch, tmp_path, capsys, argv_of):
    conn = _real_insert_fake_vault(monkeypatch, tmp_path)
    conn.insert_error = RuntimeError("runs table is gone")
    monkeypatch.setattr(cli, "run_ingest", _engine_crash(conn))
    assert cli.main(argv_of(tmp_path)) == 1
    err = capsys.readouterr().err
    assert "warning: failed to write run report to vault: runs table is gone" in err
    assert "error: fatal: RuntimeError: engine init failed" in err
    assert conn.closed


@pytest.mark.parametrize("argv_of", [lambda tmp: ["vault"], _disk_vault_argv],
                         ids=["vault", "cb_corpus-vault-ledger"])
def test_a_failing_rollback_does_not_mask_the_fatal_outcome(monkeypatch, tmp_path, capsys,
                                                            argv_of):
    conn = _real_insert_fake_vault(monkeypatch, tmp_path)

    def rollback_boom():
        raise RuntimeError("connection lost")

    conn.rollback = rollback_boom
    monkeypatch.setattr(cli, "run_ingest", _engine_crash(conn))
    assert cli.main(argv_of(tmp_path)) == 1
    err = capsys.readouterr().err
    assert "warning: failed to write run report to vault:" in err
    assert "error: fatal: RuntimeError: engine init failed" in err
    assert conn.closed


# ---------------------------------------------------------------------------
# Traceability (#8 review): which .env was read, and which Qdrant the engine
# will write to, are printed before any work.
# ---------------------------------------------------------------------------
def _minimal_cb_corpus(tmp_path):
    root = tmp_path / "cbc"
    (root / "raw").mkdir(parents=True)
    (root / "manifest").mkdir()
    (root / "manifest" / "us.jsonl").write_text("", encoding="utf-8")
    return root


def _cb_count(tmp_path):
    return cli.main(["cb_corpus", "--no-vault", "--count-only",
                     "--root", str(_minimal_cb_corpus(tmp_path))])


def test_main_prints_the_dotenv_it_read_first(tmp_path, monkeypatch, capsys):
    from data_orchestrator import config
    env = tmp_path / ".env"
    env.write_text("QDRANT_PORT=6433\n", encoding="utf-8")
    monkeypatch.setattr(config, "find_dotenv", lambda: env)
    assert _cb_count(tmp_path) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == f"Config: .env = {env}"
    assert out[1] == "Qdrant: localhost:6433"


def test_main_says_when_no_dotenv_was_found(tmp_path, capsys):
    assert _cb_count(tmp_path) == 0
    assert capsys.readouterr().out.splitlines()[0] == "Config: no .env file found"


def test_the_config_line_is_printed_for_every_source(capsys):
    cli.main(["vault", "--no-vault"])  # rejected (exit 2) after the load
    assert capsys.readouterr().out.splitlines()[0] == "Config: no .env file found"


def test_the_legacy_qdrant_port_draws_a_warning(tmp_path, capsys):
    """eigenmind's defaults (localhost:6333) are the LEGACY store."""
    assert _cb_count(tmp_path) == 0
    cap = capsys.readouterr()
    assert "Qdrant: localhost:6333" in cap.out.splitlines()
    assert "warning:" in cap.err and "6333" in cap.err and "legacy" in cap.err.lower()


def test_an_explicit_qdrant_target_draws_no_warning(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("QDRANT_HOST", "qdrant.lan")
    monkeypatch.setenv("QDRANT_PORT", "6433")
    assert _cb_count(tmp_path) == 0
    cap = capsys.readouterr()
    assert "Qdrant: qdrant.lan:6433" in cap.out.splitlines()
    assert "warning:" not in cap.err


def test_company_prints_the_qdrant_target(fake_company_corpus, capsys):
    assert cli.main(["company", "--count-only"]) == 0
    assert "Qdrant: localhost:6333" in capsys.readouterr().out.splitlines()


def test_vault_run_prints_the_qdrant_target_before_engine_work(monkeypatch, capsys):
    seen = {}

    def fake_run(args, collection):
        seen["out"] = capsys.readouterr().out
        return 0

    monkeypatch.setenv("QDRANT_PORT", "6433")
    monkeypatch.setattr(cli, "_run_vault_source", fake_run)
    monkeypatch.setattr(cli, "_validate_corpus_root", lambda corpus: None)
    assert cli.main(["vault"]) == 0
    assert "Qdrant: localhost:6433" in seen["out"].splitlines()


def test_a_missing_explicit_dotenv_is_an_operator_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DATA_ORCHESTRATOR_DOTENV", str(tmp_path / "missing.env"))
    monkeypatch.setattr(cli, "_validate_corpus_root",
                        lambda corpus: pytest.fail("work started"))
    assert cli.main(["vault"]) == 2
    cap = capsys.readouterr()
    assert cap.err.startswith("error:") and "DATA_ORCHESTRATOR_DOTENV" in cap.err
    assert "Config:" not in cap.out
