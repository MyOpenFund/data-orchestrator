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
    resolution as the vault source."""
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
    --no-resume there is an incoherent combo: reject
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
    produce the best-effort failed/1 report, same as every other fatal path."""
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
    already have been created."""
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
    """An unknown --corpus must not surface
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
# CLI wiring: validations, --include-html removal, unsupported count
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
        elif "INSERT INTO rag_ingestions" in sql:
            self._conn.ingestions.append(params)   # VaultLedger.mark
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
        self.ingestions: list = []    # INSERT INTO rag_ingestions params (VaultLedger.mark)
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
    assert rep["sources"] == []  # ecb was not touched by the run
    assert rep["unsupported_by_source"] == {"ecb": 32}
    assert conn.selection_params["supported_mimes"] == list(DEFAULT_SUPPORTED_MIME_TYPES)
    out = capsys.readouterr().out
    assert "Skipping 32 document(s)" in out
    assert conn.closed


def test_the_skipping_line_points_to_the_setting_instead_of_listing_mimes(
        monkeypatch, tmp_path, capsys):
    _conn, _reports = _fake_vault(monkeypatch, tmp_path, [("ecb", 3)])
    assert cli.main(["vault"]) == 0
    (line,) = [l for l in capsys.readouterr().out.splitlines() if l.startswith("Skipping")]
    assert "Skipping 3 document(s)" in line
    assert "DATA_ORCHESTRATOR_SUPPORTED_MIME_TYPES" in line and "README" in line
    assert "application/" not in line


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


def test_vault_count_only_labels_null_source_codes_by_corpus(monkeypatch, tmp_path, capsys):
    """A NULL source_code reads as the corpus in BOTH source breakdowns — the
    label core.source_label gives (and _bump files sources[] under); a NULL
    doc_type reads ``unknown``."""
    rows = [("d1", None, "C1", "a.pdf"), ("d2", "ecb", None, "b.pdf")]
    conn, _reports = _fake_vault(
        monkeypatch, tmp_path, [(None, 3)], selection_rows=rows,
        selection_columns=("doc_id", "source_code", "doc_type", "local_path"),
    )
    assert cli.main(["vault", "--count-only"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert "Matching documents: 2" in lines
    by_source = [line for line in lines if line.startswith("  by source_code:")]
    assert by_source == ["  by source_code: {'central-bank': 1, 'ecb': 1}",
                         "  by source_code: {'central-bank': 3}"]
    assert "  by doc_type: {'C1': 1, 'unknown': 1}" in lines


def test_a_null_source_doc_lands_under_one_label_for_new_and_unsupported(
        monkeypatch, tmp_path):
    """A selected NULL-source row (payload carries corpus, as the real
    selection does) and a NULL-source unsupported count share ONE sources[]
    entry."""
    from data_orchestrator.core import IngestStats, _bump

    (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4 stub")
    _conn, reports = _fake_vault(
        monkeypatch, tmp_path, [(None, 2)],
        selection_rows=[("d1", None, "central-bank", "a.pdf")],
        selection_columns=("doc_id", "source_code", "corpus", "local_path"),
    )

    def fake_run_ingest(items, **kwargs):
        stats = IngestStats()
        for item in items:
            stats.docs_seen += 1
            stats.docs_ingested += 1
            _bump(stats, item, new=1)
        return stats

    monkeypatch.setattr(cli, "run_ingest", fake_run_ingest)
    assert cli.main(["vault"]) == 0
    (rep,) = reports
    assert rep["sources"] == [{"source_code": "central-bank", "docs_seen": 1,
                               "docs_new": 1, "docs_failed": 0, "docs_unsupported": 2}]


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
    assert rep["unsupported_by_source"] == {"ecb": 2}


def test_vault_run_keeps_unsupported_only_sources_out_of_sources(monkeypatch, tmp_path):
    """A touched source carries its own docs_unsupported (0 when none); a
    source with only unsupported documents appears in unsupported_by_source
    only."""
    from data_orchestrator.core import IngestStats, _bump

    (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4 stub")
    _conn, reports = _fake_vault(
        monkeypatch, tmp_path, [("fr", 2)],
        selection_rows=[("d1", "ecb", "a.pdf")],
        selection_columns=("doc_id", "source_code", "local_path"),
    )

    def fake_run_ingest(items, **kwargs):
        stats = IngestStats()
        for item in items:
            stats.docs_seen += 1
            stats.docs_ingested += 1
            _bump(stats, item, new=1)
        return stats

    monkeypatch.setattr(cli, "run_ingest", fake_run_ingest)
    assert cli.main(["vault"]) == 0
    (rep,) = reports
    assert rep["sources"] == [{"source_code": "ecb", "docs_seen": 1, "docs_new": 1,
                               "docs_failed": 0, "docs_unsupported": 0}]
    assert rep["unsupported_by_source"] == {"fr": 2}
    assert rep["totals"]["docs_unsupported"] == 2


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
    assert rep["sources"] == []  # no unsupported-only entries on a fatal report
    assert rep["unsupported_by_source"] == {"ecb": 4, "fr": 1}
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
    assert "unsupported_by_source" not in rep


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
# Traceability (issue #8): which .env was read, and which Qdrant the engine
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


def test_main_says_when_no_dotenv_was_found(tmp_path, capsys):
    assert _cb_count(tmp_path) == 0
    assert capsys.readouterr().out.splitlines()[0] == "Config: no .env file found"


def test_the_config_line_is_printed_for_every_source(capsys):
    cli.main(["vault", "--no-vault"])  # rejected (exit 2) after the load
    assert capsys.readouterr().out.splitlines()[0] == "Config: no .env file found"


# The Qdrant line comes from the CONSTRUCTED store (core.run_ingest's
# on_target): a count-only run never touches Qdrant and prints no Qdrant
# line, for every source.
def _fake_engine(monkeypatch, *, host="localhost", port=6333, store=None):
    """Replace the two defaults of a run (core.build_store / build_embedder):
    the engine's real store over a recording client, the engine's fake embedder.
    Returns the list of settings the factories were called with."""
    import data_orchestrator.core as core
    from tests.test_core_engine import FakeEmbedder, FakeStore

    built = []

    def build_store(settings):
        built.append(settings)
        s = store if store is not None else FakeStore([])
        s._host, s._port = host, port
        return s

    monkeypatch.setattr(core, "build_store", build_store)
    monkeypatch.setattr(core, "build_embedder", lambda settings: FakeEmbedder())
    return built


def _cb_run(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ORCHESTRATOR_STATE_DIR", str(tmp_path / "state"))
    return cli.main(["cb_corpus", "--no-vault",
                     "--root", str(_minimal_cb_corpus(tmp_path))])


def test_cb_count_only_prints_no_qdrant_line(tmp_path, capsys):
    assert _cb_count(tmp_path) == 0
    cap = capsys.readouterr()
    assert not any(line.startswith("Qdrant:") for line in cap.out.splitlines())
    assert "warning:" not in cap.err


def test_company_count_only_prints_no_qdrant_line(fake_company_corpus, capsys):
    assert cli.main(["company", "--count-only"]) == 0
    assert "Qdrant:" not in capsys.readouterr().out


def test_vault_count_only_prints_no_qdrant_line(monkeypatch, tmp_path, capsys):
    _fake_vault(monkeypatch, tmp_path, [])
    assert cli.main(["vault", "--count-only"]) == 0
    assert "Qdrant:" not in capsys.readouterr().out


def test_a_run_prints_the_target_of_the_constructed_store(tmp_path, monkeypatch, capsys):
    """The env says one thing, the store another: the line follows the store
    (what the run writes to), not a second reading of the environment."""
    monkeypatch.setenv("QDRANT_PORT", "1111")
    _fake_engine(monkeypatch, host="qdrant.lan", port=6433)
    assert _cb_run(tmp_path, monkeypatch) == 0
    cap = capsys.readouterr()
    assert "Qdrant: qdrant.lan:6433 (api key: unset)" in cap.out.splitlines()
    assert "warning:" not in cap.err


@pytest.mark.parametrize("port", [6333, "6333", "06333", " 6333 ", "", None])
def test_the_legacy_qdrant_port_draws_a_warning(tmp_path, monkeypatch, capsys, port):
    """The engine's default (6333, the LEGACY store) is compared numerically;
    an empty/absent port reads as the default."""
    _fake_engine(monkeypatch, port=port)
    assert _cb_run(tmp_path, monkeypatch) == 0
    cap = capsys.readouterr()
    assert "Qdrant: localhost:6333 (api key: unset)" in cap.out.splitlines()
    assert "warning:" in cap.err and "6333" in cap.err and "legacy" in cap.err.lower()


def test_an_absent_host_reads_as_the_default(tmp_path, monkeypatch, capsys):
    _fake_engine(monkeypatch, host="", port=6433)
    assert _cb_run(tmp_path, monkeypatch) == 0
    assert "Qdrant: localhost:6433 (api key: unset)" in capsys.readouterr().out.splitlines()


def test_vault_run_prints_the_qdrant_target_before_ingesting(monkeypatch, tmp_path, capsys):
    import data_orchestrator.core as core
    _fake_vault(monkeypatch, tmp_path, [])
    monkeypatch.setattr(cli, "run_ingest", core.run_ingest)  # the real loop
    _fake_engine(monkeypatch, port=6433)
    assert cli.main(["vault"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert "Qdrant: localhost:6433 (api key: unset)" in out
    assert out.index("Qdrant: localhost:6433 (api key: unset)") < out.index("Done.")


def test_the_qdrant_line_says_when_no_api_key_is_set(tmp_path, monkeypatch, capsys):
    _fake_engine(monkeypatch, host="qdrant.lan", port=6433)
    assert _cb_run(tmp_path, monkeypatch) == 0
    assert "Qdrant: qdrant.lan:6433 (api key: unset)" in capsys.readouterr().out.splitlines()


def test_the_qdrant_line_says_when_an_api_key_is_set_and_never_prints_it(
        tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("QDRANT_API_KEY", "s3cr3t-value")
    _fake_engine(monkeypatch, host="qdrant.lan", port=6433)
    assert _cb_run(tmp_path, monkeypatch) == 0
    cap = capsys.readouterr()
    assert "Qdrant: qdrant.lan:6433 (api key: set)" in cap.out.splitlines()
    assert "s3cr3t-value" not in cap.out and "s3cr3t-value" not in cap.err


@pytest.mark.parametrize("raw", ["", "   "])
def test_a_blank_api_key_reads_as_unset(tmp_path, monkeypatch, capsys, raw):
    monkeypatch.setenv("QDRANT_API_KEY", raw)
    _fake_engine(monkeypatch, host="qdrant.lan", port=6433)
    assert _cb_run(tmp_path, monkeypatch) == 0
    assert "Qdrant: qdrant.lan:6433 (api key: unset)" in capsys.readouterr().out.splitlines()


def test_a_missing_explicit_dotenv_is_an_operator_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DATA_ORCHESTRATOR_DOTENV", str(tmp_path / "missing.env"))
    monkeypatch.setattr(cli, "_validate_corpus_root",
                        lambda corpus: pytest.fail("work started"))
    assert cli.main(["vault"]) == 2
    cap = capsys.readouterr()
    assert cap.err.startswith("error:") and "DATA_ORCHESTRATOR_DOTENV" in cap.err
    assert "Config:" not in cap.out


# ---------------------------------------------------------------------------
# probe: flags it does not support are rejected; a DB failure is a clean
# fatal line, never a traceback.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("flags", [["--count-only"], ["--no-vault"], ["--no-resume"],
                                   ["--ledger", "x.jsonl"]], ids=" ".join)
def test_probe_rejects_flags_it_does_not_support(monkeypatch, capsys, flags):
    _forbid_any_work(monkeypatch)
    assert cli.main(["probe", *flags]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error:") and "not supported by probe" in err
    assert flags[0] in err


def test_probe_without_database_url_is_a_clean_fatal(monkeypatch, capsys):
    monkeypatch.setenv("CB_CORPUS_ROOT", "/nonexistent")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert cli.main(["probe"]) == 1
    err = capsys.readouterr().err
    assert "error: fatal: RuntimeError: DATABASE_URL is not set" in err
    assert "Traceback" not in err


def test_probe_query_failure_is_a_clean_fatal_and_closes(monkeypatch, capsys):
    import data_orchestrator.probe as probe_mod
    import data_orchestrator.vault as vault_mod

    conn = _DispatchConn([])
    monkeypatch.setattr(vault_mod, "connect", lambda: conn)

    def boom(*args, **kwargs):
        raise RuntimeError("relation documents does not exist")

    monkeypatch.setattr(probe_mod, "run_probe", boom)
    assert cli.main(["probe"]) == 1
    assert ("error: fatal: RuntimeError: relation documents does not exist"
            in capsys.readouterr().err)
    assert conn.closed


def test_vault_count_only_without_database_url_is_a_clean_fatal(monkeypatch, tmp_path,
                                                                capsys):
    monkeypatch.setenv("CB_CORPUS_ROOT", str(tmp_path))
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert cli.main(["vault", "--count-only"]) == 1
    err = capsys.readouterr().err
    assert "error: fatal: RuntimeError: DATABASE_URL is not set" in err


def test_vault_count_only_query_failure_is_a_clean_fatal_and_closes(monkeypatch, tmp_path,
                                                                    capsys):
    import data_orchestrator.sources.vault as vault_source
    conn, _reports = _fake_vault(monkeypatch, tmp_path, [])

    def boom(*args, **kwargs):
        raise RuntimeError("count failed")

    monkeypatch.setattr(vault_source, "count_unsupported", boom)
    assert cli.main(["vault", "--count-only"]) == 1
    assert "error: fatal: RuntimeError: count failed" in capsys.readouterr().err
    assert conn.closed


# ---------------------------------------------------------------------------
# QDRANT_PORT pre-flight: an ingest run refuses a port that is not a plain
# integer in 1..65535 before any DB/engine work.
# ---------------------------------------------------------------------------
_INGEST_ARGVS = [["vault"], ["cb_corpus", "--no-vault"], ["company", "--no-vault"],
                 ["cb_corpus"], ["company"]]


@pytest.mark.parametrize("argv", _INGEST_ARGVS, ids=" ".join)
@pytest.mark.parametrize("value", ["", "  ", "abc", "0", "65536", "-1", "+6333",
                                   "6433 # comment", "6433.0", "６４３３"])
def test_invalid_qdrant_port_fails_an_ingest_run_before_any_work(monkeypatch, capsys,
                                                                 argv, value):
    monkeypatch.setenv("QDRANT_PORT", value)
    _forbid_any_work(monkeypatch)
    assert cli.main(argv) == 2
    err = capsys.readouterr().err
    assert err.startswith("error:") and "QDRANT_PORT" in err


@pytest.mark.parametrize("value", ["6433", " 6433 ", "1", "65535"])
def test_valid_qdrant_port_passes_the_preflight(value):
    assert cli._qdrant_port_error({"QDRANT_PORT": value}) is None


def test_absent_qdrant_port_passes_the_preflight():
    assert cli._qdrant_port_error({}) is None


@pytest.mark.parametrize("argv", [["vault", "--count-only"], ["probe"],
                                  ["company", "--count-only"]], ids=" ".join)
def test_qdrant_port_is_not_checked_when_qdrant_is_never_touched(monkeypatch, tmp_path,
                                                                 capsys, argv,
                                                                 fake_company_corpus):
    monkeypatch.setenv("QDRANT_PORT", "abc")
    _fake_vault(monkeypatch, tmp_path, [])
    cli.main(argv)
    assert "QDRANT_PORT" not in capsys.readouterr().err


def _last_report(tmp_path):
    import json
    lines = (tmp_path / "state" / "runs.jsonl").read_text(encoding="utf-8").splitlines()
    return json.loads(lines[-1])


def test_a_refused_api_key_exits_1_with_the_reason(tmp_path, monkeypatch, capsys):
    from tests.test_core_engine import _RefusingStore

    _fake_engine(monkeypatch, host="qdrant.lan", port=6433, store=_RefusingStore([], 401))
    assert _cb_run(tmp_path, monkeypatch) == 1
    err = capsys.readouterr().err
    assert ("error: fatal: QdrantAuthError: Qdrant refused the request (HTTP 401): "
            "check QDRANT_API_KEY") in err
    rep = _last_report(tmp_path)
    assert (rep["outcome"], rep["exit_code"]) == ("failed", 1)
    assert rep["fatal"] == ("QdrantAuthError: Qdrant refused the request (HTTP 401): "
                            "check QDRANT_API_KEY")


def _cb_corpus_with_documents(tmp_path, n):
    """A cb_corpus root with ``n`` indexed one-page PDFs of bank ``us``."""
    import json

    import fitz
    root = tmp_path / "cbc"
    folder = root / "raw" / "us" / "C1" / "2015"
    folder.mkdir(parents=True)
    (root / "manifest").mkdir()
    lines = []
    for i in range(n):
        doc = fitz.open()
        doc.new_page().insert_text((72, 72), f"Monetary policy note {i}. " * 30)
        doc.save(folder / f"doc{i}.pdf")
        doc.close()
        lines.append(json.dumps({"doc_id": f"doc{i}", "date": "2015-01-01", "title": "t"}))
    (root / "manifest" / "us.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


def test_a_key_refused_in_the_middle_of_a_run_is_one_fatal_exit_1(tmp_path, monkeypatch, capsys):
    """Backlog: a 401/403 on a write must not show as one failure per
    remaining document (a "degraded" run): it is the run's one fatal."""
    from tests.test_core_engine import _KeyRevokedStore

    _fake_engine(monkeypatch, port=6433, store=_KeyRevokedStore([], good=1, status=401))
    monkeypatch.setenv("DATA_ORCHESTRATOR_STATE_DIR", str(tmp_path / "state"))
    root = _cb_corpus_with_documents(tmp_path, 4)
    assert cli.main(["cb_corpus", "--no-vault", "--root", str(root)]) == 1
    err = capsys.readouterr().err
    assert err.count("Qdrant refused the request (HTTP 401)") == 1
    assert "error: fatal: QdrantAuthError: Qdrant refused the request (HTTP 401): " \
           "check QDRANT_API_KEY (while writing document 'doc1'" in err
    rep = _last_report(tmp_path)
    assert (rep["outcome"], rep["exit_code"]) == ("failed", 1)
    assert rep["fatal"].startswith("QdrantAuthError: Qdrant refused the request (HTTP 401)")
    # The file ledger holds the one document written before the refusal.
    ledger = (tmp_path / "state" / "central-bank-e5b-v1.jsonl").read_text(encoding="utf-8")
    assert ledger.count('"doc_id"') == 1 and '"doc_id": "doc0"' in ledger


def test_a_fatal_in_the_middle_of_a_disk_run_reports_what_was_done_before_it(
        tmp_path, monkeypatch, capsys):
    """The report of a run stopped by a refused key says what the ledger
    says: the documents ingested before the refusal, not zeros."""
    from tests.test_core_engine import _KeyRevokedStore

    _fake_engine(monkeypatch, port=6433, store=_KeyRevokedStore([], good=2, status=401))
    monkeypatch.setenv("DATA_ORCHESTRATOR_STATE_DIR", str(tmp_path / "state"))
    root = _cb_corpus_with_documents(tmp_path, 5)
    assert cli.main(["cb_corpus", "--no-vault", "--root", str(root)]) == 1
    rep = _last_report(tmp_path)
    assert (rep["outcome"], rep["exit_code"]) == ("failed", 1)
    assert rep["fatal"].startswith("QdrantAuthError: Qdrant refused the request (HTTP 401)")
    # doc0, doc1 ingested; doc2 was being written; doc3, doc4 never reached.
    assert rep["totals"] == {"docs_seen": 3, "docs_new": 2, "docs_failed": 0,
                             "docs_path_metadata": 0}
    assert rep["sources"] == [{"source_code": "us", "docs_seen": 3, "docs_new": 2,
                               "docs_failed": 0}]
    ledger = (tmp_path / "state" / "central-bank-e5b-v1.jsonl").read_text(encoding="utf-8")
    assert ledger.count('"doc_id"') == rep["totals"]["docs_new"]
    # The usual summary, then the fatal line.
    cap = capsys.readouterr()
    out = cap.out.splitlines()
    assert "Done." in out
    assert "  docs seen      : 3" in out and "  docs ingested  : 2" in out
    assert any(line.startswith("  time           : ") for line in out)
    assert cap.err.rstrip().splitlines()[-1].startswith(
        "error: fatal: QdrantAuthError: Qdrant refused the request (HTTP 401)")


def test_a_fatal_in_the_middle_of_a_vault_run_reports_what_was_done_before_it(
        monkeypatch, tmp_path, capsys):
    import data_orchestrator.core as core
    from tests.test_core_engine import _KeyRevokedStore, _md_item

    for doc_id in ("d0", "d1", "d2", "d3"):
        _md_item(tmp_path, doc_id=doc_id)  # writes <doc_id>.md
    conn, reports = _fake_vault(
        monkeypatch, tmp_path, [("ecb", 2), ("boj", 1)],
        selection_rows=[("d0", "ecb", "d0.md"), ("d1", "fr", "d1.md"),
                        ("d2", "ecb", "d2.md"), ("d3", "fr", "d3.md")],
        selection_columns=("doc_id", "source_code", "local_path"),
    )
    monkeypatch.setattr(cli, "run_ingest", core.run_ingest)  # the real loop
    _fake_engine(monkeypatch, port=6433, store=_KeyRevokedStore([], good=2, status=403))
    assert cli.main(["vault"]) == 1
    (rep,) = reports
    assert (rep["outcome"], rep["exit_code"]) == ("failed", 1)
    assert rep["fatal"].startswith("QdrantAuthError: Qdrant refused the request (HTTP 403)")
    assert "while writing document 'd2', after 2 document(s) ingested" in rep["fatal"]
    assert rep["totals"] == {"docs_seen": 3, "docs_new": 2, "docs_failed": 0,
                             "docs_path_metadata": 0, "docs_unsupported": 3}
    assert rep["sources"] == [
        {"source_code": "ecb", "docs_seen": 2, "docs_new": 1, "docs_failed": 0,
         "docs_unsupported": 2},
        {"source_code": "fr", "docs_seen": 1, "docs_new": 1, "docs_failed": 0,
         "docs_unsupported": 0},
    ]
    assert rep["unsupported_by_source"] == {"boj": 1, "ecb": 2}
    # rag_ingestions holds exactly the documents the report counts as new.
    assert [params[0] for params in conn.ingestions] == ["d0", "d1"]
    cap = capsys.readouterr()
    out = cap.out.splitlines()
    assert "Done." in out
    assert "  docs seen      : 3" in out and "  docs ingested  : 2" in out
    assert cap.err.rstrip().splitlines()[-1].startswith(
        "error: fatal: QdrantAuthError: Qdrant refused the request (HTTP 403)")
    assert conn.closed


def test_a_fatal_before_any_document_prints_no_summary(tmp_path, monkeypatch, capsys):
    from tests.test_core_engine import _RefusingStore

    _fake_engine(monkeypatch, port=6433, store=_RefusingStore([], 401))
    assert _cb_run(tmp_path, monkeypatch) == 1
    assert "Done." not in capsys.readouterr().out
    rep = _last_report(tmp_path)
    assert rep["totals"]["docs_seen"] == 0 and rep["sources"] == []


def test_a_collection_of_another_dimension_exits_1_and_says_what_to_do(
        tmp_path, monkeypatch, capsys):
    from tests.test_core_engine import FakeStore

    store = FakeStore([])
    store.client.sizes["central-bank-e5b-v1"] = 384
    _fake_engine(monkeypatch, port=6433, store=store)
    assert _cb_run(tmp_path, monkeypatch) == 1
    err = capsys.readouterr().err
    assert ("error: fatal: CollectionMismatchError: collection 'central-bank-e5b-v1': expected "
            "a single unnamed 8-d vector with Cosine distance, found 384-d vectors: "
            "use another --collection") in err
    assert _last_report(tmp_path)["fatal"].endswith("use another --collection")


# ---------------------------------------------------------------------------
# Engine settings: read once, after the .env, before any work; an invalid one
# is ONE operator error (exit 2), never a failure per document.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("argv", _INGEST_ARGVS, ids=" ".join)
@pytest.mark.parametrize("key,value", [
    ("RAG_ENGINE_BATCH_SIZE", "0"),
    ("RAG_ENGINE_BATCH_SIZE", "many"),
    ("RAG_ENGINE_OCR_LANGUAGES", "eng fra"),
    ("RAG_ENGINE_QDRANT_TIMEOUT", "-1"),
    ("RAG_ENGINE_BACKEND", "tensorflow"),
    ("QDRANT_HTTPS", "maybe"),
    ("DATA_ORCHESTRATOR_EMBEDDING_DIM", "big"),
])
def test_an_invalid_engine_setting_fails_an_ingest_run_before_any_work(
        monkeypatch, capsys, argv, key, value):
    monkeypatch.setenv(key, value)
    _forbid_any_work(monkeypatch)
    assert cli.main(argv) == 2
    err = capsys.readouterr().err
    assert err.startswith("error:") and key in err


@pytest.mark.parametrize("argv", _INGEST_ARGVS, ids=" ".join)
@pytest.mark.parametrize("value", ["onnx", " ONNX "])
def test_a_backend_other_than_torch_fails_an_ingest_run_before_any_work(
        monkeypatch, capsys, argv, value):
    """onnx is a valid engine backend (search services), but it is not
    installed here: refused up front, not an ImportError once the run started."""
    monkeypatch.setenv("RAG_ENGINE_BACKEND", value)
    _forbid_any_work(monkeypatch)
    assert cli.main(argv) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: RAG_ENGINE_BACKEND=")
    assert "ingestion uses the torch backend" in err


@pytest.mark.parametrize("value", ["torch", " Torch ", ""])
def test_the_torch_backend_named_explicitly_passes_the_preflight(tmp_path, monkeypatch, value):
    monkeypatch.setenv("RAG_ENGINE_BACKEND", value)
    _fake_engine(monkeypatch, port=6433)
    assert _cb_run(tmp_path, monkeypatch) == 0


@pytest.mark.parametrize("argv", _INGEST_ARGVS, ids=" ".join)
def test_another_model_without_its_dimension_fails_before_any_work(monkeypatch, capsys, argv):
    monkeypatch.setenv("DATA_ORCHESTRATOR_EMBEDDING_MODEL", "org/another-model")
    _forbid_any_work(monkeypatch)
    assert cli.main(argv) == 2
    err = capsys.readouterr().err
    assert "org/another-model" in err and "DATA_ORCHESTRATOR_EMBEDDING_DIM" in err


@pytest.mark.parametrize("argv", [["vault", "--count-only"], ["probe"],
                                  ["company", "--count-only"]], ids=" ".join)
def test_engine_settings_are_not_read_when_the_engine_is_never_used(
        monkeypatch, tmp_path, capsys, argv, fake_company_corpus):
    monkeypatch.setenv("RAG_ENGINE_BATCH_SIZE", "many")
    _fake_vault(monkeypatch, tmp_path, [])
    cli.main(argv)
    assert "RAG_ENGINE_BATCH_SIZE" not in capsys.readouterr().err


def test_a_run_hands_the_settings_it_validated_to_the_engine(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DATA_ORCHESTRATOR_EMBEDDING_MODEL", "org/another-model")
    monkeypatch.setenv("DATA_ORCHESTRATOR_EMBEDDING_DIM", "8")
    monkeypatch.setenv("RAG_ENGINE_OCR_LANGUAGES", "eng+fra")
    built = _fake_engine(monkeypatch, port=6433)
    assert _cb_run(tmp_path, monkeypatch) == 0
    (settings,) = built
    assert (settings.model_name, settings.model_dim, settings.ocr_languages) == (
        "org/another-model", 8, "eng+fra")
    assert "'central-bank-another-model-v1'" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# The OCR line: whether scanned pages will be read, said before the run. The
# engine reads ENABLE_OCR / TESSDATA_PREFIX from the process environment at
# each file; a silent "auto -> never" must be visible.
# ---------------------------------------------------------------------------
def _ocr_lines(capsys):
    return [line for line in capsys.readouterr().out.splitlines() if line.startswith("OCR:")]


def test_the_ocr_line_says_off_when_the_host_switches_are_not_set(tmp_path, monkeypatch, capsys):
    _fake_engine(monkeypatch, port=6433)
    assert _cb_run(tmp_path, monkeypatch) == 0
    assert _ocr_lines(capsys) == ["OCR: off (ENABLE_OCR not true or TESSDATA_PREFIX unset)"]


@pytest.mark.parametrize("env", [
    {"ENABLE_OCR": "true"},                                # no language data
    {"TESSDATA_PREFIX": "/usr/share/tessdata"},            # switch off
    {"ENABLE_OCR": "false", "TESSDATA_PREFIX": "/usr/share/tessdata"},
    {"ENABLE_OCR": "1", "TESSDATA_PREFIX": "/usr/share/tessdata"},  # only "true" turns it on
])
def test_the_ocr_line_says_off_when_one_switch_is_missing(tmp_path, monkeypatch, capsys, env):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    _fake_engine(monkeypatch, port=6433)
    assert _cb_run(tmp_path, monkeypatch) == 0
    assert _ocr_lines(capsys) == ["OCR: off (ENABLE_OCR not true or TESSDATA_PREFIX unset)"]


def test_the_ocr_line_says_available_with_the_languages(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ENABLE_OCR", "True")
    monkeypatch.setenv("TESSDATA_PREFIX", "/usr/share/tessdata")
    _fake_engine(monkeypatch, port=6433)
    assert _cb_run(tmp_path, monkeypatch) == 0
    assert _ocr_lines(capsys) == ["OCR: available (languages: eng)"]


def test_the_ocr_line_names_the_configured_languages(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ENABLE_OCR", "true")
    monkeypatch.setenv("TESSDATA_PREFIX", "/usr/share/tessdata")
    monkeypatch.setenv("RAG_ENGINE_OCR_LANGUAGES", "eng+fra")
    _fake_engine(monkeypatch, port=6433)
    assert _cb_run(tmp_path, monkeypatch) == 0
    assert _ocr_lines(capsys) == ["OCR: available (languages: eng+fra)"]


@pytest.mark.parametrize("mode, line", [
    ("never", "OCR: off (--ocr never)"),
    ("always", "OCR: forced (--ocr always, languages: eng)"),
])
def test_the_ocr_line_follows_an_explicit_mode(tmp_path, monkeypatch, capsys, mode, line):
    monkeypatch.setenv("ENABLE_OCR", "true")
    monkeypatch.setenv("TESSDATA_PREFIX", "/usr/share/tessdata")
    _fake_engine(monkeypatch, port=6433)
    monkeypatch.setenv("DATA_ORCHESTRATOR_STATE_DIR", str(tmp_path / "state"))
    assert cli.main(["cb_corpus", "--no-vault", "--ocr", mode,
                     "--root", str(_minimal_cb_corpus(tmp_path))]) == 0
    assert _ocr_lines(capsys) == [line]


@pytest.mark.parametrize("argv", _INGEST_ARGVS, ids=" ".join)
@pytest.mark.parametrize("env", [{}, {"TESSDATA_PREFIX": "  "}, {"ENABLE_OCR": "true"}],
                         ids=["unset", "blank", "only-the-switch"])
def test_forced_ocr_without_the_language_data_fails_before_any_work(
        monkeypatch, capsys, argv, env):
    """--ocr always needs TESSDATA_PREFIX whatever ENABLE_OCR says: without it
    the parser fails on every PDF, so the run cannot succeed."""
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    _forbid_any_work(monkeypatch)
    assert cli.main(argv + ["--ocr", "always"]) == 2
    cap = capsys.readouterr()
    assert cap.err.startswith("error: --ocr always needs TESSDATA_PREFIX")
    assert "OCR would fail on every PDF" in cap.err
    assert "OCR:" not in cap.out


@pytest.mark.parametrize("mode", ["auto", "never"])
def test_the_other_ocr_modes_do_not_need_the_language_data(tmp_path, monkeypatch, mode):
    _fake_engine(monkeypatch, port=6433)
    monkeypatch.setenv("DATA_ORCHESTRATOR_STATE_DIR", str(tmp_path / "state"))
    assert cli.main(["cb_corpus", "--no-vault", "--ocr", mode,
                     "--root", str(_minimal_cb_corpus(tmp_path))]) == 0


@pytest.mark.parametrize("argv", [["vault", "--count-only"], ["company", "--count-only"]],
                         ids=" ".join)
def test_a_count_only_run_does_not_check_the_language_data(monkeypatch, tmp_path, argv,
                                                           fake_company_corpus):
    _fake_vault(monkeypatch, tmp_path, [])
    assert cli.main(argv + ["--ocr", "always"]) == 0


def test_vault_run_prints_the_ocr_line_before_ingesting(monkeypatch, tmp_path, capsys):
    import data_orchestrator.core as core
    _fake_vault(monkeypatch, tmp_path, [])
    monkeypatch.setattr(cli, "run_ingest", core.run_ingest)  # the real loop
    _fake_engine(monkeypatch, port=6433)
    assert cli.main(["vault"]) == 0
    out = capsys.readouterr().out.splitlines()
    line = "OCR: off (ENABLE_OCR not true or TESSDATA_PREFIX unset)"
    assert out.index(line) < out.index("Done.")


@pytest.mark.parametrize("argv", [["vault", "--count-only"], ["company", "--count-only"]],
                         ids=" ".join)
def test_a_count_only_run_prints_no_ocr_line(monkeypatch, tmp_path, capsys, argv,
                                             fake_company_corpus):
    _fake_vault(monkeypatch, tmp_path, [])
    assert cli.main(argv) == 0
    assert "OCR:" not in capsys.readouterr().out



# ---------------------------------------------------------------------------
# Logging: the engine is a library and configures none. The CLI sends its
# records (and this package's) to stderr, as the engine fork used to do as a
# side effect of being imported.
# ---------------------------------------------------------------------------
@pytest.fixture
def bare_root_logger():
    """The root logger, restored afterwards. pytest installs its own handlers
    on it for the duration of the test call, so the test body empties
    ``handlers`` itself before configuring (basicConfig keeps existing ones)."""
    import logging
    root = logging.getLogger()
    saved = (root.handlers[:], root.level,
             {name: logging.getLogger(name).level for name in cli._NOISY_LOGGERS})
    root.handlers[:] = []
    try:
        yield root
    finally:
        for handler in root.handlers:
            handler.close()
        root.handlers[:] = saved[0]
        root.setLevel(saved[1])
        for name, level in saved[2].items():
            logging.getLogger(name).setLevel(level)


def test_logging_goes_to_stderr_at_info_by_default(bare_root_logger, capsys):
    import logging
    bare_root_logger.handlers.clear()
    cli._setup_logging({})
    logging.getLogger("rag_engine.store").info("Creating Qdrant collection %r", "c")
    logging.getLogger("rag_engine.store").debug("not shown")
    logging.getLogger("httpx").info("HTTP Request: not shown")
    err = capsys.readouterr().err
    assert "INFO      rag_engine.store  Creating Qdrant collection 'c'" in err
    assert "not shown" not in err


@pytest.mark.parametrize("raw, shown", [("warning", False), (" DEBUG ", True), ("", True)])
def test_the_log_level_comes_from_the_environment(bare_root_logger, capsys, raw, shown):
    import logging
    bare_root_logger.handlers.clear()
    cli._setup_logging({"DATA_ORCHESTRATOR_LOG_LEVEL": raw})
    logging.getLogger("rag_engine.embedding").info("Loading embedding model")
    assert ("Loading embedding model" in capsys.readouterr().err) is shown


def test_logging_already_configured_by_a_host_is_left_alone_but_for_the_noisy_loggers(
        bare_root_logger):
    """What the docstring says: with handlers already on the root logger the
    level variable does not reach it; the third-party loggers are still
    raised to WARNING at least."""
    import logging
    bare_root_logger.handlers.clear()
    host = logging.NullHandler()
    bare_root_logger.addHandler(host)
    bare_root_logger.setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.NOTSET)
    cli._setup_logging({"DATA_ORCHESTRATOR_LOG_LEVEL": "DEBUG"})
    assert bare_root_logger.handlers == [host] and bare_root_logger.level == logging.ERROR
    assert logging.getLogger("httpx").level == logging.WARNING


@pytest.mark.parametrize("argv", _ALL_SOURCE_ARGVS, ids=" ".join)
def test_an_unknown_log_level_fails_before_any_work(monkeypatch, capsys, argv):
    monkeypatch.setenv("DATA_ORCHESTRATOR_LOG_LEVEL", "verbose")
    _forbid_any_work(monkeypatch)
    assert cli.main(argv) == 2
    err = capsys.readouterr().err
    assert err.startswith("error:") and "DATA_ORCHESTRATOR_LOG_LEVEL" in err
