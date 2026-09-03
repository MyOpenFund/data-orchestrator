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
