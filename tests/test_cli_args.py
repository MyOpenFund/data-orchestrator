"""CLI argument-plumbing tests (no engine, no DB — parser-level behavior)."""
import pytest

from rag_orchestrator import cli


def test_vault_source_is_a_choice():
    with pytest.raises(SystemExit):  # argparse errors exit(2) on bad choice
        cli.main(["not-a-source", "--count-only"])


def test_vault_rejects_no_vault(capsys):
    rc = cli.main(["vault", "--no-vault"])
    assert rc == 2
    assert "--no-vault" in capsys.readouterr().err


def test_default_collection_for_vault_source(monkeypatch):
    seen = {}

    def fake_run(args, items, collection):
        seen["collection"] = collection
        return 0

    monkeypatch.setattr(cli, "_run_vault_source", fake_run)
    monkeypatch.setenv("RAGO_EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
    cli.main(["vault", "--corpus", "central-bank"])
    assert seen["collection"] == "central-bank-e5b-v1"
