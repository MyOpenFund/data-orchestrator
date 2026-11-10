"""The two ``DATA_ORCHESTRATOR_*`` environment keys, and the clean break behind them.

The prefix used to be ``RAGO_``, from the repo's old name. Pre-release, the old
names get no compat window at all: they are plain unknown keys, read by nobody.
That is a deliberate choice with a sharp edge — an unset key here is not an
error, it silently reverts to a built-in default (a different embedding model,
hence a different collection name, hence an apparently empty index) — so the
"old name does nothing" half is pinned by a test just as firmly as the
"new name works" half, and the two call sites are driven through the real
production code rather than through a helper.
"""
import warnings
from pathlib import Path

from data_orchestrator import cli, routing


def test_routing_reads_the_new_embedding_model_env(monkeypatch):
    """Driven through the real ``embedding_model_name()``, not a helper: the
    constant and its call site have to line up for the rename to reach the
    collection name that every ingest writes into.
    """
    monkeypatch.setenv("DATA_ORCHESTRATOR_EMBEDDING_MODEL", "sentence-transformers/new")
    assert routing.embedding_model_name() == "sentence-transformers/new"
    assert routing.EMBEDDING_MODEL_ENV == "DATA_ORCHESTRATOR_EMBEDDING_MODEL"


def test_the_legacy_embedding_model_env_is_ignored(monkeypatch):
    """No compat fallback: a leftover ``RAGO_EMBEDDING_MODEL`` must be inert.

    If it were still honoured, a half-migrated machine would keep embedding with
    the old override while believing it had migrated. Inert means *silently*
    inert too — no shim is left to warn from, so a warning raised here would
    mean one had crept back in.
    """
    monkeypatch.delenv("DATA_ORCHESTRATOR_EMBEDDING_MODEL", raising=False)
    monkeypatch.setenv("RAGO_EMBEDDING_MODEL", "sentence-transformers/stale-legacy")

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning becomes a test failure
        model = routing.embedding_model_name()

    assert model == routing.DEFAULT_EMBEDDING_MODEL
    assert model == "intfloat/multilingual-e5-base"


def test_state_dir_honours_the_new_env_name(monkeypatch, tmp_path):
    """``cli.state_dir()`` reads the new key, driven through the real function."""
    monkeypatch.setenv("DATA_ORCHESTRATOR_STATE_DIR", str(tmp_path / "new-state"))
    assert cli.state_dir() == tmp_path / "new-state"


def test_state_dir_reads_the_env_at_call_time(monkeypatch, tmp_path):
    """No import-time snapshot: a key set (e.g. by ``config.load_dotenv`` at the
    top of ``main()``) AFTER ``cli`` was imported must still be honoured, and
    unsetting it must bring the default back without any module reload.
    """
    default = Path(cli.__file__).resolve().parents[1] / "state"
    assert cli.state_dir() == default
    monkeypatch.setenv("DATA_ORCHESTRATOR_STATE_DIR", str(tmp_path / "late"))
    assert cli.state_dir() == tmp_path / "late"
    monkeypatch.delenv("DATA_ORCHESTRATOR_STATE_DIR")
    assert cli.state_dir() == default
    assert not hasattr(cli, "STATE_DIR")


def test_the_legacy_state_dir_env_is_ignored(monkeypatch, tmp_path):
    """The ledger directory's half of the same clean break: a leftover
    ``RAGO_STATE_DIR`` must be inert, and silently so.

    If it were still honoured the resume ledger would keep being written to the
    pre-rename location while the operator believed the new key was in charge —
    the kind of split-brain state that only shows up as a surprise re-ingest.
    Asserting the *default* (not merely "not the legacy path") is what makes the
    test fail loudly if the function stops reading an env key at all.
    """
    monkeypatch.delenv("DATA_ORCHESTRATOR_STATE_DIR", raising=False)
    monkeypatch.setenv("RAGO_STATE_DIR", str(tmp_path / "legacy-state"))

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning becomes a test failure
        resolved = cli.state_dir()
    assert resolved == Path(cli.__file__).resolve().parents[1] / "state"
    assert resolved != tmp_path / "legacy-state"


def test_state_dir_expands_a_tilde(monkeypatch, tmp_path):
    """A ``.env`` value is literal (no shell expansion), so ``~/state`` must be
    expanded by the code, not written to a folder literally named ``~``."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("DATA_ORCHESTRATOR_STATE_DIR", "~/do-state")
    assert cli.state_dir() == tmp_path / "do-state"
