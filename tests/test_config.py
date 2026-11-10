"""Unit tests for the self-contained ``.env`` parser in data_orchestrator.config.

Every other test bypasses this loader with ``monkeypatch.setenv``, so its
parsing loop — the thing that actually configures a real machine — was never
executed by the suite.

The autouse ``isolated_env`` fixture snapshots and restores ``os.environ``, so
the keys these tests let the loader write cannot leak into their neighbours.
"""
import os

from data_orchestrator import config


def _env_file(tmp_path, body):
    path = tmp_path / ".env"
    path.write_text(body, encoding="utf-8")
    return path


def test_comments_and_blank_lines_are_skipped(tmp_path):
    """A ``.env`` is a documented file: its comments and spacing must not end
    up in the environment as bogus keys."""
    env = _env_file(
        tmp_path,
        "# a header comment\n"
        "\n"
        "DATA_ORCHESTRATOR_TEST_KEY=value\n"
        "\n"
        "# DATA_ORCHESTRATOR_TEST_OFF=disabled\n",
    )
    assert config.load_dotenv(env) is True
    assert os.environ["DATA_ORCHESTRATOR_TEST_KEY"] == "value"
    # A commented-out setting must leave no trace at all — neither under its own
    # name nor as a key that still carries the "#" (which a bare split would).
    assert not [key for key in os.environ if "DATA_ORCHESTRATOR_TEST_OFF" in key]


def test_export_prefix_is_stripped(tmp_path):
    """`.env` files are routinely copy-pasted from shell snippets, so a line
    written ``export KEY=value`` must set KEY, not a key named "export KEY"."""
    env = _env_file(tmp_path, "export DATA_ORCHESTRATOR_TEST_EXPORTED=value\n")
    config.load_dotenv(env)
    assert os.environ["DATA_ORCHESTRATOR_TEST_EXPORTED"] == "value"
    assert "export DATA_ORCHESTRATOR_TEST_EXPORTED" not in os.environ


def test_quoted_values_are_unquoted(tmp_path):
    """Paths with spaces are usually quoted; keeping the quotes would produce a
    root like '"/My Drive/corpus"' that no filesystem call can resolve."""
    env = _env_file(
        tmp_path,
        'DATA_ORCHESTRATOR_TEST_DOUBLE="/My Drive/corpus"\n'
        "DATA_ORCHESTRATOR_TEST_SINGLE='/My Drive/other'\n",
    )
    config.load_dotenv(env)
    assert os.environ["DATA_ORCHESTRATOR_TEST_DOUBLE"] == "/My Drive/corpus"
    assert os.environ["DATA_ORCHESTRATOR_TEST_SINGLE"] == "/My Drive/other"


def test_a_preset_environment_value_wins_by_default(tmp_path, monkeypatch):
    """The documented contract: an explicit ``KEY=... data-orchestrator ...`` (or
    a CLI flag exported into the environment) must beat the ``.env`` on disk,
    otherwise a per-run override silently does nothing."""
    monkeypatch.setenv("DATA_ORCHESTRATOR_TEST_KEY", "from-the-shell")
    config.load_dotenv(_env_file(tmp_path, "DATA_ORCHESTRATOR_TEST_KEY=from-the-file\n"))
    assert os.environ["DATA_ORCHESTRATOR_TEST_KEY"] == "from-the-shell"


def test_override_true_replaces_a_preset_value(tmp_path, monkeypatch):
    """The opt-in half of the same rule: a caller that asks to override must
    actually get the file's value, or the flag is decorative."""
    monkeypatch.setenv("DATA_ORCHESTRATOR_TEST_KEY", "from-the-shell")
    config.load_dotenv(_env_file(tmp_path, "DATA_ORCHESTRATOR_TEST_KEY=from-the-file\n"), override=True)
    assert os.environ["DATA_ORCHESTRATOR_TEST_KEY"] == "from-the-file"


def test_an_explicit_path_is_read_even_after_the_process_wide_load(tmp_path, monkeypatch):
    """Idempotence is scoped to the auto-discovered ``.env`` (the ``_LOADED``
    flag). A caller passing a path explicitly asks for that file to be read,
    whatever happened earlier in the process."""
    monkeypatch.setattr(config, "_LOADED", True)
    assert config.load_dotenv(_env_file(tmp_path, "DATA_ORCHESTRATOR_TEST_KEY=value\n")) is True
    assert os.environ["DATA_ORCHESTRATOR_TEST_KEY"] == "value"


def test_loaded_dotenv_reports_the_file_the_search_read(tmp_path, monkeypatch):
    env = _env_file(tmp_path, "DATA_ORCHESTRATOR_TEST_KEY=value\n")
    monkeypatch.setattr(config, "find_dotenv", lambda: env)
    assert config.loaded_dotenv() is None
    assert config.load_dotenv() is True
    assert config.loaded_dotenv() == env


def test_loaded_dotenv_is_none_when_no_file_was_found():
    assert config.load_dotenv() is False  # the autouse fixture pins find_dotenv to None
    assert config.loaded_dotenv() is None


def test_explicit_dotenv_env_replaces_the_search(tmp_path, monkeypatch):
    searched = _env_file(tmp_path, "DATA_ORCHESTRATOR_TEST_KEY=searched\n")
    explicit = tmp_path / "explicit.env"
    explicit.write_text("DATA_ORCHESTRATOR_TEST_KEY=explicit\n", encoding="utf-8")
    monkeypatch.setattr(config, "find_dotenv", lambda: searched)
    monkeypatch.setenv(config.DOTENV_ENV, str(explicit))
    monkeypatch.delenv("DATA_ORCHESTRATOR_TEST_KEY", raising=False)
    assert config.load_dotenv() is True
    assert os.environ["DATA_ORCHESTRATOR_TEST_KEY"] == "explicit"
    assert config.loaded_dotenv() == explicit


def test_explicit_dotenv_env_naming_a_missing_file_raises(tmp_path, monkeypatch):
    monkeypatch.setenv(config.DOTENV_ENV, str(tmp_path / "nope.env"))
    import pytest
    with pytest.raises(FileNotFoundError, match="DATA_ORCHESTRATOR_DOTENV"):
        config.load_dotenv()


def test_a_relative_explicit_dotenv_is_loaded_and_reported_absolute(tmp_path, monkeypatch):
    (tmp_path / "conf").mkdir()
    (tmp_path / "conf" / "x.env").write_text("DATA_ORCHESTRATOR_TEST_KEY=rel\n",
                                              encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(config.DOTENV_ENV, "conf/x.env")
    monkeypatch.delenv("DATA_ORCHESTRATOR_TEST_KEY", raising=False)
    assert config.load_dotenv() is True
    assert config.loaded_dotenv() == tmp_path.resolve() / "conf" / "x.env"
    assert config.loaded_dotenv().is_absolute()
    assert os.environ["DATA_ORCHESTRATOR_TEST_KEY"] == "rel"


def test_a_missing_relative_explicit_dotenv_names_the_resolved_path(tmp_path, monkeypatch):
    import pytest
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(config.DOTENV_ENV, "nope.env")
    with pytest.raises(FileNotFoundError, match=str(tmp_path.resolve() / "nope.env")):
        config.load_dotenv()
