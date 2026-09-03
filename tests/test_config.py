"""Unit tests for the self-contained ``.env`` parser in rag_orchestrator.config.

Every other test bypasses this loader with ``monkeypatch.setenv``, so its
parsing loop — the thing that actually configures a real machine — was never
executed by the suite.

The autouse ``isolated_env`` fixture snapshots and restores ``os.environ``, so
the keys these tests let the loader write cannot leak into their neighbours.
"""
import os

from rag_orchestrator import config


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
        "RAGO_TEST_KEY=value\n"
        "\n"
        "# RAGO_TEST_OFF=disabled\n",
    )
    assert config.load_dotenv(env) is True
    assert os.environ["RAGO_TEST_KEY"] == "value"
    # A commented-out setting must leave no trace at all — neither under its own
    # name nor as a key that still carries the "#" (which a bare split would).
    assert not [key for key in os.environ if "RAGO_TEST_OFF" in key]


def test_export_prefix_is_stripped(tmp_path):
    """`.env` files are routinely copy-pasted from shell snippets, so a line
    written ``export KEY=value`` must set KEY, not a key named "export KEY"."""
    env = _env_file(tmp_path, "export RAGO_TEST_EXPORTED=value\n")
    config.load_dotenv(env)
    assert os.environ["RAGO_TEST_EXPORTED"] == "value"
    assert "export RAGO_TEST_EXPORTED" not in os.environ


def test_quoted_values_are_unquoted(tmp_path):
    """Paths with spaces are usually quoted; keeping the quotes would produce a
    root like '"/My Drive/corpus"' that no filesystem call can resolve."""
    env = _env_file(
        tmp_path,
        'RAGO_TEST_DOUBLE="/My Drive/corpus"\n'
        "RAGO_TEST_SINGLE='/My Drive/other'\n",
    )
    config.load_dotenv(env)
    assert os.environ["RAGO_TEST_DOUBLE"] == "/My Drive/corpus"
    assert os.environ["RAGO_TEST_SINGLE"] == "/My Drive/other"


def test_a_preset_environment_value_wins_by_default(tmp_path, monkeypatch):
    """The documented contract: an explicit ``KEY=... rag-orchestrator ...`` (or
    a CLI flag exported into the environment) must beat the ``.env`` on disk,
    otherwise a per-run override silently does nothing."""
    monkeypatch.setenv("RAGO_TEST_KEY", "from-the-shell")
    config.load_dotenv(_env_file(tmp_path, "RAGO_TEST_KEY=from-the-file\n"))
    assert os.environ["RAGO_TEST_KEY"] == "from-the-shell"


def test_override_true_replaces_a_preset_value(tmp_path, monkeypatch):
    """The opt-in half of the same rule: a caller that asks to override must
    actually get the file's value, or the flag is decorative."""
    monkeypatch.setenv("RAGO_TEST_KEY", "from-the-shell")
    config.load_dotenv(_env_file(tmp_path, "RAGO_TEST_KEY=from-the-file\n"), override=True)
    assert os.environ["RAGO_TEST_KEY"] == "from-the-file"


def test_an_explicit_path_is_read_even_after_the_process_wide_load(tmp_path, monkeypatch):
    """Idempotence is scoped to the auto-discovered ``.env`` (the ``_LOADED``
    flag). A caller passing a path explicitly asks for that file to be read,
    whatever happened earlier in the process."""
    monkeypatch.setattr(config, "_LOADED", True)
    assert config.load_dotenv(_env_file(tmp_path, "RAGO_TEST_KEY=value\n")) is True
    assert os.environ["RAGO_TEST_KEY"] == "value"
