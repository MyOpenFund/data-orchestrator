"""Every variable a pre-flight can refuse a run for has a row in the README's
configuration table and an assignment line (live or commented out) in
``.env.example``: an operator stopped by an ``error:`` naming a variable must
find it in both. A mention in prose does not count."""
import re
from pathlib import Path

import pytest
from rag_engine.config import ConfigError

from data_orchestrator.core import engine_settings
from tests.conftest import DOTENV_KEYS

ROOT = Path(__file__).resolve().parents[1]

# Refused by the CLI's own checks (cli.main): the .env file, the log level,
# the run policy, the supported types, forced OCR without language data.
CLI_REFUSED = (
    "DATA_ORCHESTRATOR_DOTENV",
    "DATA_ORCHESTRATOR_LOG_LEVEL",
    "DATA_ORCHESTRATOR_DEGRADED_FAILURE_RATIO",
    "DATA_ORCHESTRATOR_SUPPORTED_MIME_TYPES",
    "QDRANT_PORT",
    "TESSDATA_PREFIX",
)


def _refused_by_the_engine_settings():
    """The keys ``core.engine_settings`` refuses an invalid value of (found by
    trying one), whatever the engine pinned today reads."""
    refused = []
    for key in DOTENV_KEYS:
        try:
            engine_settings({key: "not valid!"})
        except ConfigError:
            refused.append(key)
    return refused


def test_the_probe_finds_the_settings_known_to_be_refused():
    refused = _refused_by_the_engine_settings()
    for key in ("RAG_ENGINE_BATCH_SIZE", "RAG_ENGINE_BACKEND", "RAG_ENGINE_OCR_LANGUAGES",
                "RAG_ENGINE_QDRANT_TIMEOUT", "QDRANT_HTTPS", "QDRANT_PORT",
                "DATA_ORCHESTRATOR_EMBEDDING_DIM", "DATA_ORCHESTRATOR_EMBEDDING_MODEL"):
        assert key in refused
    assert "QDRANT_HOST" not in refused and "DATABASE_URL" not in refused


CONFIGURATION_TABLE_HEADER = "| key | meaning |"


def readme_configuration_keys(text):
    """The variable names in the first cell of each row of the README's
    configuration table (the one headed ``| key | meaning |``).

    A row belongs to the table only while it carries the header's
    indentation: the table sits in a numbered list, and a row that loses its
    indentation ends both for a Markdown renderer."""
    lines = text.splitlines()
    (start,) = [i for i, line in enumerate(lines)
                if line.strip() == CONFIGURATION_TABLE_HEADER]
    indent = lines[start][:len(lines[start]) - len(lines[start].lstrip())]
    assert re.fullmatch(r"\|(\s*:?-+:?\s*\|)+", lines[start + 1].strip()), lines[start + 1]
    keys = []
    for line in lines[start + 2:]:
        if not line.startswith(indent + "|"):
            break
        first_cell = re.split(r"(?<!\\)\|", line[len(indent):])[1]
        keys.extend(re.findall(r"`([A-Z][A-Z0-9_]*)`", first_cell))
    return keys


def env_example_keys(text):
    """The names ``.env.example`` assigns: ``NAME=value`` lines and the
    commented-out ``# NAME=value`` ones. A name in a sentence of a comment is
    not an assignment."""
    return re.findall(r"^(?:#\s?)?([A-Z][A-Z0-9_]*)=", text, flags=re.MULTILINE)


DOCUMENTED = {
    "README.md": readme_configuration_keys,
    ".env.example": env_example_keys,
}


def test_the_readme_parser_reads_table_rows_only():
    text = (
        "Set `IN_PROSE` first.\n\n"
        "   | key | meaning |\n"
        "   |---|---|\n"
        "   | `A_KEY` | see `IN_A_MEANING_CELL` |\n"
        "   | `B_KEY` / `C_KEY` | a `code \\| pipe` |\n"
        "| `UNINDENTED_KEY` | this row has left the table |\n"
        "   | `AFTER_THE_BREAK` | so has this one |\n"
    )
    assert readme_configuration_keys(text) == ["A_KEY", "B_KEY", "C_KEY"]


def test_the_env_example_parser_reads_assignment_lines_only():
    text = (
        "# IN_PROSE, when set, must be a number.\n"
        "LIVE_KEY=value\n"
        "# COMMENTED_KEY=value\n"
        "#TIGHT_KEY=\n"
        "#   IN_PROSE_TOO=1 is an example inside a paragraph\n"
    )
    assert env_example_keys(text) == ["LIVE_KEY", "COMMENTED_KEY", "TIGHT_KEY"]


@pytest.mark.parametrize("document", sorted(DOCUMENTED))
def test_every_variable_a_preflight_can_refuse_is_documented(document):
    text = (ROOT / document).read_text(encoding="utf-8")
    documented = set(DOCUMENTED[document](text))
    names = sorted(set(CLI_REFUSED) | set(_refused_by_the_engine_settings()))
    assert [name for name in names if name not in documented] == []


def test_the_readme_configuration_table_names_each_variable_once():
    keys = readme_configuration_keys((ROOT / "README.md").read_text(encoding="utf-8"))
    assert sorted(keys) == sorted(set(keys))
