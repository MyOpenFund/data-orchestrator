"""Every variable a pre-flight can refuse a run for is named in ``.env.example``
and in the README's configuration table: an operator stopped by an ``error:``
naming a variable must find it in both."""
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


@pytest.mark.parametrize("document", [".env.example", "README.md"])
def test_every_variable_a_preflight_can_refuse_is_documented(document):
    text = (ROOT / document).read_text(encoding="utf-8")
    names = sorted(set(CLI_REFUSED) | set(_refused_by_the_engine_settings()))
    assert [name for name in names if name not in text] == []
