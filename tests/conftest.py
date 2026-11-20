"""Suite-wide fixtures and shared fakes.

The autouse ``isolated_env`` fixture below is what keeps the unit suite honest:
without it a developer's ``.env`` leaks into every test (issue #8), so a test
that thinks it deleted ``DATABASE_URL`` can still make ``vault.connect()`` dial
a real production database.
"""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Keys a ``.env`` can inject into the process before any test runs, and keys
# the engine reads from the process environment (its settings; ENABLE_OCR /
# TESSDATA_PREFIX at each chunked file). Unit tests must see none of them: each
# either points code at a live service (DATABASE_URL, QDRANT_*), at a
# machine-specific corpus/state folder, or changes what the engine does
# (model, batch size, OCR), which would make assertions machine-dependent.
DOTENV_KEYS = (
    "DATABASE_URL",
    "CB_CORPUS_ROOT",
    "COMPANY_CORPUS_ROOT",
    "QDRANT_HOST",
    "QDRANT_PORT",
    "QDRANT_API_KEY",
    "QDRANT_HTTPS",
    "DATA_ORCHESTRATOR_STATE_DIR",
    "DATA_ORCHESTRATOR_EMBEDDING_MODEL",
    "DATA_ORCHESTRATOR_EMBEDDING_DIM",
    "DATA_ORCHESTRATOR_SUPPORTED_MIME_TYPES",
    "DATA_ORCHESTRATOR_PAYLOAD_DROP_EXTRA_KEYS",
    "DATA_ORCHESTRATOR_DEGRADED_FAILURE_RATIO",
    "DATA_ORCHESTRATOR_DOTENV",
    "DATA_ORCHESTRATOR_LOG_LEVEL",
    "RAG_ENGINE_MODEL",
    "RAG_ENGINE_DIM",
    "RAG_ENGINE_PASSAGE_PREFIX",
    "RAG_ENGINE_QUERY_PREFIX",
    "RAG_ENGINE_BACKEND",
    "RAG_ENGINE_BATCH_SIZE",
    "RAG_ENGINE_OCR_LANGUAGES",
    "RAG_ENGINE_SEARCH_LIMIT",
    "RAG_ENGINE_SEARCH_MAX_LIMIT",
    "RAG_ENGINE_MODEL_CACHE",
    "RAG_ENGINE_PARITY_MIN",
    "RAG_ENGINE_ONNX_THREADS",
    "RAG_ENGINE_QDRANT_TIMEOUT",
    "ENABLE_OCR",
    "TESSDATA_PREFIX",
)


@pytest.fixture(autouse=True)
def isolated_env(request):
    """Guarantee, for every unit test:

    1. none of ``DOTENV_KEYS`` is set when the test body starts, whatever a
       ``.env`` on disk or the developer's shell put in the environment;
    2. ``config.load_dotenv()`` called from inside production code
       (``vault.connect``, ``get_path``) finds no file and reads none —
       ``find_dotenv`` and the process-wide ``_LOADED`` flag are both pinned, so
       the outcome does not depend on which test file ran first (that
       order-dependence is what made issue #8 intermittent);
    3. ``os.environ`` is byte-for-byte what it was before the test, once the
       test AND its own fixtures are done — including keys written directly by
       ``config.load_dotenv``, which bypasses monkeypatch by design.

    Guarantee 3 is why this fixture must NOT request the shared ``monkeypatch``
    fixture. Doing so makes monkeypatch a dependency, so it is set up first and
    torn down LAST: a test's own ``monkeypatch.setenv("CB_CORPUS_ROOT", ...)``
    recorded the key as absent (this fixture had just dropped it), and its undo
    then deleted the value the restore below had just put back. With a private
    ``MonkeyPatch`` instance, this fixture is set up before the test's
    ``monkeypatch`` and torn down after it, so the restore is always the last
    word.

    Integration tests are exempt: they configure their own environment against
    throwaway containers (``tests/integration/conftest.py``).
    """
    if request.node.get_closest_marker("integration"):
        yield
        return

    from data_orchestrator import config

    saved = dict(os.environ)
    patch = pytest.MonkeyPatch()
    try:
        patch.setattr(config, "find_dotenv", lambda: None)
        patch.setattr(config, "_LOADED", False)
        patch.setattr(config, "_LOADED_PATH", None)
        for key in DOTENV_KEYS:
            patch.delenv(key, raising=False)
        yield
    finally:
        patch.undo()
        os.environ.clear()
        os.environ.update(saved)


class DescribedFakeCursor:
    """psycopg2-like cursor over canned rows, exposing ``description``.

    The real driver reports column names through ``cursor.description``
    (name in slot 0); ``sources.vault.iter_items`` zips it with each row tuple
    to rebuild dicts, so a fake without it cannot exercise that path.
    """

    def __init__(self, conn):
        self._conn = conn
        self._schema_read = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, sql, params=None):
        self._conn.executed.append((" ".join(sql.split()), params))
        # The read of the column of the document's address
        # (``vault.document_url_column``) answers from ``schema_columns``.
        self._schema_read = "information_schema.columns" in sql

    @property
    def description(self):
        if self._conn.columns is None:
            return None
        return [(name, None, None, None, None, None, None) for name in self._conn.columns]

    def fetchall(self):
        if self._schema_read:
            return [(c,) for c in self._conn.schema_columns]
        return list(self._conn.rows)

    def __iter__(self):
        return iter(self._conn.rows)


class DescribedFakeConn:
    """Fake connection recording the SQL it was given, commits included.

    ``commit()`` appends a ``("COMMIT", None)`` entry to ``executed`` so tests
    can assert *where* in the statement stream a commit happened, not merely
    how many there were.
    """

    def __init__(self, rows=(), columns=None, schema_columns=("document_url",)):
        self.rows = list(rows)
        self.columns = list(columns) if columns is not None else None
        #: The ``documents`` columns information_schema reports for the
        #: document's address (``document_url``, or ``pdf_url`` before the rename).
        self.schema_columns = tuple(schema_columns)
        self.executed = []
        self.commits = 0

    def cursor(self):
        return DescribedFakeCursor(self)

    def commit(self):
        self.commits += 1
        self.executed.append(("COMMIT", None))
