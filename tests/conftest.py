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

# Keys a ``.env`` (this repo's, or the eigenmind fork's, which python-dotenv
# loads at ``import eigenmind.config`` time) can inject into the process before
# any test runs. Unit tests must see none of them: each either points code at a
# live service (DATABASE_URL, QDRANT_*) or at a machine-specific corpus/state
# folder that would make assertions machine-dependent.
DOTENV_KEYS = (
    "DATABASE_URL",
    "CB_CORPUS_ROOT",
    "BOTTOM_UP_CORPUS_ROOT",
    "QDRANT_HOST",
    "QDRANT_PORT",
    "RAGO_STATE_DIR",
    "RAGO_EMBEDDING_MODEL",
)


@pytest.fixture(autouse=True)
def isolated_env(request, monkeypatch):
    """Cut every unit test off from any ``.env`` on disk, whatever the run order.

    Two loaders can populate ``os.environ`` behind a test's back:

    * ``rag_orchestrator.config.load_dotenv()``, called from inside production
      code (``vault.connect``, ``get_path``). It is process-idempotent via the
      module-level ``_LOADED`` flag, which is exactly what made issue #8
      order-dependent: whether a test was protected depended on whether an
      earlier test file had already flipped ``_LOADED``. Pinning both the flag
      and the file finder makes the loader a no-op for every test, first or last.
    * python-dotenv, fired at ``import eigenmind.config`` time — i.e. before any
      fixture can run — so the only cure is to drop the keys it may have
      injected, per test.

    The whole environment is snapshotted and restored, so a test that writes to
    ``os.environ`` directly (``config.load_dotenv`` does, by design) cannot leak
    into its neighbours either.

    Integration tests are exempt: they configure their own environment against
    throwaway containers (``tests/integration/conftest.py``).
    """
    if request.node.get_closest_marker("integration"):
        yield
        return

    from rag_orchestrator import config

    saved = dict(os.environ)
    monkeypatch.setattr(config, "find_dotenv", lambda: None)
    monkeypatch.setattr(config, "_LOADED", False)
    for key in DOTENV_KEYS:
        os.environ.pop(key, None)
    try:
        yield
    finally:
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

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, sql, params=None):
        self._conn.executed.append((" ".join(sql.split()), params))

    @property
    def description(self):
        if self._conn.columns is None:
            return None
        return [(name, None, None, None, None, None, None) for name in self._conn.columns]

    def fetchall(self):
        return list(self._conn.rows)

    def __iter__(self):
        return iter(self._conn.rows)


class DescribedFakeConn:
    """Fake connection recording the SQL it was given, commits included.

    ``commit()`` appends a ``("COMMIT", None)`` entry to ``executed`` so tests
    can assert *where* in the statement stream a commit happened, not merely
    how many there were.
    """

    def __init__(self, rows=(), columns=None):
        self.rows = list(rows)
        self.columns = list(columns) if columns is not None else None
        self.executed = []
        self.commits = 0

    def cursor(self):
        return DescribedFakeCursor(self)

    def commit(self):
        self.commits += 1
        self.executed.append(("COMMIT", None))
