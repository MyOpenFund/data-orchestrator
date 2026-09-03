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

# eigenmind runs python-dotenv at ``import eigenmind.config`` time, with its own
# CWD/frame-based search rule. Importing it HERE, at collection, pins that
# injection to a single deterministic moment before any fixture or test runs —
# otherwise it happens at whatever point the first test happens to import the
# engine, and what a test sees depends on collection order.
try:
    import eigenmind.config  # noqa: F401
except ImportError:  # eigenmind is an optional editable checkout
    pass

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
    "DATA_ORCHESTRATOR_STATE_DIR",
    "DATA_ORCHESTRATOR_EMBEDDING_MODEL",
    # Pre-rename names, still honoured by config.env_with_fallback for one
    # release: a developer's un-migrated .env would otherwise reach the code
    # under test through exactly the fallback the new names were meant to
    # sidestep, and the isolation would silently have a hole in it.
    "RAGO_STATE_DIR",
    "RAGO_EMBEDDING_MODEL",
)


@pytest.fixture(autouse=True)
def isolated_env(request):
    """Guarantee, for every unit test:

    1. none of ``DOTENV_KEYS`` is set when the test body starts, whatever a
       ``.env`` on disk or eigenmind's python-dotenv put in the environment;
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
