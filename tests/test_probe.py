"""Unit tests for the facts probe (pymupdf-generated fixtures, fake conn)."""
import logging
from pathlib import Path

from data_orchestrator.probe import probe_file, run_probe

from .conftest import DescribedFakeConn


def _pdf(tmp_path, name, with_text):
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    if with_text:
        page.insert_text((72, 72), "hello world")
    doc.new_page()  # second, always blank
    doc.save(tmp_path / name)
    doc.close()
    return tmp_path / name


def test_probe_text_pdf(tmp_path):
    assert probe_file(_pdf(tmp_path, "t.pdf", True)) == (True, 2)


def test_probe_scanned_like_pdf(tmp_path):
    assert probe_file(_pdf(tmp_path, "s.pdf", False)) == (False, 2)


def test_probe_non_pdf(tmp_path):
    p = tmp_path / "n.md"
    p.write_text("# hi")
    assert probe_file(p) == (True, None)


def test_probe_missing_file(tmp_path):
    assert probe_file(tmp_path / "nope.pdf") == (None, None)


class FakeCursor:
    def __init__(self, conn):
        self._c = conn

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        self._c.executed.append((" ".join(sql.split()), params))

    def fetchall(self):
        return self._c.select_rows


class FakeConn:
    def __init__(self, select_rows):
        self.select_rows = select_rows
        self.executed = []
        self.commits = 0

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        self.commits += 1


def test_run_probe_updates_facts(tmp_path, monkeypatch):
    pdf = _pdf(tmp_path, "t.pdf", True)
    monkeypatch.setenv("CB_CORPUS_ROOT", str(tmp_path))
    conn = FakeConn([("d1", "t.pdf"), ("d2", "gone.pdf")])
    stats = run_probe(conn, "central-bank")
    assert stats == {"probed": 1, "skipped": 1, "errors": 0}
    updates = [e for e in conn.executed if e[0].startswith("UPDATE documents")]
    assert updates[0][1] == (True, 2, "d1")
    assert conn.commits >= 1


class FailingFakeCursor:
    """Like FakeCursor, but raises once when updating a designated doc_id."""

    def __init__(self, conn):
        self._c = conn

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        sql_norm = " ".join(sql.split())
        self._c.executed.append((sql_norm, params))
        if (
            sql_norm.startswith("UPDATE documents")
            and params is not None
            and params[-1] == self._c.fail_doc_id
            and not self._c.already_failed
        ):
            self._c.already_failed = True
            raise RuntimeError("simulated UPDATE failure")

    def fetchall(self):
        return self._c.select_rows


class FailingFakeConn:
    """FakeConn variant whose UPDATE for ``fail_doc_id`` raises exactly once."""

    def __init__(self, select_rows, fail_doc_id):
        self.select_rows = select_rows
        self.executed = []
        self.commits = 0
        self.fail_doc_id = fail_doc_id
        self.already_failed = False

    def cursor(self):
        return FailingFakeCursor(self)

    def commit(self):
        self.commits += 1


def test_run_probe_isolates_failing_doc_with_savepoint(tmp_path, monkeypatch):
    """One doc's UPDATE raising must not stop the next doc from being probed,
    and must leave a SAVEPOINT/ROLLBACK TO SAVEPOINT trail around the failure
    and SAVEPOINT/RELEASE SAVEPOINT around each success."""
    _pdf(tmp_path, "a.pdf", True)
    _pdf(tmp_path, "b.pdf", True)
    _pdf(tmp_path, "c.pdf", True)
    monkeypatch.setenv("CB_CORPUS_ROOT", str(tmp_path))

    conn = FailingFakeConn(
        [("a", "a.pdf"), ("b", "b.pdf"), ("c", "c.pdf")],
        fail_doc_id="b",
    )
    stats = run_probe(conn, "central-bank")

    # "b" errors, but "a" (before it) and "c" (after it) are still probed.
    assert stats == {"probed": 2, "skipped": 0, "errors": 1}

    updates = [
        (i, sql, params)
        for i, (sql, params) in enumerate(conn.executed)
        if sql.startswith("UPDATE documents")
    ]
    updated_doc_ids = [params[-1] for _, _, params in updates]
    assert updated_doc_ids == ["a", "b", "c"]

    sqls = [sql for sql, _ in conn.executed]

    def surrounding(doc_id):
        idx = next(i for i, _, params in updates if params[-1] == doc_id)
        return sqls[idx - 1], sqls[idx + 1]

    # Successes: SAVEPOINT before, RELEASE SAVEPOINT after.
    assert surrounding("a") == ("SAVEPOINT probe_doc", "RELEASE SAVEPOINT probe_doc")
    assert surrounding("c") == ("SAVEPOINT probe_doc", "RELEASE SAVEPOINT probe_doc")

    # Failure: SAVEPOINT before, ROLLBACK TO SAVEPOINT after.
    assert surrounding("b") == ("SAVEPOINT probe_doc", "ROLLBACK TO SAVEPOINT probe_doc")


def _notes(tmp_path, count):
    """`count` probeable non-PDF documents, as (doc_id, local_path) rows."""
    rows = []
    for i in range(count):
        name = f"n{i:03d}.md"
        (tmp_path / name).write_text("note")
        rows.append((f"doc-{i:03d}", name))
    return rows


def _statement_stream(conn):
    """The UPDATEs and COMMITs in the order the connection saw them."""
    return [sql for sql, _ in conn.executed
            if sql.startswith("UPDATE documents") or sql == "COMMIT"]


def test_run_probe_commits_at_every_batch_boundary(tmp_path, monkeypatch):
    """A pass over a real corpus is tens of thousands of documents: it must
    commit every `batch` rows, so an interrupted run keeps the facts it already
    probed instead of throwing the whole transaction away."""
    monkeypatch.setenv("CB_CORPUS_ROOT", str(tmp_path))
    conn = DescribedFakeConn(rows=_notes(tmp_path, 120))

    stats = run_probe(conn, "central-bank")

    assert stats == {"probed": 120, "skipped": 0, "errors": 0}
    stream = _statement_stream(conn)
    assert stream.index("COMMIT") == 50           # the first 50 updates, then a commit
    assert stream.count("COMMIT") == 3            # at 50, at 100, and the final flush
    assert stream[-1] == "COMMIT"                 # the tail of the batch is never lost


def test_run_probe_batch_size_is_configurable(tmp_path, monkeypatch):
    """The batch size is a knob, not a constant baked into the loop."""
    monkeypatch.setenv("CB_CORPUS_ROOT", str(tmp_path))
    conn = DescribedFakeConn(rows=_notes(tmp_path, 6))

    run_probe(conn, "central-bank", batch=2)

    assert _statement_stream(conn).count("COMMIT") == 4  # at 2, 4, 6, then the final


def test_run_probe_limit_truncates_the_probed_rows(tmp_path, monkeypatch):
    """`--limit` is how the pass is tried out on a live corpus without
    committing to a full sweep, so it must cap the work actually done."""
    monkeypatch.setenv("CB_CORPUS_ROOT", str(tmp_path))
    conn = DescribedFakeConn(rows=_notes(tmp_path, 10))

    stats = run_probe(conn, "central-bank", limit=3)

    assert stats == {"probed": 3, "skipped": 0, "errors": 0}
    updates = [params for sql, params in conn.executed if sql.startswith("UPDATE documents")]
    assert [params[-1] for params in updates] == ["doc-000", "doc-001", "doc-002"]


def test_probe_file_survives_a_corrupt_pdf(tmp_path, caplog):
    """Corpora contain truncated downloads. One unreadable PDF must be reported
    as "unknown facts" (and logged), never raise: the alternative is a probe
    pass that dies partway through a 30k-document corpus."""
    corrupt = tmp_path / "truncated.pdf"
    corrupt.write_bytes(b"%PDF-1.4\n" + b"\xde\xad\xbe\xef" * 40 + b"\ntrailer\n")

    with caplog.at_level(logging.WARNING, logger="data_orchestrator.probe"):
        assert probe_file(corrupt) == (None, None)

    assert "probe failed for" in caplog.text
    assert "truncated.pdf" in caplog.text
