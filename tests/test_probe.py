"""Unit tests for the facts probe (pymupdf-generated fixtures, fake conn)."""
from pathlib import Path

from rag_orchestrator.probe import probe_file, run_probe


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
