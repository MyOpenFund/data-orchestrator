"""Offline tests for the company_corpus source connector.

The connector is a thin shim over ``company_corpus.rag.iter_items``. We fake
that module (so no SEC network, no Chrome render, no real corpus on disk) and
assert the shim: forwards/normalises its arguments, re-wraps each item into this
repo's :class:`SourceItem`, and gives a helpful error when the dependency is
missing. A final test drives the CLI ``--count-only`` path for the new source.
"""
from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from data_orchestrator.core import SourceItem
from data_orchestrator.sources import company_corpus as connector


@dataclass
class _FakeItem:
    """Mirror of company_corpus's SourceItem (doc_id, path, payload)."""

    doc_id: str
    path: Path
    payload: dict = field(default_factory=dict)


@pytest.fixture
def fake_company_corpus(monkeypatch):
    """Install a fake ``company_corpus.rag`` module and capture call kwargs.

    Returns the ``captured`` dict (filled when the shim calls ``iter_items``) and
    lets each test set the items the fake should yield via ``set_items``.
    """
    captured: dict = {}
    state: dict = {"items": []}

    def iter_items(root=None, *, ciks=None, doctypes=None,
                   year_min=None, year_max=None, prefer="pdf"):
        captured.update(
            root=root, ciks=ciks, doctypes=doctypes,
            year_min=year_min, year_max=year_max, prefer=prefer,
        )
        yield from state["items"]

    pkg = types.ModuleType("company_corpus")
    rag = types.ModuleType("company_corpus.rag")
    rag.iter_items = iter_items
    pkg.rag = rag
    monkeypatch.setitem(sys.modules, "company_corpus", pkg)
    monkeypatch.setitem(sys.modules, "company_corpus.rag", rag)

    def set_items(items):
        state["items"] = items

    captured["set_items"] = set_items
    return captured


def test_maps_items_to_sourceitem(fake_company_corpus):
    fake_company_corpus["set_items"]([
        _FakeItem(
            doc_id="320193-10K-2024",
            path=Path("/corpus/aapl/2024/10-K.pdf"),
            payload={"source": "company-corpus", "cik": "320193",
                     "company": "Apple Inc.", "doc_type": "A1",
                     "year": 2024, "url": "https://sec.gov/..."},
        ),
    ])

    out = list(connector.iter_items())

    assert len(out) == 1
    item = out[0]
    assert isinstance(item, SourceItem)
    assert item.doc_id == "320193-10K-2024"
    assert item.path == Path("/corpus/aapl/2024/10-K.pdf")
    assert item.payload["cik"] == "320193"
    assert item.payload["source"] == "company-corpus"


def test_payload_is_copied_not_aliased(fake_company_corpus):
    original = {"source": "company-corpus", "cik": "320193"}
    fake_company_corpus["set_items"]([
        _FakeItem(doc_id="d1", path=Path("/x.pdf"), payload=original),
    ])

    item = next(iter(connector.iter_items()))
    item.payload["cik"] = "MUTATED"
    assert original["cik"] == "320193"  # source payload untouched


def test_ciks_csv_string_is_split(fake_company_corpus):
    fake_company_corpus["set_items"]([])
    list(connector.iter_items(ciks="320193, 789019 ,"))
    assert fake_company_corpus["ciks"] == ["320193", "789019"]


def test_arguments_forwarded(fake_company_corpus):
    fake_company_corpus["set_items"]([])
    list(connector.iter_items(
        root=Path("/data/bu"),
        ciks=["320193"],
        doctypes=["A1", "C"],
        year_min=2020,
        year_max=2025,
        prefer="text",
    ))
    cap = fake_company_corpus
    assert cap["root"] == str(Path("/data/bu"))
    assert cap["ciks"] == ["320193"]
    assert cap["doctypes"] == ["A1", "C"]
    assert cap["year_min"] == 2020
    assert cap["year_max"] == 2025
    assert cap["prefer"] == "text"


def test_default_prefer_is_pdf(fake_company_corpus):
    fake_company_corpus["set_items"]([])
    list(connector.iter_items())
    assert fake_company_corpus["prefer"] == "pdf"


def test_missing_dependency_raises_helpful_error(monkeypatch):
    # Ensure the package is absent, then expect a clear, actionable ImportError.
    monkeypatch.delitem(sys.modules, "company_corpus", raising=False)
    monkeypatch.delitem(sys.modules, "company_corpus.rag", raising=False)
    monkeypatch.setattr(connector, "default_root", lambda: None)

    with pytest.raises(ImportError) as excinfo:
        list(connector.iter_items())
    msg = str(excinfo.value)
    assert "company_corpus" in msg
    assert "git+https://github.com/MyOpenFund/company-corpus" in msg


def test_cli_count_only_groups_by_cik_and_doctype(fake_company_corpus, capsys):
    from data_orchestrator import cli

    fake_company_corpus["set_items"]([
        _FakeItem("d1", Path("/a.pdf"),
                  {"cik": "320193", "doc_type": "A1"}),
        _FakeItem("d2", Path("/b.pdf"),
                  {"cik": "320193", "doc_type": "C1"}),
        _FakeItem("d3", Path("/c.pdf"),
                  {"cik": "789019", "doc_type": "A1"}),
    ])

    rc = cli.main(["company", "--count-only"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Matching documents: 3" in out
    assert "by cik:" in out
    assert "by doc_type:" in out
    assert "320193" in out
