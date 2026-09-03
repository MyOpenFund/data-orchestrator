"""Unit tests for the vault selection source (pure SQL builder + row mapping)."""
import logging
from datetime import date

from rag_orchestrator.sources.vault import (
    _SELECT_COLUMNS, build_selection_sql, iter_items, row_to_item,
)

from .conftest import DescribedFakeConn


def test_sql_is_the_anti_join():
    sql, params = build_selection_sql("central-bank", "central-bank-e5b-v1")
    flat = " ".join(sql.split())
    assert "LEFT JOIN rag_ingestions r" in flat
    assert "r.doc_id IS NULL" in flat
    assert "d.deleted_at IS NULL" in flat
    assert params["corpus"] == "central-bank"
    assert params["collection"] == "central-bank-e5b-v1"


def test_sql_filters_compose():
    sql, params = build_selection_sql(
        "central-bank", "c", source_codes=["ecb", "fr"], doctypes=["C1"],
        year_min=2015, year_max=2020, languages=["en"],
    )
    flat = " ".join(sql.split())
    assert "d.source_code = ANY(%(source_codes)s)" in flat
    assert "d.doc_type = ANY(%(doctypes)s)" in flat
    assert "d.year >= %(year_min)s" in flat and "d.year <= %(year_max)s" in flat
    assert "d.language = ANY(%(languages)s)" in flat
    assert params["source_codes"] == ["ecb", "fr"] and params["year_min"] == 2015


def test_sql_no_filters_no_clauses():
    sql, _ = build_selection_sql("central-bank", "c")
    # No filter clauses: no ANY() for sequences, no year range conditions
    assert "ANY(" not in sql
    assert "year >=" not in sql and "year <=" not in sql


def _row(**over):
    row = {
        "doc_id": "d1", "corpus": "central-bank", "source_code": "us",
        "doc_type": "C1", "title": "Speech", "date": date(2015, 1, 13),
        "year": 2015, "language": "en", "sha256": "ff", "provenance": "bis",
        "local_path": "data/raw/us/C1/2015/d1.pdf", "extra": {"cik": "0001"},
    }
    row.update(over)
    return row


def test_row_to_item_maps_payload_and_path(monkeypatch, tmp_path):
    # local_path is repo-root-relative ("data/raw/..."); resolve_local_path
    # (routing) strips "data/" for central-bank before joining with the root.
    monkeypatch.setenv("CB_CORPUS_ROOT", str(tmp_path))
    item = row_to_item(_row(), "central-bank")
    assert item.doc_id == "d1"
    assert item.path == tmp_path / "raw" / "us" / "C1" / "2015" / "d1.pdf"
    assert item.payload["source_code"] == "us"
    assert item.payload["date"] == "2015-01-13"      # isoformat string
    assert item.payload["cik"] == "0001"             # extra merged
    assert item.payload["corpus"] == "central-bank"  # column wins over extra


def test_row_to_item_extra_never_overrides_columns(monkeypatch, tmp_path):
    monkeypatch.setenv("CB_CORPUS_ROOT", str(tmp_path))
    item = row_to_item(_row(extra={"source_code": "evil"}), "central-bank")
    assert item.payload["source_code"] == "us"


def test_row_to_item_null_fields_dropped(monkeypatch, tmp_path):
    monkeypatch.setenv("CB_CORPUS_ROOT", str(tmp_path))
    item = row_to_item(_row(date=None, extra=None, title=None), "central-bank")
    assert "date" not in item.payload and "title" not in item.payload


def test_row_to_item_null_local_path_is_none():
    # No CB_CORPUS_ROOT needed: local_path is NULL, so the function returns
    # before resolving anything.
    assert row_to_item(_row(local_path=None), "central-bank") is None


def _tuple_row(**over):
    """One selection row as the driver returns it: a tuple in _SELECT_COLUMNS order."""
    row = _row(**over)
    return tuple(row[col] for col in _SELECT_COLUMNS)


def test_iter_items_skips_null_local_path_rows_and_says_how_many(
    monkeypatch, tmp_path, caplog
):
    """A vault row can be known but not yet downloaded (local_path NULL). Those
    rows must be dropped rather than yielded as items with a broken path — and
    the count must be logged, because a silent drop looks exactly like a corpus
    that is simply smaller than expected."""
    monkeypatch.setenv("CB_CORPUS_ROOT", str(tmp_path))
    conn = DescribedFakeConn(
        rows=[_tuple_row(doc_id="downloaded"),
              _tuple_row(doc_id="not-downloaded", local_path=None)],
        columns=_SELECT_COLUMNS,
    )

    with caplog.at_level(logging.WARNING, logger="rag_orchestrator.sources.vault"):
        items = list(iter_items(conn, "central-bank", "central-bank-e5b-v1"))

    assert [item.doc_id for item in items] == ["downloaded"]
    assert items[0].path == tmp_path / "raw" / "us" / "C1" / "2015" / "d1.pdf"
    assert "1 row(s) without local_path skipped" in caplog.text


def test_iter_items_stays_quiet_when_every_row_has_a_path(
    monkeypatch, tmp_path, caplog
):
    """The warning must mean something: a healthy selection must not log one,
    or operators learn to ignore it."""
    monkeypatch.setenv("CB_CORPUS_ROOT", str(tmp_path))
    conn = DescribedFakeConn(rows=[_tuple_row()], columns=_SELECT_COLUMNS)

    with caplog.at_level(logging.WARNING, logger="rag_orchestrator.sources.vault"):
        assert len(list(iter_items(conn, "central-bank", "central-bank-e5b-v1"))) == 1

    assert caplog.text == ""
