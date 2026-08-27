"""Unit tests for the vault selection source (pure SQL builder + row mapping)."""
from datetime import date
from pathlib import Path

from rag_orchestrator.sources.vault import build_selection_sql, row_to_item


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


def test_row_to_item_maps_payload_and_path():
    item = row_to_item(_row(), Path("/corpus"))
    assert item.doc_id == "d1"
    assert item.path == Path("/corpus/data/raw/us/C1/2015/d1.pdf")
    assert item.payload["source_code"] == "us"
    assert item.payload["date"] == "2015-01-13"      # isoformat string
    assert item.payload["cik"] == "0001"             # extra merged
    assert item.payload["corpus"] == "central-bank"  # column wins over extra


def test_row_to_item_extra_never_overrides_columns():
    item = row_to_item(_row(extra={"source_code": "evil"}), Path("/c"))
    assert item.payload["source_code"] == "us"


def test_row_to_item_null_fields_dropped():
    item = row_to_item(_row(date=None, extra=None, title=None), Path("/c"))
    assert "date" not in item.payload and "title" not in item.payload


def test_row_to_item_null_local_path_is_none():
    assert row_to_item(_row(local_path=None), Path("/c")) is None
