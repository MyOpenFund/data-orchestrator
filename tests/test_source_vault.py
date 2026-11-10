"""Unit tests for the vault selection source (pure SQL builder + row mapping)."""
import logging
from datetime import date

import pytest

from data_orchestrator.sources.vault import (
    _SELECT_COLUMNS, _SUPPORTED_PREDICATE, _where, build_selection_sql,
    build_unsupported_sql, count_unsupported, iter_items, row_to_item,
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
    # No filter clauses: no ANY() for filter sequences, no year range conditions
    # (the supported-type predicate always carries its own ANY())
    for param in ("source_codes", "doctypes", "languages"):
        assert f"ANY(%({param})s)" not in sql
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

    with caplog.at_level(logging.WARNING, logger="data_orchestrator.sources.vault"):
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

    with caplog.at_level(logging.WARNING, logger="data_orchestrator.sources.vault"):
        assert len(list(iter_items(conn, "central-bank", "central-bank-e5b-v1"))) == 1

    assert caplog.text == ""


def test_selection_requires_a_supported_extension_and_a_null_or_supported_mime():
    sql, params = build_selection_sql("central-bank", "c",
                                      supported_mimes=("application/pdf",))
    flat = " ".join(sql.split())
    # the engine dispatches on the extension: the path test is unconditional
    assert ("lower(rtrim(d.local_path, E' \\t\\r\\n')) "
            "LIKE ANY(%(supported_ext_patterns)s) AND") in flat
    assert "IS NULL OR" in flat and "= ANY(%(supported_mimes)s)" in flat
    assert "btrim(split_part(lower(d.mime_type), ';', 1), E' \\t\\r\\n')" in flat
    assert flat.count("coalesce(") == 1
    assert params["supported_mimes"] == ["application/pdf"]
    assert params["supported_ext_patterns"] == ["%.pdf"]


def test_selection_default_mimes_come_from_routing(monkeypatch):
    monkeypatch.setenv("DATA_ORCHESTRATOR_SUPPORTED_MIME_TYPES", "text/markdown")
    _sql, params = build_selection_sql("central-bank", "c")
    assert params["supported_mimes"] == ["text/markdown"]
    assert params["supported_ext_patterns"] == ["%.md"]


_ALL_FILTERS = dict(source_codes=["ecb", "fr"], doctypes=["A1"], year_min=2015,
                    year_max=2020, languages=["en"], supported_mimes=("application/pdf",))


def _where_clause(sql):
    return sql.split("WHERE ", 1)[1].split("\n", 1)[0]


def test_unsupported_sql_negates_the_predicate_with_the_same_filters():
    sel, sel_params = build_selection_sql("central-bank", "c", **_ALL_FILTERS)
    uns, uns_params = build_unsupported_sql("central-bank", "c", **_ALL_FILTERS)
    assert _SUPPORTED_PREDICATE in sel
    assert f"NOT {_SUPPORTED_PREDICATE}" in uns
    sel_where, uns_where = _where_clause(sel), _where_clause(uns)
    # identical WHERE except for the negation: an exact partition of candidates
    assert uns_where == sel_where.replace(_SUPPORTED_PREDICATE,
                                          f"NOT {_SUPPORTED_PREDICATE}")
    assert sel_where.count(_SUPPORTED_PREDICATE) == 1
    for clause in ("d.source_code = ANY(%(source_codes)s)", "d.doc_type = ANY(%(doctypes)s)",
                   "d.year >= %(year_min)s", "d.year <= %(year_max)s",
                   "d.language = ANY(%(languages)s)", "r.doc_id IS NULL",
                   "d.deleted_at IS NULL", "d.corpus = %(corpus)s"):
        assert clause in sel_where
    assert "GROUP BY d.source_code" in uns
    assert uns_params == sel_params


def test_where_takes_typed_keywords_only():
    clauses, params = _where("central-bank", "c", source_codes=["ecb"], year_min=2015,
                             supported_mimes=("text/markdown",))
    assert "d.source_code = ANY(%(source_codes)s)" in clauses
    assert params["supported_ext_patterns"] == ["%.md"]
    with pytest.raises(TypeError):
        _where("central-bank", "c", doc_types=["A1"])  # typo is an error
    with pytest.raises(TypeError):
        build_selection_sql("central-bank", "c", doc_types=["A1"])


@pytest.mark.parametrize("builder", [build_selection_sql, build_unsupported_sql])
@pytest.mark.parametrize("mimes,match", [
    ((), "supported_mimes lists no mime type"),
    (("application/pdf", "text/html"), "supported_mimes: the engine cannot read text/html"),
    (("Application/PDF",), "the engine cannot read Application/PDF"),
    ((" text/plain",), "the engine cannot read  text/plain"),
])
def test_explicit_supported_mimes_are_validated(builder, mimes, match):
    with pytest.raises(ValueError, match=match):
        builder("central-bank", "c", supported_mimes=mimes)


def test_count_unsupported_validates_explicit_mimes_before_querying():
    conn = DescribedFakeConn(rows=[])
    with pytest.raises(ValueError, match="text/html"):
        count_unsupported(conn, "central-bank", "c", supported_mimes=("text/html",))
    assert conn.executed == []


def test_count_unsupported_folds_null_source_code_into_unknown():
    conn = DescribedFakeConn(rows=[("ecb", 3), (None, 2), ("us", 1)])
    assert count_unsupported(conn, "central-bank", "c",
                             supported_mimes=("application/pdf",)) == {
        "ecb": 3, "unknown": 2, "us": 1}
