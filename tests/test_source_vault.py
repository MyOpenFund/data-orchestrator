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
        "document_url": "https://example.org/d1.pdf", "source_url": "https://example.org/d1",
        "mime_type": "application/pdf",
        "date_precision": None, "date_kind": None, "entity_scheme": None, "entity_value": None,
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


@pytest.mark.parametrize("tail", [" ", "\t", "\r\n", " \t\r\n "])
def test_row_to_item_strips_trailing_whitespace_from_local_path(monkeypatch, tmp_path, tail):
    """The SQL compares on rtrim(local_path, ' \\t\\r\\n'): a row it selected as
    '.pdf' must resolve to the '.pdf' file, not to a name ending in blanks."""
    monkeypatch.setenv("CB_CORPUS_ROOT", str(tmp_path))
    item = row_to_item(_row(local_path="data/raw/us/C1/2015/d1.pdf" + tail), "central-bank")
    assert item.path == tmp_path / "raw" / "us" / "C1" / "2015" / "d1.pdf"


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


def test_count_unsupported_folds_null_source_code_into_the_corpus():
    """Same label core._bump gives a NULL-source vault item (core.source_label)."""
    conn = DescribedFakeConn(rows=[("ecb", 3), (None, 2), ("us", 1), ("", 1)])
    assert count_unsupported(conn, "central-bank", "c",
                             supported_mimes=("application/pdf",)) == {
        "ecb": 3, "central-bank": 3, "us": 1}


# ---------------------------------------------------------------------------
# Reserved payload keys: ``extra`` is free-form; a key the engine writes itself
# is left out and reported on the item (the engine would refuse the document).
# ---------------------------------------------------------------------------
def test_row_to_item_drops_the_engine_s_keys_from_extra_and_reports_them(monkeypatch, tmp_path):
    monkeypatch.setenv("CB_CORPUS_ROOT", str(tmp_path))
    extra = {"filename": "old.pdf", "text": "stale", "page": 4, "chunk_number": 2,
             "ingestion_date": "2020-01-01", "cik": "0001"}
    item = row_to_item(_row(extra=extra), "central-bank")
    assert sorted(item.dropped_keys) == ["chunk_number", "filename", "ingestion_date",
                                         "page", "text"]
    assert not set(item.dropped_keys) & set(item.payload)
    assert item.payload["cik"] == "0001"  # the rest of the extra is kept
    assert "filename" in extra            # the row is not changed


def test_row_to_item_reports_nothing_for_a_clean_extra(monkeypatch, tmp_path):
    monkeypatch.setenv("CB_CORPUS_ROOT", str(tmp_path))
    assert row_to_item(_row(), "central-bank").dropped_keys == ()
    assert row_to_item(_row(extra=None), "central-bank").dropped_keys == ()


def test_row_to_item_never_drops_a_vault_column(monkeypatch, tmp_path):
    """The columns are the vault's own: all of them reach the payload, and the
    ``doc_id`` is always the row's (an ``extra`` that repeats a column loses,
    as before — that is the column rule, not a dropped key)."""
    from data_orchestrator.sources.vault import _PAYLOAD_COLUMNS

    monkeypatch.setenv("CB_CORPUS_ROOT", str(tmp_path))
    item = row_to_item(_row(extra={"doc_id": "someone-else", "year": 1900, "text": "x"},
                            date_precision="day", date_kind="meeting", entity_scheme="cik",
                            entity_value="0000320193"),
                       "central-bank")
    assert set(_PAYLOAD_COLUMNS) <= set(item.payload)
    assert item.payload["doc_id"] == item.doc_id == "d1" and item.payload["year"] == 2015
    assert item.dropped_keys == ("text",)


def test_no_vault_column_is_a_key_the_engine_refuses():
    from rag_engine.schema import RESERVED_KEYS

    from data_orchestrator.sources.vault import _PAYLOAD_COLUMNS

    assert set(_PAYLOAD_COLUMNS) & set(RESERVED_KEYS) == {"doc_id"}


def test_a_vault_item_with_a_dirty_extra_is_ingested_not_failed(monkeypatch, tmp_path):
    """End to end through the engine: the stray keys cannot fail the document,
    the engine's own values are written, and the run counts what was dropped."""
    from data_orchestrator.core import run_ingest
    from tests.test_core_engine import FakeEmbedder, FakeStore, _md_item

    monkeypatch.setenv("CB_CORPUS_ROOT", str(tmp_path))
    (tmp_path / "raw").mkdir()
    path = _md_item(tmp_path / "raw", doc_id="d1").path
    item = row_to_item(_row(local_path="data/raw/d1.md",
                            extra={"filename": "old.pdf", "page": 4, "cik": "0001"}),
                       "central-bank")
    assert item.path == path
    store = FakeStore([])
    stats = run_ingest([item], collection="c", store=store, embedder=FakeEmbedder())
    assert (stats.docs_ingested, stats.docs_error) == (1, 0)
    assert stats.payload_keys_dropped == {"filename": 1, "page": 1}
    payload = store.client.points[0].payload
    assert payload["filename"] == "d1.md" and "page" not in payload and payload["cik"] == "0001"


def test_a_vault_extra_holding_the_derived_interval_is_ingested_not_failed(monkeypatch, tmp_path):
    """The vault's ``extra`` must never make ingestion fail: a stale
    ``date_start`` / ``date_end`` there is left out and counted, and the engine
    writes the interval it derives from ``date`` and ``date_precision``, which
    pass through from ``extra`` with ``date_kind``."""
    import datetime

    from data_orchestrator.core import run_ingest
    from tests.test_core_engine import FakeEmbedder, FakeStore, _md_item

    monkeypatch.setenv("CB_CORPUS_ROOT", str(tmp_path))
    (tmp_path / "raw").mkdir()
    _md_item(tmp_path / "raw", doc_id="d1")
    item = row_to_item(_row(local_path="data/raw/d1.md", date=datetime.date(2016, 7, 1),
                            year=2016,
                            extra={"date_precision": "month", "date_kind": "meeting",
                                   "date_source": "title",
                                   "date_start": "1999-01-01T00:00:00Z", "date_end": 5}),
                       "central-bank")
    assert item.dropped_keys == ("date_start", "date_end")
    store = FakeStore([])
    stats = run_ingest([item], collection="c", store=store, embedder=FakeEmbedder())
    assert (stats.docs_ingested, stats.docs_error) == (1, 0)
    assert stats.payload_keys_dropped == {"date_start": 1, "date_end": 1}
    payload = store.client.points[0].payload
    assert payload["date"] == "2016-07-01" and payload["year"] == 2016
    assert payload["date_precision"] == "month" and payload["date_kind"] == "meeting"
    assert payload["date_source"] == "title"
    assert payload["date_start"] == "2016-07-01T00:00:00Z"
    assert payload["date_end"] == "2016-07-31T00:00:00Z"
    assert list(payload)[-2:] == ["date_start", "date_end"]


def test_an_undated_vault_row_gets_no_interval(monkeypatch, tmp_path):
    from data_orchestrator.core import run_ingest
    from tests.test_core_engine import FakeEmbedder, FakeStore, _md_item

    monkeypatch.setenv("CB_CORPUS_ROOT", str(tmp_path))
    (tmp_path / "raw").mkdir()
    _md_item(tmp_path / "raw", doc_id="d1")
    item = row_to_item(_row(local_path="data/raw/d1.md", date=None, year=None,
                            extra={"date_precision": "garbage"}), "central-bank")
    store = FakeStore([])
    run_ingest([item], collection="c", store=store, embedder=FakeEmbedder())
    payload = store.client.points[0].payload
    assert "date" not in payload and "date_start" not in payload and "date_end" not in payload
    assert payload["date_precision"] == "garbage"  # copied as given; the engine reads it as a day


# ---------------------------------------------------------------------------
# Links (F7): document_url, source_url and mime_type are selected and reach the
# payload; the keys of ``extra`` dropped by design never do.
# ---------------------------------------------------------------------------
def test_the_selection_reads_the_link_columns():
    sql, _ = build_selection_sql("central-bank", "c")
    for col in ("document_url", "source_url", "mime_type"):
        assert col in _SELECT_COLUMNS
        assert f"d.{col}" in sql
    assert "pdf_url" not in sql


def test_the_link_columns_reach_the_payload(monkeypatch, tmp_path):
    monkeypatch.setenv("CB_CORPUS_ROOT", str(tmp_path))
    item = row_to_item(_row(document_url="https://x.org/a.pdf", source_url="https://x.org/a",
                            mime_type="application/pdf"), "central-bank")
    assert item.payload["document_url"] == "https://x.org/a.pdf"
    assert "pdf_url" not in item.payload
    assert item.payload["source_url"] == "https://x.org/a"
    assert item.payload["mime_type"] == "application/pdf"


def test_null_link_columns_are_absent_and_remove_the_same_key_of_extra(monkeypatch, tmp_path):
    """NULL column -> key absent, even when ``extra`` holds that column's name
    (the column wins, as for every other column)."""
    monkeypatch.setenv("CB_CORPUS_ROOT", str(tmp_path))
    item = row_to_item(_row(document_url=None, source_url=None, mime_type=None,
                            extra={"document_url": "https://stale.org/x.pdf",
                                   "pdf_url": "https://stale.org/x.pdf", "source_url": "s"}),
                       "central-bank")
    for col in ("document_url", "pdf_url", "source_url", "mime_type"):
        assert col not in item.payload
    assert item.dropped_keys == ()


def test_a_link_column_wins_over_extra(monkeypatch, tmp_path):
    monkeypatch.setenv("CB_CORPUS_ROOT", str(tmp_path))
    item = row_to_item(_row(document_url="https://col.org/a.pdf",
                            extra={"document_url": "https://extra.org/a.pdf"}), "central-bank")
    assert item.payload["document_url"] == "https://col.org/a.pdf"


def test_a_pdf_url_of_extra_never_reaches_the_payload(monkeypatch, tmp_path):
    """``pdf_url`` is the former name of the ``document_url`` column: the column
    wins, so the old name is left out of the payload (silently, not counted),
    whatever ``extra`` holds."""
    monkeypatch.setenv("CB_CORPUS_ROOT", str(tmp_path))
    item = row_to_item(_row(document_url="https://col.org/a.pdf",
                            extra={"pdf_url": "https://extra.org/a.pdf", "cik": "1"}),
                       "central-bank")
    assert "pdf_url" not in item.payload and item.dropped_keys == ()
    assert item.payload["document_url"] == "https://col.org/a.pdf"
    assert item.payload["cik"] == "1"


# ---------------------------------------------------------------------------
# Transition (2026-11 rename): the selection reads whichever of document_url
# and pdf_url the vault has, as document_url.
# ---------------------------------------------------------------------------
def test_a_vault_before_the_rename_is_selected_through_pdf_url():
    sql, _ = build_selection_sql("central-bank", "c", document_url_from="pdf_url")
    assert "d.pdf_url AS document_url" in sql and "d.document_url" not in sql


def test_build_selection_sql_refuses_another_address_column():
    with pytest.raises(ValueError, match="address"):
        build_selection_sql("central-bank", "c", document_url_from="html_path")


@pytest.mark.parametrize("schema_columns, selected", [
    (("document_url",), "d.document_url,"),
    (("pdf_url",), "d.pdf_url AS document_url"),
    (("pdf_url", "document_url"), "d.document_url,"),
    ((), "d.document_url,"),  # no documents table found: Postgres names the missing column
])
def test_iter_items_reads_the_column_of_the_vault_first(monkeypatch, tmp_path, schema_columns,
                                                        selected):
    monkeypatch.setenv("CB_CORPUS_ROOT", str(tmp_path))
    conn = DescribedFakeConn(rows=[_tuple_row()], columns=_SELECT_COLUMNS,
                             schema_columns=schema_columns)
    (item,) = iter_items(conn, "central-bank", "c")
    assert "information_schema.columns" in conn.executed[0][0]
    assert selected in conn.executed[1][0]
    assert item.payload["document_url"] == "https://example.org/d1.pdf"


def test_the_column_is_read_once_per_connection(monkeypatch, tmp_path):
    monkeypatch.setenv("CB_CORPUS_ROOT", str(tmp_path))
    conn = DescribedFakeConn(rows=[_tuple_row()], columns=_SELECT_COLUMNS,
                             schema_columns=("pdf_url",))
    list(iter_items(conn, "central-bank", "c"))
    list(iter_items(conn, "central-bank", "c"))
    reads = [sql for sql, _ in conn.executed if "information_schema" in sql]
    assert len(reads) == 1
    assert "d.pdf_url AS document_url" in conn.executed[-1][0]


def test_the_keys_dropped_by_design_never_reach_the_payload_and_are_not_counted(
        monkeypatch, tmp_path):
    """``html_path`` (a producer-machine path), ``alt_urls``, ``alt_source_urls``,
    ``local_path`` are left out silently; a reserved key next to them is still
    left out AND counted; the useful keys of ``extra`` stay."""
    monkeypatch.setenv("CB_CORPUS_ROOT", str(tmp_path))
    extra = {"html_path": "/Users/someone/cb/x.html", "alt_urls": ["https://a", "https://b"],
             "alt_source_urls": [], "local_path": "data/raw/x.pdf", "text": "stale",
             "date_precision": "month", "date_kind": "meeting", "date_source": "title",
             "repec_handle": "RePEc:x:y:z"}
    item = row_to_item(_row(extra=extra), "central-bank")
    for key in ("html_path", "alt_urls", "alt_source_urls", "local_path", "text"):
        assert key not in item.payload
    assert item.dropped_keys == ("text",)
    assert {k: item.payload[k] for k in ("date_precision", "date_kind", "date_source",
                                         "repec_handle")} == {
        "date_precision": "month", "date_kind": "meeting", "date_source": "title",
        "repec_handle": "RePEc:x:y:z"}


def test_the_dropped_keys_follow_the_setting(monkeypatch, tmp_path):
    monkeypatch.setenv("CB_CORPUS_ROOT", str(tmp_path))
    extra = {"html_path": "/x.html", "repec_handle": "h"}
    monkeypatch.setenv("DATA_ORCHESTRATOR_PAYLOAD_DROP_EXTRA_KEYS", "repec_handle")
    item = row_to_item(_row(extra=extra), "central-bank")
    assert item.payload["html_path"] == "/x.html" and "repec_handle" not in item.payload
    monkeypatch.setenv("DATA_ORCHESTRATOR_PAYLOAD_DROP_EXTRA_KEYS", "none")
    item = row_to_item(_row(extra=extra), "central-bank")
    assert item.payload["html_path"] == "/x.html" and item.payload["repec_handle"] == "h"


def test_the_anti_join_matches_the_corpus_and_the_doc_id():
    """A document is (corpus, doc_id) in the vault: a row of another corpus
    with the same doc_id in rag_ingestions never hides this one."""
    for sql, _params in (build_selection_sql("company", "company-e5b-v1"),
                         build_unsupported_sql("company", "company-e5b-v1")):
        flat = " ".join(sql.split())
        assert ("ON r.corpus = d.corpus AND r.doc_id = d.doc_id "
                "AND r.collection = %(collection)s") in flat
