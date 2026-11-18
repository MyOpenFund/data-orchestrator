"""The payload rule shared by the vault and the cb_corpus disk source
(``data_orchestrator.payload``)."""
import pytest

from data_orchestrator import payload as pl

ENV = "DATA_ORCHESTRATOR_PAYLOAD_DROP_EXTRA_KEYS"


# ---------------------------------------------------------------------------
# The setting
# ---------------------------------------------------------------------------
def test_the_default_drops_the_producer_paths_and_the_alternative_urls():
    assert pl.dropped_extra_keys() == ("html_path", "alt_urls", "alt_source_urls", "local_path")


@pytest.mark.parametrize("raw,expected", [
    ("", pl.DEFAULT_DROP_EXTRA_KEYS),
    ("   ", pl.DEFAULT_DROP_EXTRA_KEYS),
    ("none", ()),
    (" NONE ", ()),
    ("html_path", ("html_path",)),
    (" html_path , foo,html_path,, ", ("html_path", "foo")),
])
def test_the_setting_is_read_from_the_environment(monkeypatch, raw, expected):
    monkeypatch.setenv(ENV, raw)
    assert pl.dropped_extra_keys() == expected


@pytest.mark.parametrize("raw,match", [
    (",", "lists no key"),
    (" , ,", "lists no key"),
    ("html_path,title", "title cannot be dropped"),
    ("pdf_url", "pdf_url cannot be dropped"),
    ("text,page", "text, page cannot be dropped"),
    ("date_start", "date_start cannot be dropped"),
])
def test_the_setting_refuses_what_it_cannot_drop(monkeypatch, raw, match):
    monkeypatch.setenv(ENV, raw)
    with pytest.raises(ValueError, match=match):
        pl.dropped_extra_keys()


# ---------------------------------------------------------------------------
# A manifest row, split like the vault splits it
# ---------------------------------------------------------------------------
def test_a_manifest_row_is_split_into_the_vault_s_columns_and_extra():
    rec = {"doc_id": "a", "bank_code": "us", "doc_type": "C1", "title": "T",
           "pdf_url": "https://x/a.pdf", "date": "2015-06-30", "year": "2015",
           "local_path": "data/raw/us/C1/2015/a.pdf", "date_precision": "day",
           "html_path": "/Users/x/a.html", "unknown_key": {"nested": [1, 2]}}
    row = pl.manifest_row_to_vault_row(rec, "central-bank")
    assert row["corpus"] == "central-bank" and row["source_code"] == "us"
    assert row["year"] == 2015 and row["date"] == "2015-06-30"
    assert row["local_path"] == "data/raw/us/C1/2015/a.pdf"
    assert row["source_url"] is None and row["mime_type"] is None
    assert row["extra"] == {"date_precision": "day", "html_path": "/Users/x/a.html",
                            "unknown_key": {"nested": [1, 2]}}


def test_source_code_wins_over_the_legacy_bank_code_and_an_empty_one_does_not():
    assert pl.manifest_row_to_vault_row({"source_code": "ecb", "bank_code": "fr"},
                                        "c")["source_code"] == "ecb"
    assert pl.manifest_row_to_vault_row({"source_code": "", "bank_code": "fr"},
                                        "c")["source_code"] == "fr"


@pytest.mark.parametrize("raw,expected", [
    ("", None), (None, None), ("2015-06-30", "2015-06-30"),
    ("2015-06-30T10:00:00Z", "2015-06-30"), ("2015-06", "2015-06"),
])
def test_the_date_is_what_the_vault_s_date_column_returns(raw, expected):
    assert pl.manifest_row_to_vault_row({"date": raw}, "c")["date"] == expected


def test_the_row_s_own_corpus_is_kept():
    assert pl.manifest_row_to_vault_row({"corpus": "other"}, "central-bank")["corpus"] == "other"


def test_a_manifest_payload_applies_the_vault_rules():
    """Columns win over extra; NULL -> absent; dropped-by-design keys silent;
    reserved keys left out and reported; unknown keys kept."""
    rec = {"doc_id": "a", "source_code": "us", "title": None, "date": "2015-06-30",
           "html_path": "/x.html", "alt_urls": ["u"], "page": 3, "series": "S"}
    payload, dropped = pl.manifest_row_to_payload(rec, "central-bank")
    assert payload == {"series": "S", "doc_id": "a", "corpus": "central-bank",
                       "source_code": "us", "date": "2015-06-30"}
    assert dropped == ("page",)


# ---------------------------------------------------------------------------
# Blank links and null keys of extra
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("blank", ["", "   ", "\t\n"])
def test_a_blank_link_is_absent_like_a_null_one(blank):
    """The vault stores ``source_url`` as the manifest gives it, and committed
    central-bank manifests give ``""``: an empty or whitespace-only
    ``pdf_url``, ``source_url`` or ``mime_type`` is no link, so it is left
    out (and removes the same key of ``extra``, as a NULL column does)."""
    row = {"doc_id": "a", "corpus": "central-bank", "pdf_url": blank, "source_url": blank,
           "mime_type": blank, "extra": {"source_url": "https://stale"}}
    payload, dropped = pl.row_to_payload(row)
    assert payload == {"doc_id": "a", "corpus": "central-bank"} and dropped == ()


def test_a_manifest_row_with_an_empty_source_url_has_no_source_url():
    rec = {"doc_id": "a", "source_code": "us", "pdf_url": "https://x/a.pdf", "source_url": ""}
    payload, _ = pl.manifest_row_to_payload(rec, "central-bank")
    assert payload["pdf_url"] == "https://x/a.pdf" and "source_url" not in payload


def test_a_real_link_is_kept_as_given():
    payload, _ = pl.row_to_payload({"doc_id": "a", "source_url": " https://x/a "})
    assert payload["source_url"] == " https://x/a "


def test_a_null_key_of_extra_is_left_out_and_not_counted():
    """README: an absent or null field is left out. A null reserved key is
    left out like any null, not counted as a reserved key refused."""
    row = {"doc_id": "a", "extra": {"date_kind": None, "series": "S", "page": None}}
    payload, dropped = pl.row_to_payload(row)
    assert payload == {"series": "S", "doc_id": "a"} and dropped == ()


# ---------------------------------------------------------------------------
# Contract with the vault: its KNOWN_FIELDS, read from a checkout
# ---------------------------------------------------------------------------
VAULT_CHECKOUT_ENV = "DATA_ORCHESTRATOR_VAULT_CHECKOUT"


def vault_known_fields(checkout):
    """``KNOWN_FIELDS`` of the vault's ``ingestion/ingest.py``, read from its
    source (parsed, not imported: the vault's dependencies are not this
    package's). None when the module assigns no literal ``KNOWN_FIELDS``."""
    import ast
    from pathlib import Path

    tree = ast.parse((Path(checkout) / "ingestion" / "ingest.py").read_text(encoding="utf-8"))
    for node in tree.body:
        targets = (node.targets if isinstance(node, ast.Assign)
                   else [node.target] if isinstance(node, ast.AnnAssign) and node.value else [])
        if any(isinstance(t, ast.Name) and t.id == "KNOWN_FIELDS" for t in targets):
            return frozenset(ast.literal_eval(node.value))
    return None


def test_the_known_fields_reader_reads_a_literal_set(tmp_path):
    (tmp_path / "ingestion").mkdir()
    (tmp_path / "ingestion" / "ingest.py").write_text(
        'import os\nKNOWN_FIELDS = {\n    "doc_id", "title",\n}\nOTHER = 1\n', encoding="utf-8")
    assert vault_known_fields(tmp_path) == {"doc_id", "title"}
    (tmp_path / "ingestion" / "ingest.py").write_text("X = 1\n", encoding="utf-8")
    assert vault_known_fields(tmp_path) is None


def test_the_vault_split_copies_the_vault_s_known_fields():
    """The split of a manifest row is the vault's when ``VAULT_KNOWN_FIELDS`` is
    its ``KNOWN_FIELDS``: checked against a vault checkout named by
    ``DATA_ORCHESTRATOR_VAULT_CHECKOUT`` (skipped without one)."""
    import os

    checkout = os.environ.get(VAULT_CHECKOUT_ENV)
    if not checkout:
        pytest.skip(f"{VAULT_CHECKOUT_ENV} names no vault checkout")
    known = vault_known_fields(checkout)
    assert known is not None, f"no literal KNOWN_FIELDS in {checkout}/ingestion/ingest.py"
    assert pl.VAULT_KNOWN_FIELDS == known


def test_the_vault_row_has_a_column_for_every_known_field_but_the_legacy_one():
    row = pl.manifest_row_to_vault_row({}, "c")
    assert set(row) == (pl.VAULT_KNOWN_FIELDS - {"bank_code"}) | {"extra"}
    assert set(pl.PAYLOAD_COLUMNS) <= pl.VAULT_KNOWN_FIELDS
