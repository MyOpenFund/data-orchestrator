"""The payload rule shared by the vault and the cb_corpus disk source
(``data_orchestrator.payload``)."""
import json

import pytest
from vault_contract import golden

from data_orchestrator import payload as pl

ENV = "DATA_ORCHESTRATOR_PAYLOAD_DROP_EXTRA_KEYS"


# ---------------------------------------------------------------------------
# The setting
# ---------------------------------------------------------------------------
def test_the_default_drops_the_producer_paths_and_the_alternative_urls():
    assert pl.dropped_extra_keys() == ("html_path", "alt_urls", "alt_source_urls", "local_path",
                                       "pdf_path", "text_path", "primary_path",
                                       "submission_path")


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
    ("document_url", "document_url cannot be dropped"),
    ("pdf_url", "pdf_url cannot be dropped"),  # the former name: already left out
    ("text,page", "text, page cannot be dropped"),
    ("date_start", "date_start cannot be dropped"),
])
def test_the_setting_refuses_what_it_cannot_drop(monkeypatch, raw, match):
    monkeypatch.setenv(ENV, raw)
    with pytest.raises(ValueError, match=match):
        pl.dropped_extra_keys()


# ---------------------------------------------------------------------------
# A manifest row, split like the vault splits it (vault_contract.split_row)
# ---------------------------------------------------------------------------
def test_a_manifest_row_is_split_into_the_vault_s_columns_and_extra():
    rec = {"doc_id": "a", "bank_code": "us", "doc_type": "C1", "title": "T",
           "document_url": "https://x/a.pdf", "date": "2015-06-30", "year": 2015,
           "local_path": "data/raw/us/C1/2015/a.pdf", "date_precision": "day",
           "html_path": "/Users/x/a.html", "unknown_key": {"nested": [1, 2]}}
    row = pl.manifest_row_to_vault_row(rec, "central-bank")
    assert row["corpus"] == "central-bank" and row["source_code"] == "us"
    assert row["year"] == 2015 and row["date"] == "2015-06-30"
    assert row["date_precision"] == "day" and row["date_kind"] is None
    assert row["local_path"] == "raw/us/C1/2015/a.pdf"  # transition: no data/ prefix
    assert row["document_url"] == "https://x/a.pdf"
    assert row["source_url"] is None and row["mime_type"] is None
    assert row["entity_scheme"] is None and row["entity_value"] is None
    assert row["extra"] == {"html_path": "/Users/x/a.html", "unknown_key": {"nested": [1, 2]}}


def test_the_vault_row_has_every_contract_column_and_extra():
    from vault_contract import COLUMNS

    row = pl.manifest_row_to_vault_row({}, "c")
    assert list(row) == [*COLUMNS, "extra"]
    assert row["corpus"] == "c" and row["extra"] is None


@pytest.mark.parametrize("rec,expected", [
    ({"document_url": "https://new"}, "https://new"),
    ({"pdf_url": "https://old"}, "https://old"),  # a manifest row not migrated yet
    ({"document_url": "https://new", "pdf_url": "https://old"}, "https://new"),
    ({"document_url": "", "pdf_url": "https://old"}, "https://old"),
    ({"document_url": None, "pdf_url": "https://old"}, "https://old"),
    ({}, None),
])
def test_document_url_wins_over_the_legacy_pdf_url_and_an_empty_one_does_not(rec, expected):
    row = pl.manifest_row_to_vault_row(rec, "c")
    assert row["document_url"] == expected
    assert "pdf_url" not in row and row["extra"] is None  # the old name is no extra field


def test_a_legacy_manifest_row_gives_the_payload_document_url():
    payload, _ = pl.manifest_row_to_payload({"doc_id": "a", "pdf_url": "https://x/a.pdf"},
                                            "central-bank")
    assert payload["document_url"] == "https://x/a.pdf" and "pdf_url" not in payload


@pytest.mark.parametrize("old", ["pdf_url", "bank_code"])
def test_a_former_column_name_in_extra_is_left_out_of_the_payload_and_not_counted(old):
    """``pdf_url`` and ``bank_code`` name the ``document_url`` and
    ``source_code`` columns (their former names): the column always wins, so
    the payload never carries the old name."""
    row = {"doc_id": "a", "document_url": None, "extra": {old: "stale", "s": 1}}
    assert pl.row_to_payload(row) == ({"s": 1, "doc_id": "a"}, ())


def test_the_former_names_are_the_contract_s_transition_aliases():
    from vault_contract import TRANSITION_ALIASES

    assert pl.LEGACY_COLUMN_NAMES == TRANSITION_ALIASES
    assert set(pl.LEGACY_COLUMN_NAMES.values()) <= set(pl.PAYLOAD_COLUMNS)


def test_source_code_wins_over_the_legacy_bank_code_and_an_empty_one_does_not():
    assert pl.manifest_row_to_vault_row({"source_code": "ecb", "bank_code": "fr"},
                                        "c")["source_code"] == "ecb"
    assert pl.manifest_row_to_vault_row({"source_code": "", "bank_code": "fr"},
                                        "c")["source_code"] == "fr"


@pytest.mark.parametrize("raw,expected", [("", None), (None, None), ("2015-06-30", "2015-06-30")])
def test_an_empty_date_is_null(raw, expected):
    assert pl.manifest_row_to_vault_row({"date": raw}, "c")["date"] == expected


def test_an_undated_row_loses_its_precision_and_kind_as_in_the_vault():
    row = pl.manifest_row_to_vault_row({"date": "", "date_precision": "month",
                                        "date_kind": "meeting"}, "c")
    assert row["date_precision"] is None and row["date_kind"] is None and row["extra"] is None


@pytest.mark.parametrize("corpus", [None, "", "  "])
def test_an_absent_or_blank_corpus_is_the_default_one(corpus):
    assert pl.manifest_row_to_vault_row({"corpus": corpus}, "central-bank")["corpus"] \
        == "central-bank"


def test_the_row_s_own_corpus_is_kept():
    assert pl.manifest_row_to_vault_row({"corpus": "other"}, "central-bank")["corpus"] == "other"


def test_the_split_does_not_mutate_the_manifest_row():
    rec = {"doc_id": "a", "bank_code": "us", "local_path": "data/raw/a.pdf", "x": [1]}
    before = json.loads(json.dumps(rec))
    pl.manifest_row_to_vault_row(rec, "c")
    assert rec == before


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
    ``document_url``, ``source_url`` or ``mime_type`` is no link, so it is left
    out (and removes the same key of ``extra``, as a NULL column does)."""
    row = {"doc_id": "a", "corpus": "central-bank", "document_url": blank, "source_url": blank,
           "mime_type": blank, "extra": {"source_url": "https://stale"}}
    payload, dropped = pl.row_to_payload(row)
    assert payload == {"doc_id": "a", "corpus": "central-bank"} and dropped == ()


def test_a_manifest_row_with_an_empty_source_url_has_no_source_url():
    rec = {"doc_id": "a", "source_code": "us", "document_url": "https://x/a.pdf",
           "source_url": ""}
    payload, _ = pl.manifest_row_to_payload(rec, "central-bank")
    assert payload["document_url"] == "https://x/a.pdf" and "source_url" not in payload


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
# The contract's columns in the payload
# ---------------------------------------------------------------------------
def test_the_contract_columns_come_last_and_leave_every_earlier_position_unchanged():
    assert pl.PAYLOAD_COLUMNS[:13] == (
        "doc_id", "corpus", "source_code", "doc_type", "title", "date",
        "year", "language", "sha256", "provenance",
        "document_url", "source_url", "mime_type")
    assert pl.PAYLOAD_COLUMNS[13:] == ("date_precision", "date_kind", "entity_scheme",
                                       "entity_value")
    assert pl.CONTRACT_COLUMNS_FROM_EXTRA == pl.PAYLOAD_COLUMNS[13:]


def test_every_payload_column_is_a_contract_column_and_local_path_is_not_carried():
    from vault_contract import COLUMNS

    assert set(pl.PAYLOAD_COLUMNS) == set(COLUMNS) - {"local_path"}


def test_a_contract_column_wins_over_the_same_key_of_extra():
    row = {"doc_id": "a", "date_precision": "day", "entity_value": "0000320193",
           "extra": {"date_precision": "month", "entity_value": "320193"}}
    payload, _ = pl.row_to_payload(row)
    assert payload["date_precision"] == "day" and payload["entity_value"] == "0000320193"


def test_a_null_contract_column_keeps_the_value_of_extra_of_a_row_stored_before_the_contract():
    """A vault row stored before the contract has the four new columns NULL
    and ``date_precision`` / ``date_kind`` in ``extra``: the payload keeps them
    (any other NULL column removes the same key of ``extra``)."""
    row = {"doc_id": "a", "date_precision": None, "date_kind": None, "title": None,
           "extra": {"date_precision": "month", "date_kind": "meeting", "title": "stale"}}
    payload, _ = pl.row_to_payload(row)
    assert payload == {"date_precision": "month", "date_kind": "meeting", "doc_id": "a"}


def test_a_company_row_carries_its_entity():
    case = golden.accepted()["sec_10k"]
    payload, dropped = pl.manifest_row_to_payload(case["row"], "company")
    assert dropped == ()
    assert payload["corpus"] == "company" and payload["doc_type"] == "A1"
    assert payload["entity_scheme"] == "cik" and payload["entity_value"] == "0000320193"
    assert payload["date_precision"] == "day" and payload["date_kind"] == "publication"
    assert payload["cik"] == "0000320193" and payload["accession"] == "0000320193-24-000123"
    assert "local_path" not in payload and "alt_urls" not in payload
    assert not {"pdf_path", "text_path", "primary_path", "submission_path"} & set(payload)


@pytest.mark.parametrize("name", sorted(golden.accepted()))
def test_the_split_of_every_golden_row_is_the_contract_s(name):
    """The contract's reference rows (shipped in the package, the cases the
    vault tests its ingestion with): the vault row built here holds exactly
    their expected columns."""
    case = golden.accepted()[name]
    row = pl.manifest_row_to_vault_row(case["row"], case["expected_corpus"])
    expected = dict(case["expected_columns"])
    if expected["corpus"] is None:  # transition: the ingesting service's CORPUS applies
        expected["corpus"] = case["expected_corpus"]
    assert {k: v for k, v in row.items() if k != "extra"} == expected


# ---------------------------------------------------------------------------
# Central-bank points ingested before the contract keep their payload
# ---------------------------------------------------------------------------
# Four real rows of the public central-bank manifests (repo central-bank-corpus,
# data/manifest, commit ea679d4), as the producer wrote them before the
# contract: a precision and a kind, a month precision, no precision, and one of
# the 7 undated rows that carry a precision. OLD_PAYLOADS is what this repo's
# payload rule gave them before the contract (main 81ead10, from the manifest
# and from the vault row alike), written out here.
CB_ROWS = [
    {"bank_code": "au", "doc_type": "A1", "title": "RBA monetary policy decision 1997-07-30",
     "document_url": "https://www.rba.gov.au/media-releases/1997/mr-97-15.html",
     "source_url": "https://www.rba.gov.au/monetary-policy/int-rate-decisions/1997/",
     "date": "1997-07-30", "language": "en", "provenance": "bank_site",
     "mime_type": "application/pdf",
     "sha256": "d72df39e4bfe9b9ff92c69f26f16c6bfa7a66311a17eb9c5dce5e8016a1008b6",
     "local_path": "data/raw/au/A1/1997/ac6330092872d2f4.pdf",
     "html_path": "data/raw/au/A1/1997/ac6330092872d2f4.html", "doc_id": "ac6330092872d2f4",
     "year": 1997, "date_precision": "day", "date_source": "bank_site",
     "date_kind": "publication"},
    {"bank_code": "au", "doc_type": "D1",
     "title": "Tracking Mergers and Acquisitions Using Australian Administrative Data",
     "document_url": "https://www.rba.gov.au/publications/rdp/2026/pdf/rdp2026-04.pdf",
     "source_url": "https://ideas.repec.org/p/rba/rbardp/rdp2026-04.html", "date": "2026-07-01",
     "language": "en", "provenance": "repec_discovery", "mime_type": "application/pdf",
     "sha256": "ad7c93f07f733488e34d2fab98cb170318ccaa5b6ca3bafcfb0d0aba5124ca6a",
     "local_path": "data/raw/au/D1/2026/ae7f793211656688.pdf", "html_path": None,
     "date_precision": "month", "date_source": "repec", "repec_handle": "",
     "alt_urls": ["https://libkey.io/10.47688/rdp2026-04?utm_source=ideas"],
     "doc_id": "ae7f793211656688", "year": 2026},
    {"bank_code": "ae", "doc_type": "C1",
     "title": "Sultan Bin Nasser Al-Suwaidi: The global financial crisis and issues for a "
              "regulatory response",
     "document_url": "https://www.bis.org/review/r100205e.pdf",
     "source_url": "https://www.bis.org/review/r100205e.htm", "date": "2010-02-05",
     "language": "en", "provenance": "bis_index", "mime_type": "application/pdf",
     "sha256": "4f48a3f641bea8a028091980ca64632102703dd453d33438d0a28fa8beeece94",
     "local_path": "data/raw/ae/C1/2010/a5b8ba5e477ad19a.pdf", "doc_id": "a5b8ba5e477ad19a",
     "year": 2010},
    {"bank_code": "ca", "doc_type": "D1", "title": "Exchange Rate Volatility and Trade: A Survey",
     "document_url": "https://doi.org/10.34989/swp-1994-5",
     "source_url": "https://ideas.repec.org/p/bca/bocawp/94-5.html", "date": None,
     "language": "en", "provenance": "repec_discovery", "mime_type": "application/pdf",
     "sha256": "d73a5518868730f8c6a314989bf249d00fed1c48ba6d1ad0e6917457b11218a5",
     "local_path": "data/raw/ca/D1/0/34578afa2cca7273.pdf", "html_path": None,
     "date_precision": "month", "date_source": "repec", "repec_handle": "",
     "doc_id": "34578afa2cca7273", "year": None},
]

_COMMON = {"corpus": "central-bank", "language": "en", "mime_type": "application/pdf"}
OLD_PAYLOADS = {
    "ac6330092872d2f4": {
        "date_precision": "day", "date_source": "bank_site", "date_kind": "publication",
        "doc_id": "ac6330092872d2f4", **_COMMON, "source_code": "au", "doc_type": "A1",
        "title": "RBA monetary policy decision 1997-07-30", "date": "1997-07-30", "year": 1997,
        "sha256": "d72df39e4bfe9b9ff92c69f26f16c6bfa7a66311a17eb9c5dce5e8016a1008b6",
        "provenance": "bank_site",
        "document_url": "https://www.rba.gov.au/media-releases/1997/mr-97-15.html",
        "source_url": "https://www.rba.gov.au/monetary-policy/int-rate-decisions/1997/"},
    "ae7f793211656688": {
        "date_precision": "month", "date_source": "repec", "repec_handle": "",
        "doc_id": "ae7f793211656688", **_COMMON, "source_code": "au", "doc_type": "D1",
        "title": "Tracking Mergers and Acquisitions Using Australian Administrative Data",
        "date": "2026-07-01", "year": 2026,
        "sha256": "ad7c93f07f733488e34d2fab98cb170318ccaa5b6ca3bafcfb0d0aba5124ca6a",
        "provenance": "repec_discovery",
        "document_url": "https://www.rba.gov.au/publications/rdp/2026/pdf/rdp2026-04.pdf",
        "source_url": "https://ideas.repec.org/p/rba/rbardp/rdp2026-04.html"},
    "a5b8ba5e477ad19a": {
        "doc_id": "a5b8ba5e477ad19a", **_COMMON, "source_code": "ae", "doc_type": "C1",
        "title": "Sultan Bin Nasser Al-Suwaidi: The global financial crisis and issues for a "
                 "regulatory response",
        "date": "2010-02-05", "year": 2010,
        "sha256": "4f48a3f641bea8a028091980ca64632102703dd453d33438d0a28fa8beeece94",
        "provenance": "bis_index", "document_url": "https://www.bis.org/review/r100205e.pdf",
        "source_url": "https://www.bis.org/review/r100205e.htm"},
    "34578afa2cca7273": {
        "date_precision": "month", "date_source": "repec", "repec_handle": "",
        "doc_id": "34578afa2cca7273", **_COMMON, "source_code": "ca", "doc_type": "D1",
        "title": "Exchange Rate Volatility and Trade: A Survey",
        "sha256": "d73a5518868730f8c6a314989bf249d00fed1c48ba6d1ad0e6917457b11218a5",
        "provenance": "repec_discovery", "document_url": "https://doi.org/10.34989/swp-1994-5",
        "source_url": "https://ideas.repec.org/p/bca/bocawp/94-5.html"},
}
#: The undated row with a precision: the contract drops a precision without a
#: date (notice precision_without_date), so its new payload has no
#: date_precision. The only change; SYNC-METADATA.md names these rows.
UNDATED_WITH_PRECISION = "34578afa2cca7273"


def _migrated(rec):
    """The row as central-bank-corpus's migrate-contract rewrites it: corpus
    added, bank_code renamed source_code, local_path without data/."""
    out = {("source_code" if k == "bank_code" else k): v for k, v in rec.items()}
    out["corpus"] = "central-bank"
    out["local_path"] = rec["local_path"][len("data/"):]
    return out


def _vault_row_before_the_contract(rec):
    """The documents row a vault before the contract stored (vault 2d0457e):
    its 16 columns, every other field in extra, date_precision and date_kind
    there too; date a datetime.date."""
    import datetime

    known = {"doc_id", "corpus", "source_code", "bank_code", "doc_type", "title",
             "document_url", "pdf_url", "source_url", "date", "year", "language", "provenance",
             "mime_type", "sha256", "local_path"}
    return {
        "doc_id": rec["doc_id"], "corpus": "central-bank", "source_code": rec["bank_code"],
        "doc_type": rec["doc_type"], "title": rec["title"], "document_url": rec["document_url"],
        "source_url": rec["source_url"],
        "date": datetime.date.fromisoformat(rec["date"]) if rec["date"] else None,
        "year": rec["year"], "language": rec["language"], "provenance": rec["provenance"],
        "mime_type": rec["mime_type"], "sha256": rec["sha256"], "local_path": rec["local_path"],
        # the vault reads the four new columns as NULL on such a row
        "date_precision": None, "date_kind": None, "entity_scheme": None, "entity_value": None,
        "extra": {k: v for k, v in rec.items() if k not in known} or None,
    }


def _vault_row_on_the_contract(rec):
    """The documents row the vault stores on the contract (vault 12ffd11):
    split_row's columns (date_precision and date_kind among them), the rest
    in extra."""
    import datetime

    from vault_contract import split_row

    columns, extra = split_row(rec)
    columns["corpus"] = columns["corpus"] or "central-bank"
    if columns["date"]:
        columns["date"] = datetime.date.fromisoformat(columns["date"])
    return {**columns, "extra": extra or None}


def _expected(doc_id, *, undated_precision_kept=False):
    want = dict(OLD_PAYLOADS[doc_id])
    if doc_id == UNDATED_WITH_PRECISION and not undated_precision_kept:
        del want["date_precision"]
    return want


@pytest.mark.parametrize("rec", CB_ROWS, ids=lambda r: r["doc_id"])
@pytest.mark.parametrize("form", ["legacy manifest", "migrated manifest",
                                  "vault row before the contract", "vault row on the contract"])
def test_a_central_bank_payload_is_the_one_its_points_were_ingested_with(rec, form):
    """Same keys, same values: ``date_precision`` and ``date_kind`` now come
    from their columns instead of ``extra`` (a column is a payload key under
    the same name), so no Qdrant point of the central-bank collection needs a
    rewrite, whichever form the manifest or the vault row is in. The one
    exception, by the contract: an undated row's precision is dropped."""
    if form == "legacy manifest":
        payload, dropped = pl.manifest_row_to_payload(rec, "central-bank")
    elif form == "migrated manifest":
        payload, dropped = pl.manifest_row_to_payload(_migrated(rec), "central-bank")
    elif form == "vault row before the contract":
        payload, dropped = pl.row_to_payload(_vault_row_before_the_contract(rec))
    else:
        payload, dropped = pl.row_to_payload(_vault_row_on_the_contract(rec))
    # A vault row stored before the contract still has the precision in extra.
    kept = form == "vault row before the contract"
    assert payload == _expected(rec["doc_id"], undated_precision_kept=kept)
    assert dropped == ()


def test_the_engine_writes_the_same_points_for_both_vault_rows():
    """What the engine writes from the payload (``build_payload``) is the same
    for the vault row before and on the contract, interval included."""
    from rag_engine.schema import build_payload

    for rec in CB_ROWS[:3]:
        points = [build_payload(rec["doc_id"], pl.row_to_payload(row)[0], filename="f.pdf",
                                chunk_number=0, ingestion_date="t", text="x", page=1)
                  for row in (_vault_row_before_the_contract(rec),
                              _vault_row_on_the_contract(rec))]
        assert points[0] == points[1]
