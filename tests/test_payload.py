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
    ("text,page", "text, page cannot be dropped"),
    ("date_start", "date_start cannot be dropped"),
])
def test_the_setting_refuses_what_it_cannot_drop(monkeypatch, raw, match):
    monkeypatch.setenv(ENV, raw)
    with pytest.raises(ValueError, match=match):
        pl.dropped_extra_keys()


@pytest.mark.parametrize("old", ["pdf_url", "bank_code"])
def test_a_former_column_name_can_be_dropped_like_any_extra_key(monkeypatch, old):
    """The contract's transition is over: ``pdf_url`` and ``bank_code`` are
    ordinary keys of ``extra``, which the setting may drop."""
    monkeypatch.setenv(ENV, f"html_path,{old}")
    assert pl.dropped_extra_keys() == ("html_path", old)


# ---------------------------------------------------------------------------
# A manifest row, split like the vault splits it (vault_contract.split_row,
# transition=False: the contract proper, as the vault ingests since its
# transition cleanup)
# ---------------------------------------------------------------------------
def test_a_manifest_row_is_split_into_the_vault_s_columns_and_extra():
    rec = {"doc_id": "a", "corpus": "central-bank", "source_code": "us", "doc_type": "C1",
           "title": "T", "document_url": "https://x/a.pdf", "date": "2015-06-30",
           "year": 2015, "local_path": "raw/us/C1/2015/a.pdf", "date_precision": "day",
           "html_path": "/Users/x/a.html", "unknown_key": {"nested": [1, 2]}}
    row = pl.manifest_row_to_vault_row(rec, "central-bank")
    assert row["corpus"] == "central-bank" and row["source_code"] == "us"
    assert row["year"] == 2015 and row["date"] == "2015-06-30"
    assert row["date_precision"] == "day" and row["date_kind"] is None
    assert row["local_path"] == "raw/us/C1/2015/a.pdf"
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
    ({"document_url": "https://new", "pdf_url": "https://old"}, "https://new"),
    ({"document_url": "", "pdf_url": "https://old"}, None),
    ({"pdf_url": "https://old"}, None),
    ({}, None),
])
def test_document_url_comes_from_document_url_only(rec, expected):
    """No alias any more: a ``pdf_url`` never fills ``document_url``, it is
    an ordinary field of ``extra``."""
    row = pl.manifest_row_to_vault_row(rec, "c")
    assert row["document_url"] == expected
    assert (row["extra"] or {}).get("pdf_url") == rec.get("pdf_url")


def test_bank_code_never_fills_source_code():
    assert pl.manifest_row_to_vault_row({"source_code": "ecb", "bank_code": "fr"},
                                        "c")["source_code"] == "ecb"
    row = pl.manifest_row_to_vault_row({"source_code": "", "bank_code": "fr"}, "c")
    assert not row["source_code"] and row["extra"] == {"bank_code": "fr"}


def test_a_data_prefix_is_no_longer_stripped_from_local_path():
    """The vault refuses such a row (bad_local_path); the split leaves the
    path as it is."""
    row = pl.manifest_row_to_vault_row({"local_path": "data/raw/a.pdf"}, "c")
    assert row["local_path"] == "data/raw/a.pdf"


@pytest.mark.parametrize("old", ["pdf_url", "bank_code"])
def test_a_former_column_name_in_extra_is_an_ordinary_key_of_the_payload(old):
    row = {"doc_id": "a", "document_url": None, "extra": {old: "kept", "s": 1}}
    assert pl.row_to_payload(row) == ({old: "kept", "s": 1, "doc_id": "a"}, ())


def test_the_transition_names_are_gone():
    for name in ("LEGACY_COLUMN_NAMES", "CONTRACT_COLUMNS_FROM_EXTRA"):
        assert not hasattr(pl, name), name


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


def test_every_payload_column_is_a_contract_column_and_local_path_is_not_carried():
    from vault_contract import COLUMNS

    assert set(pl.PAYLOAD_COLUMNS) == set(COLUMNS) - {"local_path"}


def test_a_contract_column_wins_over_the_same_key_of_extra():
    row = {"doc_id": "a", "date_precision": "day", "entity_value": "0000320193",
           "extra": {"date_precision": "month", "entity_value": "320193"}}
    payload, _ = pl.row_to_payload(row)
    assert payload["date_precision"] == "day" and payload["entity_value"] == "0000320193"


@pytest.mark.parametrize("column", ["date_precision", "date_kind", "entity_scheme",
                                    "entity_value", "title"])
def test_a_null_contract_column_removes_the_same_key_of_extra(column):
    """No extra fallback for the four columns the contract added: every live
    vault row was rewritten on the contract (gate query, README: 0 rows), so a
    NULL column removes the same key of ``extra``, as every column does."""
    row = {"doc_id": "a", column: None, "extra": {column: "stale", "s": 1}}
    payload, _ = pl.row_to_payload(row)
    assert payload == {"s": 1, "doc_id": "a"}


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


#: The golden case of the pre-contract central-bank form: accepted with
#: transition=True only. The vault ingests with transition=False and refuses it.
GOLDEN_LEGACY_FORM = "cb_legacy_form"


@pytest.mark.parametrize("name", sorted(set(golden.accepted()) - {GOLDEN_LEGACY_FORM}))
def test_the_split_of_every_golden_row_is_the_contract_s(name):
    """The contract's reference rows (shipped in the package, the cases the
    vault tests its ingestion with): the vault row built here holds exactly
    their expected columns, and the vault accepts them as it ingests."""
    from vault_contract import check_row

    case = golden.accepted()[name]
    assert check_row(case["row"], None, expected_corpus=case["expected_corpus"],
                     transition=False).accepted
    row = pl.manifest_row_to_vault_row(case["row"], case["expected_corpus"])
    assert {k: v for k, v in row.items() if k != "extra"} == case["expected_columns"]


def test_the_golden_legacy_form_is_refused_as_the_vault_ingests():
    from vault_contract import check_row

    case = golden.accepted()[GOLDEN_LEGACY_FORM]
    checked = check_row(case["row"], None, expected_corpus=case["expected_corpus"],
                        transition=False)
    assert {r.reason for r in checked.refusals} == {"missing_corpus", "missing_source_code",
                                                    "bad_local_path"}


# ---------------------------------------------------------------------------
# Central-bank points ingested before the contract keep their payload
# ---------------------------------------------------------------------------
# Five real rows of the public central-bank manifests (repo central-bank-corpus,
# data/manifest, commit ea679d4), as the producer wrote them before the
# contract: a precision and a kind, a month precision, no precision, one of
# the 7 undated rows that carry a precision, and one of the 110 rows with a
# blank title. OLD_PAYLOADS is what this repo's
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
    {"bank_code": "ca", "doc_type": "E1", "title": "",
     "document_url": "https://www.bankofcanada.ca/publications/mpr/mpr-2026-10-28/in-focus-3/",
     "source_url": "https://www.bankofcanada.ca/wp-sitemap.xml", "date": "2026-10-28",
     "language": "en", "provenance": "bank_site", "mime_type": "application/pdf",
     "sha256": "79cf7158ea4c4b7aa6502af030244d70f5d4f2d70f7aa1a6db0ef95bdcce1276",
     "local_path": "data/raw/ca/E1/2026/0149e9c0c8a10104.pdf",
     "html_path": "data/raw/ca/E1/2026/0149e9c0c8a10104.html", "date_precision": "day",
     "date_source": "bank_site", "repec_handle": "", "alt_urls": [],
     "doc_id": "0149e9c0c8a10104", "year": 2026, "date_kind": "publication"},
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
    "0149e9c0c8a10104": {
        "date_precision": "day", "date_source": "bank_site", "repec_handle": "",
        "date_kind": "publication", "doc_id": "0149e9c0c8a10104", **_COMMON,
        "source_code": "ca", "doc_type": "E1", "title": "", "date": "2026-10-28", "year": 2026,
        "sha256": "79cf7158ea4c4b7aa6502af030244d70f5d4f2d70f7aa1a6db0ef95bdcce1276",
        "provenance": "bank_site",
        "document_url": "https://www.bankofcanada.ca/publications/mpr/mpr-2026-10-28/in-focus-3/",
        "source_url": "https://www.bankofcanada.ca/wp-sitemap.xml"},
}
#: The undated row with a precision: the contract drops a precision without a
#: date (notice precision_without_date), so its new payload has no
#: date_precision. The only change; SYNC-METADATA.md names these rows.
UNDATED_WITH_PRECISION = "34578afa2cca7273"
#: One of the 110 rows with a blank title: the contract reads a blank optional
#: value as absent, so its payload has no title (it had ``""``), whichever the
#: form (a vault row stored before the contract holds ``""``, read as absent).
BLANK_TITLE = "0149e9c0c8a10104"


def _migrated(rec):
    """The row as central-bank-corpus's migrate-contract rewrites it: corpus
    added, bank_code renamed source_code, local_path without data/."""
    out = {("source_code" if k == "bank_code" else k): v for k, v in rec.items()}
    out["corpus"] = "central-bank"
    out["local_path"] = rec["local_path"][len("data/"):]
    return out


def _vault_row_on_the_contract(rec):
    """The documents row the vault stores on the contract (vault 2859fa6,
    transition=False): split_row's columns (date_precision and date_kind
    among them), the rest in extra; ``date`` a ``datetime.date``."""
    import datetime

    from vault_contract import split_row

    columns, extra = split_row(rec, transition=False)
    if columns["date"]:
        columns["date"] = datetime.date.fromisoformat(columns["date"])
    return {**columns, "extra": extra or None}


def _expected(doc_id):
    want = dict(OLD_PAYLOADS[doc_id])
    if doc_id == UNDATED_WITH_PRECISION:
        del want["date_precision"]
    if doc_id == BLANK_TITLE:
        del want["title"]
    return want


@pytest.mark.parametrize("rec", CB_ROWS, ids=lambda r: r["doc_id"])
@pytest.mark.parametrize("form", ["migrated manifest", "vault row on the contract"])
def test_a_central_bank_payload_is_the_one_its_points_were_ingested_with(rec, form):
    """Same keys, same values, from the row in contract form (the manifest as
    migrate-contract rewrote it, or the vault row the vault stores for it):
    no Qdrant point of the central-bank collection needs a rewrite. Two
    exceptions, by the contract: an undated row's precision is dropped (7
    public rows) and a blank title is absent (110 public rows)."""
    if form == "migrated manifest":
        payload, dropped = pl.manifest_row_to_payload(_migrated(rec), "central-bank")
    else:
        payload, dropped = pl.row_to_payload(_vault_row_on_the_contract(_migrated(rec)))
    assert payload == _expected(rec["doc_id"])
    assert dropped == ()


@pytest.mark.parametrize("rec", CB_ROWS, ids=lambda r: r["doc_id"])
def test_a_central_bank_row_in_contract_form_is_accepted_as_the_vault_ingests(rec):
    from vault_contract import check_row

    assert check_row(_migrated(rec), None, expected_corpus="central-bank",
                     transition=False).accepted


@pytest.mark.parametrize("rec", CB_ROWS, ids=lambda r: r["doc_id"])
def test_a_central_bank_row_in_the_legacy_form_is_refused_and_not_mapped(rec):
    """The pre-contract form (``bank_code``, no ``corpus``, ``data/raw/...``)
    is refused as the vault ingests (transition=False), and its split maps no
    old key: ``bank_code`` stays an ordinary extra field, ``source_code`` is
    absent."""
    from vault_contract import check_row

    checked = check_row(rec, None, expected_corpus="central-bank", transition=False)
    assert {r.reason for r in checked.refusals} == {"missing_corpus", "missing_source_code",
                                                    "bad_local_path"}
    payload, _ = pl.manifest_row_to_payload(rec, "central-bank")
    assert "source_code" not in payload and payload["bank_code"] == rec["bank_code"]


def test_the_engine_writes_the_same_points_from_the_manifest_and_the_vault_row():
    """What the engine writes from the payload (``build_payload``) is the same
    for the migrated manifest row and the vault row, interval included."""
    from rag_engine.schema import build_payload

    for rec in CB_ROWS:
        points = [build_payload(rec["doc_id"], payload, filename="f.pdf", chunk_number=0,
                                ingestion_date="t", text="x", page=1)
                  for payload in (pl.manifest_row_to_payload(_migrated(rec), "central-bank")[0],
                                  pl.row_to_payload(_vault_row_on_the_contract(
                                      _migrated(rec)))[0])]
        assert points[0] == points[1]
