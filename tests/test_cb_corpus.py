"""Unit tests for the cb_corpus connector's disk walk and its config guards."""
import datetime
import json

import pytest

from data_orchestrator.sources.cb_corpus import ROOT_ENV_KEY, iter_items


def _corpus(tmp_path, *rel_paths):
    """Build a synthetic ``<root>/raw/<bank>/<doctype>/<year>/<file>`` tree."""
    root = tmp_path / "corpus"
    for rel in rel_paths:
        f = root / "raw" / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"%PDF-1.4 stub")
    return root


def _ids(root, **kwargs):
    return [item.doc_id for item in iter_items(root, prefer_manifest=False, **kwargs)]


def test_walk_yields_every_pdf_under_raw(tmp_path):
    """The disk is the source of truth for completeness: every PDF under raw/
    must be yielded, whatever the manifest knows, and non-document files
    (.DS_Store and friends) must not become documents."""
    root = _corpus(
        tmp_path,
        "us/C1/2015/a.pdf",
        "us/D1/2020/b.pdf",
        "ecb/A3/2019/c.pdf",
        "ecb/A3/2019/.DS_Store",
    )
    assert sorted(_ids(root)) == ["a", "b", "c"]


def test_bank_allow_list_is_case_insensitive(tmp_path):
    """Bank codes come from CLI flags typed by hand ("ECB"), while the on-disk
    folder is lowercase — the filter must not depend on which one was typed."""
    root = _corpus(tmp_path, "us/C1/2015/a.pdf", "ecb/C1/2015/b.pdf")
    assert _ids(root, banks=["ECB"]) == ["b"]


def test_doctype_allow_list_is_case_insensitive(tmp_path):
    """Same for doctype codes: the taxonomy is upper-case ("C1") but the folder
    on disk may not be, and only the allowed doctype may survive."""
    root = _corpus(tmp_path, "us/c1/2015/a.pdf", "us/A3/2015/b.pdf")
    assert _ids(root, doctypes=["C1"]) == ["a"]


def test_group_allow_list_selects_a_whole_doctype_family(tmp_path):
    """A group is the doctype's first letter, so "C" must select C1 and C2
    together — that is how a caller asks for "all the speeches"."""
    root = _corpus(
        tmp_path, "us/C1/2015/speech.pdf", "us/C2/2015/interview.pdf",
        "us/A3/2015/minutes.pdf",
    )
    assert sorted(_ids(root, groups=["C"])) == ["interview", "speech"]


def test_year_bounds_are_inclusive(tmp_path):
    """The bounds define a point-in-time slice of the corpus; an off-by-one at
    either edge silently drops (or adds) a full year of documents."""
    root = _corpus(
        tmp_path, "us/C1/2014/a.pdf", "us/C1/2015/b.pdf",
        "us/C1/2016/c.pdf", "us/C1/2017/d.pdf",
    )
    assert sorted(_ids(root, year_min=2015, year_max=2016)) == ["b", "c"]


def test_html_is_never_yielded_and_include_html_is_gone(tmp_path):
    """HTML is converted to .md/PDF before the engine (issue #17): the walk
    yields .pdf only, whether or not the .html has a PDF sibling, and the old
    include_html opt-in no longer exists."""
    root = _corpus(
        tmp_path, "us/C1/2015/both.pdf", "us/C1/2015/both.html",
        "us/C1/2015/online-only.html",
    )
    items = list(iter_items(root, prefer_manifest=False))
    assert [(i.doc_id, i.path.suffix) for i in items] == [("both", ".pdf")]
    with pytest.raises(TypeError):
        list(iter_items(root, prefer_manifest=False, include_html=True))


def _manifest(root, bank, *rows):
    """Write ``<root>/manifest/<bank>.jsonl`` — the per-bank layout cb_corpus
    produces (the single-file ``manifest.jsonl`` no longer exists upstream)."""
    d = root / "manifest"
    d.mkdir(exist_ok=True)
    (d / f"{bank}.jsonl").write_text("".join(r + "\n" for r in rows), encoding="utf-8")


def test_prefer_manifest_false_ignores_an_existing_manifest(tmp_path):
    """Opting out of the manifest must yield purely path-derived metadata even
    when a manifest sits right next to raw/ — otherwise the flag cannot be used
    to test (or reproduce) the disk-only view of the corpus."""
    root = _corpus(tmp_path, "us/C1/2015/a.pdf")
    _manifest(root, "us", '{"doc_id": "a", "date": "2015-06-30", "title": "Speech"}')
    (item,) = iter_items(root, prefer_manifest=False)
    assert item.payload["metadata_source"] == "path"
    assert item.payload["date"] == "2015-01-01"
    assert "title" not in item.payload


def test_manifest_hit_carries_the_exact_date(tmp_path):
    """The manifest is the source of truth for metadata: an indexed document
    must get its real publication date and title, flagged as coming from the
    manifest so the point-in-time layer can trust the date."""
    root = _corpus(tmp_path, "us/C1/2015/a.pdf")
    _manifest(root, "us", '{"doc_id": "a", "date": "2015-06-30", "title": "A speech",'
                          ' "document_url": "https://example.org/a.pdf", "sha256": "ab12",'
                          ' "provenance": "bis"}')
    (item,) = iter_items(root)
    assert item.payload["metadata_source"] == "manifest"
    assert item.payload["date"] == "2015-06-30"
    assert item.payload["title"] == "A speech"
    assert item.payload["document_url"] == "https://example.org/a.pdf"
    assert item.payload["sha256"] == "ab12"
    assert item.payload["provenance"] == "bis"


def test_a_manifest_row_not_migrated_yet_gives_document_url(tmp_path):
    """A manifest row written before the rename names the address ``pdf_url``:
    the payload carries it as ``document_url``, never under the old name."""
    root = _corpus(tmp_path, "us/C1/2015/a.pdf")
    _manifest(root, "us", '{"doc_id": "a", "pdf_url": "https://example.org/a.pdf"}')
    (item,) = iter_items(root)
    assert item.payload["document_url"] == "https://example.org/a.pdf"
    assert "pdf_url" not in item.payload


def test_manifest_miss_falls_back_to_the_path_year(tmp_path):
    """A file the manifest has not indexed yet must still be yielded — with a
    conservative <year>-01-01 date flagged as path-derived, so a later reindex
    can upgrade it instead of the document being silently dropped."""
    root = _corpus(tmp_path, "us/C1/2015/indexed.pdf", "us/C1/2015/fresh.pdf")
    _manifest(root, "us", '{"doc_id": "indexed", "date": "2015-06-30"}')
    payloads = {i.doc_id: i.payload for i in iter_items(root)}
    assert payloads["fresh"] == {
        "date_precision": "year", "doc_id": "fresh", "corpus": "central-bank",
        "source_code": "us", "doc_type": "C1", "date": "2015-01-01", "year": 2015,
        "metadata_source": "path"}
    assert payloads["indexed"]["metadata_source"] == "manifest"


def test_malformed_manifest_line_does_not_cost_its_neighbours(tmp_path):
    """A torn line (a crashed writer) must cost exactly the document it
    describes: the rows around it stay indexed instead of the whole corpus
    degrading to path-derived metadata."""
    root = _corpus(tmp_path, "us/C1/2015/a.pdf", "us/C1/2015/b.pdf")
    _manifest(
        root, "us",
        '{"doc_id": "a", "date": "2015-06-30"}',
        '{"doc_id": "b", "date": "2015-0',  # torn tail
        '{"doc_id": "b", "date": "2015-07-31"}',
    )
    payloads = {i.doc_id: i.payload for i in iter_items(root)}
    assert payloads["a"]["date"] == "2015-06-30"
    assert payloads["b"]["date"] == "2015-07-31"


def test_per_bank_manifest_layout_is_read(tmp_path):
    """cb_corpus splits its manifest per bank (``manifest/<bank>.jsonl``); the
    connector must read that layout, or every real corpus document loses its
    exact publication date without any error being raised (issue #5)."""
    root = _corpus(tmp_path, "us/C1/2015/a.pdf")
    _manifest(root, "us", '{"doc_id": "a", "date": "2015-06-30"}')
    (item,) = iter_items(root)
    assert item.payload["metadata_source"] == "manifest"
    assert item.payload["date"] == "2015-06-30"


def test_all_bank_files_are_merged_into_one_index(tmp_path):
    """Every ``manifest/<bank>.jsonl`` feeds the index: a document indexed in
    ``ecb.jsonl`` must be enriched just like one indexed in ``us.jsonl`` —
    reading only the first file would silently degrade every other bank."""
    root = _corpus(tmp_path, "us/C1/2015/a.pdf", "ecb/A3/2019/c.pdf")
    _manifest(root, "us", '{"doc_id": "a", "date": "2015-06-30", "title": "US"}')
    _manifest(root, "ecb", '{"doc_id": "c", "date": "2019-03-07", "title": "ECB"}')
    payloads = {i.doc_id: i.payload for i in iter_items(root)}
    assert payloads["a"]["metadata_source"] == "manifest"
    assert payloads["a"]["date"] == "2015-06-30"
    assert payloads["c"]["metadata_source"] == "manifest"
    assert payloads["c"]["date"] == "2019-03-07"
    assert payloads["c"]["title"] == "ECB"


def test_single_file_manifest_is_not_recognised(tmp_path):
    """The legacy single ``manifest.jsonl`` is gone upstream and is deliberately
    NOT read (pre-release: clean break, no compat shim). With a per-bank dir
    present, a stray single file must have no effect on the index."""
    root = _corpus(tmp_path, "us/C1/2015/a.pdf", "us/C1/2015/b.pdf")
    _manifest(root, "us", '{"doc_id": "a", "date": "2015-06-30"}')
    (root / "manifest.jsonl").write_text('{"doc_id": "b", "date": "2015-07-31"}\n', encoding="utf-8")
    payloads = {i.doc_id: i.payload for i in iter_items(root)}
    assert payloads["a"]["metadata_source"] == "manifest"
    assert payloads["b"]["metadata_source"] == "path"


def test_unconfigured_root_raises_an_actionable_value_error(monkeypatch):
    """With no root argument and no CB_CORPUS_ROOT, the error must name the key
    and the two ways to set it — this is the first thing a new machine hits."""
    monkeypatch.delenv(ROOT_ENV_KEY, raising=False)  # explicit precondition
    with pytest.raises(ValueError) as excinfo:
        list(iter_items())
    message = str(excinfo.value)
    assert ROOT_ENV_KEY in message
    assert ".env" in message and "--root" in message


def test_root_without_raw_dir_raises_an_actionable_file_not_found(tmp_path):
    """Pointing at the repo root instead of its data dir is the classic
    misconfiguration; the error must show the raw/ path that was expected
    rather than yielding zero documents and looking like an empty corpus."""
    root = tmp_path / "corpus"
    root.mkdir()
    with pytest.raises(FileNotFoundError) as excinfo:
        list(iter_items(root))
    assert str(root / "raw") in str(excinfo.value)


def test_missing_manifest_dir_is_an_actionable_error(tmp_path):
    """No ``manifest/`` under the root means a mis-pointed CB_CORPUS_ROOT or an
    unsynced share — NOT a corpus whose every document happens to be undated.
    Fail with the expected path instead of ingesting a fully degraded corpus."""
    root = _corpus(tmp_path, "us/C1/2015/a.pdf")
    with pytest.raises(FileNotFoundError) as excinfo:
        list(iter_items(root))
    assert str(root / "manifest") in str(excinfo.value)


def test_empty_manifest_dir_is_the_same_error(tmp_path):
    """An existing but empty ``manifest/`` (no ``*.jsonl``) is just as wrong."""
    root = _corpus(tmp_path, "us/C1/2015/a.pdf")
    (root / "manifest").mkdir()
    with pytest.raises(FileNotFoundError) as excinfo:
        list(iter_items(root))
    assert str(root / "manifest") in str(excinfo.value)


def test_prefer_manifest_false_needs_no_manifest_at_all(tmp_path):
    """The disk-only mode is the explicit escape hatch: no manifest, no error,
    purely path-derived metadata."""
    root = _corpus(tmp_path, "us/C1/2015/a.pdf")
    (item,) = iter_items(root, prefer_manifest=False)
    assert item.payload["metadata_source"] == "path"


def test_non_numeric_year_dir_is_excluded_only_when_a_bound_is_set(tmp_path):
    """``raw/<bank>/<type>/undated/`` has no year: without bounds it is walked
    like today, but a --year-min/--year-max slice must not smuggle it in
    (issue #5, third checkbox) — an unknown year cannot satisfy a bound."""
    root = _corpus(tmp_path, "us/C1/2015/a.pdf", "us/C1/undated/u.pdf")
    assert sorted(_ids(root)) == ["a", "u"]
    assert _ids(root, year_min=2000) == ["a"]
    assert _ids(root, year_max=2020) == ["a"]


@pytest.mark.parametrize("indexed", [True, False])
def test_the_payload_never_sets_a_key_the_engine_writes_itself(tmp_path, indexed):
    """No key the engine refuses (filename, text, page, ...): the only reserved
    key is ``doc_id``, the document's own, which the engine accepts (the vault
    source sends it too)."""
    from rag_engine.schema import RESERVED_KEYS, build_payload

    root = _corpus(tmp_path, "us/C1/2015/a.pdf")
    if indexed:
        _manifest(root, "us", '{"doc_id": "a", "date": "2015-06-30", "title": "A speech"}')
    (item,) = iter_items(root, prefer_manifest=indexed)
    assert set(item.payload) & set(RESERVED_KEYS) == {"doc_id"}
    assert item.payload["doc_id"] == item.doc_id
    assert item.dropped_keys == ()
    build_payload(item.doc_id, item.payload, filename="a.pdf", chunk_number=0,
                  ingestion_date="t", text="t", page=1)  # does not raise


# ---------------------------------------------------------------------------
# The vault's vocabulary (F12): a document gets the same payload from the disk
# source and from the vault source.
# ---------------------------------------------------------------------------
def test_an_undated_path_document_has_no_date_and_no_precision(tmp_path):
    root = _corpus(tmp_path, "us/C1/undated/u.pdf")
    (item,) = iter_items(root, prefer_manifest=False)
    assert item.payload == {"doc_id": "u", "corpus": "central-bank", "source_code": "us",
                            "doc_type": "C1", "metadata_source": "path"}


def test_the_corpus_of_a_disk_payload_is_the_run_s(tmp_path):
    root = _corpus(tmp_path, "us/C1/2015/a.pdf", "us/C1/2015/b.pdf")
    _manifest(root, "us", '{"doc_id": "a", "date": "2015-06-30"}',
              '{"doc_id": "b", "corpus": "own", "date": "2015-06-30"}')
    payloads = {i.doc_id: i.payload for i in iter_items(root, corpus="scratch")}
    assert payloads["a"]["corpus"] == "scratch"
    assert payloads["b"]["corpus"] == "own"  # the row's own corpus, like the vault column
    (item,) = iter_items(_corpus(tmp_path / "p", "us/C1/2015/c.pdf"), prefer_manifest=False,
                         corpus="scratch")
    assert item.payload["corpus"] == "scratch"


def test_a_manifest_row_with_unknown_dropped_and_reserved_keys(tmp_path):
    """Unknown keys are kept (they are the vault's ``extra``), keys dropped by
    design are left out silently, a reserved key is left out and counted, a
    null or absent field is absent, never an empty string."""
    root = _corpus(tmp_path, "us/C1/2015/a.pdf")
    _manifest(root, "us", json.dumps({
        "doc_id": "a", "bank_code": "us", "doc_type": "C1", "title": None, "date": "2015-06-30",
        "year": 2015, "html_path": "/Users/someone/a.html", "alt_urls": ["https://alt"],
        "alt_source_urls": [], "text": "stale", "repec_handle": "RePEc:x", "speaker": "J. Doe"}))
    (item,) = iter_items(root)
    assert item.payload == {
        "repec_handle": "RePEc:x", "speaker": "J. Doe", "doc_id": "a", "corpus": "central-bank",
        "source_code": "us", "doc_type": "C1", "date": "2015-06-30", "year": 2015,
        "metadata_source": "manifest"}
    assert item.dropped_keys == ("text",)


# Manifest rows, adversarial on purpose: a legacy bank_code, a legacy pdf_url
# (a row not migrated to document_url yet, one with both names), NULL columns, an
# extra-bound key named like a payload key that the vault keeps as a column,
# keys dropped by design next to a reserved key, unknown keys, a date with a
# time, a year as a string.
PARITY_ROWS = [
    {"doc_id": "a", "corpus": "central-bank", "source_code": "us", "doc_type": "C1",
     "title": "A speech", "document_url": "https://x.org/a.pdf", "source_url": "https://x.org/a",
     "date": "2015-06-30", "year": 2015, "language": "en", "provenance": "bis",
     "mime_type": "application/pdf", "sha256": "ab12", "local_path": "data/raw/us/C1/2015/a.pdf",
     "date_precision": "day", "date_kind": "publication", "date_source": "title"},
    {"doc_id": "b", "bank_code": "us", "doc_type": "C1", "title": None, "pdf_url": None,
     "date": "", "year": "2015", "html_path": "/Users/someone/b.html",
     "alt_urls": ["https://alt/b"], "alt_source_urls": ["https://alt/b2"], "page": 7,
     "chunk_number": 1, "repec_handle": "RePEc:b", "nested": {"k": [1, 2]}},
    {"doc_id": "c", "source_code": "", "bank_code": "us", "doc_type": "C1",
     "date": "2015-03-01T09:00:00Z", "year": 2015, "date_precision": "month",
     "local_path": "data/raw/us/C1/2015/c.pdf", "provenance": None, "flag": False,
     "source_url": "", "mime_type": "  ", "date_kind": None, "pdf_url": "https://x.org/c.pdf"},
    {"doc_id": "d", "source_code": "us", "doc_type": "C1", "date": "2015-04-01",
     "document_url": "", "pdf_url": "https://x.org/d.pdf"},
]


def _vault_row_by_hand(rec):
    """The ``documents`` row the vault stores for ``rec``, written out here
    independently of ``payload.manifest_row_to_vault_row``: the columns of the
    vault's KNOWN_FIELDS, ``date`` a ``datetime.date``, ``year`` an integer, the
    other fields in ``extra`` after a JSON round trip (JSONB)."""
    known = {"doc_id", "corpus", "source_code", "bank_code", "doc_type", "title",
             "document_url", "pdf_url", "source_url", "date", "year", "language", "provenance",
             "mime_type", "sha256", "local_path"}
    date = rec.get("date") or None
    extra = {k: v for k, v in rec.items() if k not in known}
    return {
        "doc_id": rec["doc_id"], "corpus": rec.get("corpus") or "central-bank",
        "source_code": rec.get("source_code") or rec.get("bank_code"),
        "doc_type": rec.get("doc_type"), "title": rec.get("title"),
        "document_url": rec.get("document_url") or rec.get("pdf_url"),
        "source_url": rec.get("source_url"),
        "date": datetime.date.fromisoformat(date[:10]) if date else None,
        "year": int(rec["year"]) if rec.get("year") is not None else None,
        "language": rec.get("language"), "provenance": rec.get("provenance"),
        "mime_type": rec.get("mime_type"), "sha256": rec.get("sha256"),
        "local_path": rec.get("local_path") or f"data/raw/us/C1/2015/{rec['doc_id']}.pdf",
        "extra": json.loads(json.dumps(extra)) if extra else None,
    }


@pytest.mark.parametrize("drop", [None, "none", "repec_handle"])
def test_the_disk_and_the_vault_sources_give_the_same_payload(tmp_path, monkeypatch, drop):
    from data_orchestrator.sources.vault import row_to_item

    if drop is not None:
        monkeypatch.setenv("DATA_ORCHESTRATOR_PAYLOAD_DROP_EXTRA_KEYS", drop)
    root = _corpus(tmp_path, *(f"us/C1/2015/{r['doc_id']}.pdf" for r in PARITY_ROWS))
    _manifest(root, "us", *(json.dumps(r) for r in PARITY_ROWS))
    monkeypatch.setenv(ROOT_ENV_KEY, str(root))
    disk = {i.doc_id: i for i in iter_items(root)}
    assert len(disk) == len(PARITY_ROWS)
    for rec in PARITY_ROWS:
        vault = row_to_item(_vault_row_by_hand(rec), "central-bank")
        payload = dict(disk[rec["doc_id"]].payload)
        assert payload.pop("metadata_source") == "manifest"
        assert payload == vault.payload, rec["doc_id"]
        assert disk[rec["doc_id"]].dropped_keys == vault.dropped_keys
    # A blank link and a null field are left out, whichever the source.
    assert not {"source_url", "mime_type", "date_kind"} & set(disk["c"].payload)
    # The legacy pdf_url of a row not migrated yet is carried as document_url.
    assert disk["c"].payload["document_url"] == "https://x.org/c.pdf"
    assert disk["d"].payload["document_url"] == "https://x.org/d.pdf"
    assert not any("pdf_url" in i.payload for i in disk.values())
