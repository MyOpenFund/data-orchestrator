"""Unit tests for the cb_corpus connector's disk walk and its config guards."""
import pytest

from rag_orchestrator.sources.cb_corpus import ROOT_ENV_KEY, iter_items


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


def test_html_is_ignored_unless_asked_for(tmp_path):
    """PDFs are the default artifact: an .html sibling-less page is still not
    yielded unless the caller opts in with include_html."""
    root = _corpus(tmp_path, "us/C1/2015/a.pdf", "us/C1/2015/b.html")
    assert _ids(root) == ["a"]


def test_html_yielded_only_when_it_has_no_pdf_sibling(tmp_path):
    """With include_html, an .html that also exists as .pdf must be skipped:
    both would carry the same doc_id (the filename stem), so yielding both
    would ingest the same document twice under one id."""
    root = _corpus(
        tmp_path, "us/C1/2015/both.pdf", "us/C1/2015/both.html",
        "us/C1/2015/online-only.html",
    )
    items = list(iter_items(root, prefer_manifest=False, include_html=True))
    assert sorted((i.doc_id, i.payload["ext"]) for i in items) == [
        ("both", "pdf"), ("online-only", "html"),
    ]


def test_prefer_manifest_false_ignores_an_existing_manifest(tmp_path):
    """Opting out of the manifest must yield purely path-derived metadata even
    when a manifest sits right next to raw/ — otherwise the flag cannot be used
    to test (or reproduce) the disk-only view of the corpus."""
    root = _corpus(tmp_path, "us/C1/2015/a.pdf")
    (root / "manifest.jsonl").write_text(
        '{"doc_id": "a", "date": "2015-06-30", "title": "Speech"}\n', encoding="utf-8"
    )
    (item,) = iter_items(root, prefer_manifest=False)
    assert item.payload["metadata_source"] == "path"
    assert item.payload["publication_date"] == "2015-01-01"
    assert item.payload["title"] == ""


def _legacy_manifest(root, *rows):
    """Write the single-file manifest layout this connector still looks for.

    ``_find_manifest`` searches ``<root>/manifest.jsonl`` and
    ``<root>/data/manifest.jsonl`` — see the issue-#5 test below.
    """
    (root / "manifest.jsonl").write_text("".join(r + "\n" for r in rows), encoding="utf-8")


def test_manifest_hit_carries_the_exact_publication_date(tmp_path):
    """The manifest is the source of truth for metadata: an indexed document
    must get its real publication date and title, flagged as coming from the
    manifest so the point-in-time layer can trust the date."""
    root = _corpus(tmp_path, "us/C1/2015/a.pdf")
    _legacy_manifest(root, '{"doc_id": "a", "date": "2015-06-30", "title": "A speech",'
                           ' "pdf_url": "https://example.org/a.pdf", "sha256": "ab12",'
                           ' "provenance": "bis"}')
    (item,) = iter_items(root)
    assert item.payload["metadata_source"] == "manifest"
    assert item.payload["publication_date"] == "2015-06-30"
    assert item.payload["date_granularity"] == "source"
    assert item.payload["title"] == "A speech"
    assert item.payload["url"] == "https://example.org/a.pdf"
    assert item.payload["sha256"] == "ab12"


def test_manifest_miss_falls_back_to_the_path_year(tmp_path):
    """A file the manifest has not indexed yet must still be yielded — with a
    conservative <year>-01-01 date flagged as path-derived, so a later reindex
    can upgrade it instead of the document being silently dropped."""
    root = _corpus(tmp_path, "us/C1/2015/indexed.pdf", "us/C1/2015/fresh.pdf")
    _legacy_manifest(root, '{"doc_id": "indexed", "date": "2015-06-30"}')
    payloads = {i.doc_id: i.payload for i in iter_items(root)}
    assert payloads["fresh"]["metadata_source"] == "path"
    assert payloads["fresh"]["publication_date"] == "2015-01-01"
    assert payloads["fresh"]["date_granularity"] == "year"
    assert payloads["fresh"]["provenance"] == "disk"
    assert payloads["indexed"]["metadata_source"] == "manifest"


def test_malformed_manifest_line_does_not_cost_its_neighbours(tmp_path):
    """A torn line (a crashed writer) must cost exactly the document it
    describes: the rows around it stay indexed instead of the whole corpus
    degrading to path-derived metadata."""
    root = _corpus(tmp_path, "us/C1/2015/a.pdf", "us/C1/2015/b.pdf")
    _legacy_manifest(
        root,
        '{"doc_id": "a", "date": "2015-06-30"}',
        '{"doc_id": "b", "date": "2015-0',  # torn tail
        '{"doc_id": "b", "date": "2015-07-31"}',
    )
    payloads = {i.doc_id: i.payload for i in iter_items(root)}
    assert payloads["a"]["publication_date"] == "2015-06-30"
    assert payloads["b"]["publication_date"] == "2015-07-31"


@pytest.mark.xfail(
    strict=True,
    reason="issue #5: _find_manifest only looks for the extinct single-file "
           "manifest.jsonl, so with cb_corpus's per-bank manifest/<bank>.jsonl "
           "layout every document silently degrades to path-derived metadata",
)
def test_per_bank_manifest_layout_is_read(tmp_path):
    """cb_corpus splits its manifest per bank (``manifest/<bank>.jsonl``); the
    connector must read that layout, or every real corpus document loses its
    exact publication date without any error being raised."""
    root = _corpus(tmp_path, "us/C1/2015/a.pdf")
    (root / "manifest").mkdir()
    (root / "manifest" / "us.jsonl").write_text(
        '{"doc_id": "a", "date": "2015-06-30"}\n', encoding="utf-8"
    )
    (item,) = iter_items(root)
    assert item.payload["metadata_source"] == "manifest"
    assert item.payload["publication_date"] == "2015-06-30"


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
