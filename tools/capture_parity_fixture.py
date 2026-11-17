"""Capture the parity fixture of ``tests/test_engine_parity.py`` from the previous pipeline.

Purpose: provenance. ``tests/fixtures/engine_parity/expected.json`` holds the Qdrant points
(id, payload, vector) that this repository's ``core.ingest_item`` wrote for the documents of
``tests/fixtures/engine_parity/corpus/`` while it still drove the engine fork. The parity
test asserts that ``ingest_item``, now delegating to ``rag_engine``, writes the same points,
which is what keeps the points already stored valid. This script is how the file was made;
the README next to the fixture records the command, the versions and the commits.

It is not part of the package and is not run in CI. It only works on a checkout of this
repository from BEFORE the switch (``7196bfc``), with the fork on ``PYTHONPATH``, in an
environment that has ChunkNorris and ``rag_engine`` (for its ``FakeEmbedder`` only)::

    PYTHONPATH=<fork checkout> python tools/capture_parity_fixture.py --force

``--make-documents`` writes the documents first. It was used once: PyMuPDF does not write
the same bytes twice, so the documents are committed and never generated again (an existing
document is never overwritten). An existing ``expected.json`` is not overwritten unless
``--force`` is given.

Run it again only to capture from a known-good pipeline; never to make a failing test pass.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
import zipfile
from importlib.metadata import version
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FIXTURE_DIR = REPO / "tests" / "fixtures" / "engine_parity"
CORPUS_DIR = FIXTURE_DIR / "corpus"
EXPECTED = FIXTURE_DIR / "expected.json"

CORPUS = "central-bank"
COLLECTION = "central-bank-e5b-v1"
# The fake embedder of the capture and of the test: a small dimension keeps the file small.
MODEL = {"name": "parity-fake", "dim": 8, "passage_prefix": "passage: ", "query_prefix": "query: "}
SEED = 0
# "never" rather than the default "auto": what "auto" does depends on the host (ENABLE_OCR,
# TESSDATA_PREFIX), and the fixture must not.
OCR = "never"

# Fabricated vault rows, as ``sources.vault.iter_items`` reads them from ``documents``
# (``_SELECT_COLUMNS``). ``date`` is a ``datetime.date`` in the real rows: it is stored here
# as text and converted by ``vault_row``. ``sha256`` is filled in from the file. The rows
# differ on purpose: columns left NULL, an ``extra`` with keys outside the columns, an
# ``extra`` that repeats a column (the column wins), and one ``extra`` holding keys the
# engine writes itself (``filename``, ``text``, ``chunk_number``, ``ingestion_date``), which
# the previous pipeline silently overwrote.
ROWS = (
    {
        "doc_id": "ecb-a1-2025-10-30-statement",
        "corpus": CORPUS, "source_code": "ecb", "doc_type": "A1",
        "title": "Décision de politique monétaire", "date": "2025-10-30", "year": 2025,
        "language": "fr", "provenance": "manifest",
        "local_path": "data/raw/ecb/A1/2025/ecb-a1-2025-10-30-statement.md",
        "extra": {"doc_group": "A", "pdf_url": "https://example.org/ecb/2025-10-30.html"},
    },
    {
        "doc_id": "us-a3-2024-06-12-minutes",
        "corpus": CORPUS, "source_code": "us", "doc_type": "A3",
        "title": "Minutes of the Federal Open Market Committee", "date": "2024-06-12",
        "year": 2024, "language": "en", "provenance": "manifest",
        "local_path": "data/raw/us/A3/2024/us-a3-2024-06-12-minutes.txt",
        "extra": {
            "filename": "stale-name.txt", "text": "stale text", "chunk_number": 99,
            "ingestion_date": "1999-01-01T00:00:00", "meeting": "June 2024",
        },
    },
    {
        "doc_id": "jp-f1-2023-rates",
        "corpus": CORPUS, "source_code": "jp", "doc_type": "F1",
        "title": None, "date": None, "year": 2023, "language": None, "provenance": "disk",
        "local_path": "data/raw/jp/F1/2023/jp-f1-2023-rates.csv",
        "extra": None,
    },
    {
        "doc_id": "gb-e1-2024-08-01-report",
        "corpus": CORPUS, "source_code": "gb", "doc_type": "E1",
        "title": "Monetary Policy Report, August 2024 (extract)", "date": "2024-08-01",
        "year": 2024, "language": "en", "provenance": "manifest",
        "local_path": "data/raw/gb/E1/2024/gb-e1-2024-08-01-report.pdf",
        "extra": {"series": "Monetary Policy Report", "year": 1900, "doc_id": "stale-id"},
    },
    {
        "doc_id": "au-e1-2022-02-statement",
        "corpus": CORPUS, "source_code": "au", "doc_type": "E1",
        "title": "Statement on Monetary Policy", "date": "2022-02-01", "year": 2022,
        "language": "en", "provenance": "manifest",
        "local_path": "data/raw/au/E1/2022/au-e1-2022-02-statement.pdf",
        "extra": {},
    },
    {
        "doc_id": "ca-a3-2023-01-25-deliberations",
        "corpus": CORPUS, "source_code": None, "doc_type": "A3",
        "title": "Summary of Governing Council deliberations", "date": "2023-01-25",
        "year": 2023, "language": "en", "provenance": "manifest",
        "local_path": "data/raw/ca/A3/2023/ca-a3-2023-01-25-deliberations.docx  ",
        "extra": {"date_granularity": "source"},
    },
)

MARKDOWN = """# Décision de politique monétaire

## Taux directeurs

Le Conseil des gouverneurs a décidé aujourd'hui de laisser inchangés les trois taux d'intérêt
directeurs de la BCE. L'inflation se situe actuellement à un niveau proche de l'objectif de
2 % à moyen terme et l'évaluation des perspectives d'inflation reste globalement inchangée.

## Economic activity

The economy has continued to grow despite the challenging global environment. The robust
labour market, solid private sector balance sheets and the past interest rate cuts remain
important sources of resilience, while the outlook is still uncertain owing to trade disputes
and geopolitical tensions.

## Programme d'achats

Les portefeuilles du programme d'achats d'actifs et du programme d'achats d'urgence face à la
pandémie se contractent à un rythme mesuré et prévisible, l'Eurosystème ne réinvestissant plus
les remboursements au titre du principal des titres arrivant à échéance.
""" + "".join(
    f"""
## Annexe {n} : évaluation des risques

Section {n}. Les risques pesant sur la croissance économique restent orientés à la baisse. Une
nouvelle escalade des tensions commerciales mondiales et les incertitudes qui y sont associées
pourraient freiner la croissance en pesant sur les exportations, l'investissement et la
consommation. Une détérioration du sentiment sur les marchés financiers pourrait entraîner un
durcissement des conditions de financement et une plus grande aversion au risque.

The outlook for inflation is more uncertain than usual, on account of the still volatile
global trade policy environment. A stronger euro could bring inflation down further than
expected, while higher defence and infrastructure spending could raise it over the medium
term. Extreme weather events could drive up food prices by more than expected in annex {n}.
"""
    for n in range(1, 5)
)

TEXT = """Minutes of the Federal Open Market Committee

Participants observed that inflation had eased over the past year but remained elevated. They
noted that recent monthly data had shown modest further progress toward the two per cent
objective, and that the labour market had come into better balance.

In their discussion of monetary policy, all participants judged it appropriate to maintain the
target range for the federal funds rate. Participants emphasised that they did not expect it
would be appropriate to lower the target range until additional information had emerged that
gave them greater confidence that inflation was moving sustainably toward two per cent.

Several participants remarked that the staff outlook was little changed, with real output
growth projected to slow, and unemployment to edge up over the coming quarters.
""" + "".join(
    f"""
Paragraph {n} of the staff review. The staff provided an update on its assessment of the
stability of the financial system. Asset valuation pressures were judged to be notable, house
prices remained high relative to rents, and corporate bond spreads were narrow by historical
standards. Vulnerabilities associated with business and household debt were characterised as
moderate, while leverage in the financial sector was described as notable, in part because of
the sensitivity of some banks to changes in longer-term interest rates noted in review {n}.
"""
    for n in range(1, 7)
)

CSV = (
    "date,policy_rate,decision,comment\n"
    "2023-01-18,-0.10,hold,Yield curve control maintained with a wider band\n"
    "2023-04-28,-0.10,hold,Review of past monetary policy measures announced\n"
    "2023-07-28,-0.10,hold,Greater flexibility in the conduct of yield curve control\n"
    "2023-10-31,-0.10,hold,Upper bound of one per cent becomes a reference\n"
    "2023-12-19,-0.10,hold,No change while wage developments are assessed\n"
    + "".join(
        f"2024-{month:02d}-{day:02d},{0.05 * month:.2f},review,"
        f"Scheduled review number {month * 10 + day} of market operations and collateral terms\n"
        for month in range(1, 13) for day in (5, 12, 19, 26)
    )
)

REPORT_PAGES = (
    ("Monetary Policy Report", (
        "The Monetary Policy Committee sets monetary policy to meet the two per cent inflation "
        "target, and in a way that helps to sustain growth and employment. At its meeting ending "
        "on 31 July, the Committee voted by a majority of five to four to reduce Bank Rate by a "
        "quarter of a percentage point, to five per cent.",
        "Twelve-month consumer price inflation was at the target in both May and June. The "
        "Committee expects the fall in headline inflation to continue to feed through to weaker "
        "pay and price-setting dynamics over the coming quarters.",
    )),
    ("Demand and output", (
        "A margin of slack is projected to emerge in the economy as activity falls below "
        "potential and the labour market eases further. Unemployment is expected to rise "
        "gradually over the forecast period.",
    )),
    ("Costs and prices", (
        "Services price inflation has remained elevated, at 5.7 per cent in June. Private sector "
        "regular average weekly earnings growth has fallen to 5.6 per cent in the three months to "
        "May, and is expected to ease further.",
        "Conditional on market interest rates, inflation is projected to fall back to 1.7 per "
        "cent in two years' time and to 1.5 per cent in three years, reflecting the continued "
        "restrictiveness of the policy stance.",
    )),
)

# Second PDF: the middle page is blank (no text layer on it).
STATEMENT_PAGES = (
    ("Statement on Monetary Policy", (
        "The Australian economy has been resilient to the Omicron outbreak. Household spending "
        "is expected to recover strongly, supported by a high household saving rate, a tight "
        "labour market and the easing of restrictions.",
        "Inflation has picked up more quickly than the Bank had expected, but remains lower than "
        "in many other countries. Underlying inflation has reached the middle of the target band "
        "for the first time in more than seven years.",
    )),
    None,
    ("Outlook", (
        "The central forecast is for gross domestic product to grow by around four and a quarter "
        "per cent over 2022 and two per cent over 2023. The unemployment rate is expected to "
        "decline to below four per cent later in the year and to remain there.",
        "Wages growth has picked up but, at the aggregate level, has only returned to the "
        "relatively low rates prevailing before the pandemic. A further pick-up in wages growth "
        "is expected as the labour market tightens, although this is likely to be gradual.",
        "The Board is prepared to be patient as it monitors how the various factors affecting "
        "inflation in Australia evolve. It will not increase the cash rate until actual "
        "inflation is sustainably within the two to three per cent target range.",
    )),
    ("Financial conditions", (
        "Financial conditions remain highly accommodative, with lending rates for households and "
        "businesses at record lows. Housing credit growth has been strong, although the value "
        "of new housing loan commitments has stabilised over recent months at a high level.",
        "The Australian dollar exchange rate is around its lowest level in a year and a half, "
        "while bond yields have risen alongside the increase in yields in other economies.",
    )),
)

DOCX_PARAGRAPHS = (
    ("Heading1", "Summary of Governing Council deliberations"),
    (None, "Governing Council members discussed the evolution of the global economy. They noted "
           "that growth had slowed in most regions as higher interest rates weighed on demand, "
           "while supply chains had continued to improve and energy prices had declined."),
    ("Heading2", "Considerations for monetary policy"),
    (None, "Members agreed that the policy rate needed to rise further and debated whether to "
           "signal a pause. They concluded that a conditional pause would give time to assess "
           "whether monetary policy was sufficiently restrictive to return inflation to target."),
    (None, "The Council also agreed that quantitative tightening was complementing the increases "
           "in the policy rate and that it should continue as planned."),
) + tuple(
    entry
    for n in range(1, 5)
    for entry in (
        ("Heading2", f"Domestic conditions, part {n}"),
        (None, f"In part {n} of the discussion, members reviewed domestic data. The economy "
               "remained in excess demand and labour markets were tight, with the unemployment "
               "rate near its record low. Growth in household spending was slowing, particularly "
               "for housing and big-ticket items that are sensitive to interest rates, and the "
               "effects of past increases were expected to keep working through the economy."),
        (None, "Members observed that three-month measures of core inflation had come down, "
               "suggesting that core inflation had peaked. They discussed the risk that services "
               f"price inflation and wage growth could remain persistent (item {n})."),
    )
)


def _make_pdf(target: Path, pages) -> None:
    import pymupdf

    document = pymupdf.open()
    for content in pages:
        page = document.new_page(width=595, height=842)
        if content is None:
            continue
        title, paragraphs = content
        page.insert_text((72, 96), title, fontsize=18, fontname="hebo")
        top = 130
        for paragraph in paragraphs:
            rect = pymupdf.Rect(72, top, 523, top + 110)
            if page.insert_textbox(rect, paragraph, fontsize=11, fontname="helv") < 0:
                raise RuntimeError(f"paragraph does not fit its box on page {title!r}")
            top += 120
    document.save(target, garbage=4, deflate=True)
    document.close()


def _make_docx(target: Path) -> None:
    """A minimal Word file written by hand (no python-docx needed): styled paragraphs only."""
    from xml.sax.saxutils import escape

    body = []
    for style, text in DOCX_PARAGRAPHS:
        props = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
        body.append(f'<w:p>{props}<w:r><w:t xml:space="preserve">{escape(text)}</w:t></w:r></w:p>')
    ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:document xmlns:w="{ns}"><w:body>{"".join(body)}</w:body></w:document>'
    )
    styles = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:styles xmlns:w="{ns}">'
        '<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/></w:style>'
        '<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/></w:style>'
        '</w:styles>'
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-'
        'officedocument.wordprocessingml.document.main+xml"/>'
        '<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-'
        'officedocument.wordprocessingml.styles+xml"/>'
        '</Types>'
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/'
        'relationships/officeDocument" Target="word/document.xml"/>'
        '</Relationships>'
    )
    doc_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/'
        'relationships/styles" Target="styles.xml"/>'
        '</Relationships>'
    )
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", rels)
        archive.writestr("word/document.xml", document)
        archive.writestr("word/styles.xml", styles)
        archive.writestr("word/_rels/document.xml.rels", doc_rels)


def document_path(row: dict) -> Path:
    """Where the document of ``row`` is in the fixture corpus (``data/`` is the stripped prefix)."""
    return CORPUS_DIR / row["local_path"].strip()[len("data/"):]


def make_documents() -> None:
    writers = {
        ".md": lambda p: p.write_text(MARKDOWN, encoding="utf-8"),
        ".txt": lambda p: p.write_text(TEXT, encoding="utf-8"),
        ".csv": lambda p: p.write_text(CSV, encoding="utf-8"),
        ".docx": _make_docx,
    }
    pdf_pages = {"gb-e1-2024-08-01-report": REPORT_PAGES, "au-e1-2022-02-statement": STATEMENT_PAGES}
    for row in ROWS:
        path = document_path(row)
        if path.exists():
            print(f"kept     {path.relative_to(REPO)}")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".pdf":
            _make_pdf(path, pdf_pages[row["doc_id"]])
        else:
            writers[path.suffix](path)
        print(f"written  {path.relative_to(REPO)}")


def vault_row(row: dict) -> dict:
    """``row`` as the vault hands it over: ``date`` a ``datetime.date``, ``sha256`` of the file."""
    out = dict(row)
    if out.get("date"):
        out["date"] = datetime.date.fromisoformat(out["date"])
    out["sha256"] = hashlib.sha256(document_path(row).read_bytes()).hexdigest()
    return out


class RecordingClient:
    """The part of a Qdrant client the previous ``ingest_item`` uses; keeps what it is asked to write."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []
        self.points: list = []

    def upsert(self, collection_name, points, **_ignored) -> None:
        self.calls.append((collection_name, len(points)))
        self.points.extend(points)


class RecordingStore:
    def __init__(self) -> None:
        self.client = RecordingClient()


def _git_head(path: Path) -> str:
    out = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True)
    return out.stdout.strip() or "unknown"


def capture() -> dict:
    os.environ["CB_CORPUS_ROOT"] = str(CORPUS_DIR)
    # The previous pipeline, imported as it is: nothing of it is replaced.
    import eigenmind
    from data_orchestrator.core import ingest_item
    from data_orchestrator.sources.vault import row_to_item

    from rag_engine.embedding import FakeEmbedder, ModelSpec

    documents = []
    for row in ROWS:
        item = row_to_item(vault_row(row), CORPUS)
        store = RecordingStore()
        written = ingest_item(
            item, COLLECTION, store=store,
            embedder=FakeEmbedder(ModelSpec(**MODEL), seed=SEED), ocr=OCR,
        )
        points = store.client.points
        assert written == len(points), (written, len(points))
        assert all(name == COLLECTION for name, _ in store.client.calls)
        documents.append({
            "row": row,
            "written": written,
            "points": [{"id": p.id, "payload": p.payload, "vector": p.vector} for p in points],
        })
    return {
        "captured_from": {
            "data_orchestrator": _git_head(REPO),
            "engine_fork": _git_head(Path(eigenmind.__file__).resolve().parent.parent),
            "python": platform.python_version(),
            "chunknorris": version("chunknorris"),
            "pymupdf": version("pymupdf"),
            "qdrant_client": version("qdrant-client"),
            "rag_engine_fake_embedder": version("rag-engine"),
        },
        "corpus": CORPUS,
        "collection": COLLECTION,
        "embedder": {**MODEL, "seed": SEED},
        "ocr": OCR,
        "documents": documents,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--make-documents", action="store_true")
    parser.add_argument("--force", action="store_true", help="overwrite an existing expected.json")
    parser.add_argument("--output", type=Path, default=EXPECTED)
    args = parser.parse_args()
    if args.make_documents:
        make_documents()
    if args.output.exists() and not args.force:
        print(f"error: {args.output} exists (use --force, or --output elsewhere)", file=sys.stderr)
        return 1
    result = capture()
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for document in result["documents"]:
        print(f"{document['row']['doc_id']}: {document['written']} point(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
