# rag-orchestrator

Installable Python package (repo: `RAGDataOrchestrator`) that feeds
**mvp-graph-rag**. Its single job:

> read data stored *somewhere* → move it into the vector DB.

It does **not** reimplement the RAG pipeline. It *drives* the existing
mvp-graph-rag functions — `load_pdf` → chunk → `embed` → Qdrant upsert — adding
per-source metadata, batching, progress and **resumability**. New data sources
are added as one module each under [`rag_orchestrator/sources/`](rag_orchestrator/sources);
the core engine stays untouched.

**The dependency is one-way.** This package *uses* mvp-graph-rag; mvp-graph-rag
never imports this one and keeps running entirely on its own (Streamlit,
`mini_rag.py`, etc.). The orchestrator is a pure add-on for bulk-ingesting many
documents from fixed sources.

This repo lives **next to** `mvp-graph-rag`:

```
CODES/
  mvp-graph-rag/        # the RAG pipeline (provides src/)
  RAGDataOrchestrator/  # this repo (package: rag_orchestrator)
```

mvp-graph-rag is a standalone app of flat scripts under its `src/`, **not** a
published package — so we don't pip-depend on it. Instead the pipeline `src/` is
located at runtime: a sibling `mvp-graph-rag/src`, or the `MVP_GRAPH_RAG_SRC`
environment variable if your layout differs.

## Layout

```
RAGDataOrchestrator/
  pyproject.toml          # packaging — installs the `rag-orchestrator` CLI
  rag_orchestrator/
    __init__.py           # public API: SourceItem, run_ingest, Ledger, …
    core.py               # engine: walk → load → chunk → embed → upsert + resume ledger
    cli.py                # console entrypoint (`rag-orchestrator`)
    sources/
      cb_corpus.py            # source #1: central-bank PDF corpus (cb_corpus project)
      bottom_up_corpus.py     # source #2: SEC EDGAR filings (bottom_up_corpus project)
  state/                  # resume ledgers (gitignored)
```

The corpus family has two layers feeding the same RAG:

- **Macro** — `cb_corpus`: central-bank documents.
- **Micro** — `bottom_up_corpus`: company filings from SEC EDGAR.

## Source #1 — `cb_corpus`

Reads the corpus produced by the [`cb_corpus`](https://github.com/jeulinmarc/cb_corpus)
project, laid out on disk as:

```
<root>/raw/<bank>/<doctype>/<year>/<doc_id>.pdf
```

`bank`, `doctype` and `year` are parsed from the path and attached to **every
chunk's payload**, so the vector DB can be filtered and cited per bank /
document-type / year. Default root points at the synced OneDrive copy
(override with `--root`).

Each chunk's Qdrant payload:

| field | example | source |
|-------|---------|--------|
| `source` | `cb_corpus` | connector |
| `bank_code` | `ecb` | path |
| `doc_type` | `C1` | path |
| `doc_type_label` | `Speech` | taxonomy |
| `doc_group` | `C` | path |
| `year` | `2019` | path |
| `doc_id` | `ecb/C1/2019/3c03….pdf` | path |
| `filename`, `page`, `chunk_index`, `text` | — | pipeline |

## Source #2 — `bottom_up_corpus`

The **micro** layer: company filings from SEC EDGAR, produced by the
[`bottom_up_corpus`](https://github.com/jeulinmarc/bottom_up_corpus) project. That
project discovers filings, downloads and decomposes the complete submission,
extracts clean text and renders each filing's primary document to a
human-readable, **page-anchored PDF** — which flows through the same mvp-graph-rag
PDF loader with no change. Cleaned text is the fallback when no PDF exists.

This connector is a thin shim over `bottom_up_corpus.rag.iter_items` — all
discovery/render logic stays in that project; nothing is vendored here. Each
chunk's Qdrant payload carries at least:

| field | example | source |
|-------|---------|--------|
| `source` | `bottom_up_corpus` | connector |
| `cik` | `320193` | bottom_up_corpus |
| `company` | `Apple Inc.` | bottom_up_corpus |
| `doc_type` | `A1` (10-K) | bottom_up_corpus |
| `year` | `2024` | bottom_up_corpus |
| `url` | `https://www.sec.gov/...` | bottom_up_corpus |
| `filename`, `page`, `chunk_index`, `text` | — | pipeline |

Default narrative scope is families **A** (10-K/10-Q/20-F) and **C** (proxy);
8-K/6-K and ownership forms are high-volume / low-narrative, so down-weight or
opt-in (see the family-weighting note in `bottom_up_corpus/docs/INGESTION_RAG.md`).

Install the dependency (it is not vendored) and point a root at the rendered
corpus:

```bash
pip install -e .[bottom_up]        # pulls bottom_up_corpus from GitHub
# then, with BOTTOM_UP_CORPUS_ROOT set in .env (or --root):
rag-orchestrator bottom_up_corpus --ciks 320193 --collection bottom_up_corpus
```

`bottom_up_corpus` can also be used from a local checkout (`pip install -e .` in
that repo) or simply put on `PYTHONPATH`.

## Prerequisites

- The sibling `mvp-graph-rag` repo present on disk (or `MVP_GRAPH_RAG_SRC`
  set): the orchestrator drives that pipeline's *code* at runtime.
- Qdrant up (`docker compose up -d` from the mvp-graph-rag repo).
- A Python env with the pipeline deps — see Install below.

## Install

**Recommended: a dedicated virtualenv** for the orchestrator, so its
dependencies stay isolated from mvp-graph-rag (mvp keeps living entirely on its
own). The package's `pyproject.toml` declares everything the pipeline needs, so
the venv is self-sufficient — only the mvp *source folder* has to be reachable
(not its venv).

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

pip install -e .            # adds the `rag-orchestrator` console script
# optional OCR extra (needs system tesseract + poppler):
pip install -e .[ocr]
```

> Alternatively, you can install into the existing mvp-graph-rag venv instead
> of creating a new one — it already has all the deps. The dedicated venv is
> just cleaner and keeps the two projects fully decoupled.

## Usage

Once installed, use the console script from anywhere:

```bash
# 1. Preview what matches a filter (no ingestion, no embedding)
rag-orchestrator cb_corpus --count-only --banks ecb

# 2. Small test subset: 20 ECB speeches into the 'cb_corpus' collection
rag-orchestrator cb_corpus --banks ecb --doctypes C1 --limit 20

# 3. A year-bounded slice
rag-orchestrator cb_corpus --year-min 2020 --year-max 2025

# 4. Everything (resumable — safe to stop and re-run)
rag-orchestrator cb_corpus
```

> Equivalent module form: `python -m rag_orchestrator.cli cb_corpus ...`

### Use as a library

```python
from rag_orchestrator import run_ingest, Ledger
from rag_orchestrator.sources import cb_corpus

items = cb_corpus.iter_items(banks=["ecb"], doctypes=["C1"])
stats = run_ingest(items, collection="cb_corpus", ledger=Ledger("state/cb_corpus.jsonl"))
print(stats.as_dict())
```

### Useful flags

| flag | meaning |
|------|---------|
| `--root PATH` | corpus root (the source's data folder) |
| `--doctypes C1,A3` | only these doc-type codes |
| `--year-min` / `--year-max` | inclusive year bounds |
| `--collection NAME` | target Qdrant collection (default: the source name) |
| `--limit N` | stop after N newly ingested docs |
| `--ocr auto\|always\|never` | OCR fallback for scanned pages |
| `--no-resume` | ignore the ledger, re-ingest everything |
| `--count-only` | just count matches |
| `--banks a,b` | *(cb_corpus)* only these bank codes |
| `--groups A,C` | *(cb_corpus)* only these doc groups |
| `--include-html` | *(cb_corpus)* also ingest `.html` with no `.pdf` sibling |
| `--ciks 320193,789019` | *(bottom_up_corpus)* only these SEC CIK numbers |
| `--prefer pdf\|text` | *(bottom_up_corpus)* rendered PDF (default) or cleaned text |

## Resume / idempotency

- Upsert is **idempotent**: a chunk's point id is `sha1(doc_id::page::chunk_index)`,
  so re-ingesting overwrites in place — never duplicates.
- A JSON-lines **ledger** (`state/<collection>.jsonl`) records each ingested
  `doc_id`, so re-runs skip already-done documents and only process new ones.
  Delete the ledger (or pass `--no-resume`) to force a full re-ingest.

## Querying after ingestion

The corpus lands in its own collection (`cb_corpus`). To query it with the
existing mvp-graph-rag tooling, point the retriever at that collection (the
demo CLI defaults to `documents`).

## Adding a new source

Create `rag_orchestrator/sources/<name>.py` exposing a generator that yields
`core.SourceItem(doc_id, path, payload)`, then wire it into `cli.py`
(`source` choices + dispatch). The core engine needs no changes.
