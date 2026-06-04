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
      cb_corpus.py        # source #1: the central-bank PDF corpus (cb_corpus project)
  state/                  # resume ledgers (gitignored)
```

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
| `--root PATH` | corpus root (folder containing `raw/`) |
| `--banks a,b` | only these bank codes |
| `--doctypes C1,A3` | only these doc-type codes |
| `--groups A,C` | only these doc groups |
| `--year-min` / `--year-max` | inclusive year bounds |
| `--collection NAME` | target Qdrant collection (default `cb_corpus`) |
| `--limit N` | stop after N newly ingested docs |
| `--ocr auto\|always\|never` | OCR fallback for scanned pages |
| `--include-html` | also ingest `.html` with no `.pdf` sibling |
| `--no-resume` | ignore the ledger, re-ingest everything |
| `--count-only` | just count matches |

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
