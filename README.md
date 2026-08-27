# rag-orchestrator

Installable Python package (repo: `RAGDataOrchestrator`). It is the **policy
layer** that sits between three systems, each owning one thing:

| system | owns | this repo... |
|---|---|---|
| **vault** (Postgres) | facts, document selection, ingestion state (`documents`, `rag_ingestions`) | reads the selection, writes the ingestion record |
| **eigenmind** (fork) | the RAG engine — chunking, embedding, Qdrant client | drives it as a library, never reimplements it |
| **Qdrant** | the derived vector projection | writes to it, treats it as rebuildable from vault + corpus |

> read whatever the vault says is new → drive it through eigenmind → land it in Qdrant

The orchestrator carries **no RAG logic of its own** — chunking, embedding and
the Qdrant client all live in [eigenmind](https://github.com/jeulinmarc/eigenmind),
consumed as a normal (editable) dependency. What this repo owns is orchestration
policy: which documents to pick up next, per-corpus path routing, resumability,
and the write protocol that keeps Qdrant and the vault consistent.

**The dependency is one-way.** This package *uses* eigenmind; eigenmind never
imports this one and keeps working entirely on its own. It also *uses* the
vault's Postgres schema (`documents`, `rag_ingestions`) but never owns or
migrates it — the vault's own ingestion service runs that DDL.

## Layout

```
RAGDataOrchestrator/
  pyproject.toml          # packaging — installs the `rag-orchestrator` CLI
  rag_orchestrator/
    __init__.py            # public API: SourceItem, run_ingest, Ledger, …
    core.py                # engine: chunk -> embed -> upsert + resume ledger
    routing.py              # per-corpus root/local_path routing + collection naming
    vault.py                 # Postgres connection + the rag_ingestions-backed ledger
    probe.py                 # facts probe (has_text_layer / page_count) for OCR policy
    cli.py                    # console entrypoint (`rag-orchestrator`)
    sources/
      vault.py                # source #1: vault-selected documents (any corpus)
      cb_corpus.py              # source #2: the central-bank PDF corpus (disk fallback)
  state/                   # resume ledgers when running with --no-vault (gitignored)
```

## Setup

1. **The eigenmind engine.** Clone [jeulinmarc/eigenmind](https://github.com/jeulinmarc/eigenmind)
   locally and install it editable — it is consumed at fork-HEAD, not from PyPI:

   ```bash
   pip install -e $EIGENMIND_PATH
   pip install -e .
   ```

2. **`.env`.** Copy `.env.example` to `.env` (git-ignored, per-machine) and set:

   | key | meaning |
   |---|---|
   | `EIGENMIND_PATH` | local clone of the eigenmind fork |
   | `DATABASE_URL` | vault Postgres, e.g. `postgresql://user:pass@host:5432/documents` |
   | `CB_CORPUS_ROOT` | local root of the `cb_corpus` corpus (folder containing `raw/`) — only needed for the `cb_corpus` fallback source |
   | `QDRANT_HOST` / `QDRANT_PORT` | optional, default to `localhost` / `6333` (read by eigenmind) |
   | `RAGO_EMBEDDING_MODEL` | optional, overrides the embedding model (see [Collections](#collections--embedding-model)) |

3. **Qdrant** reachable at `QDRANT_HOST:QDRANT_PORT`.

### Path resolution convention

The vault stores each document's `local_path` **relative to its corpus repo
root**, e.g. `data/raw/us/C1/2010/<doc_id>.pdf` for `cb_corpus`. A corpus's
`*_ROOT` env key, however, may point at a folder that already sits one level
inside that repo (for `cb_corpus`, `CB_CORPUS_ROOT` is documented as "the
folder that contains `raw/`", i.e. the repo's `data/` dir itself). Joining
`root / local_path` naively would then double-nest into `<root>/data/raw/...`
and find nothing.

`routing.resolve_local_path(corpus, local_path)` reconciles this: each
`CorpusRoute` in `rag_orchestrator/routing.py` carries an optional
`local_path_strip` — a **leading-prefix** stripped from `local_path` before
joining with the root. `central-bank` strips `"data/"` for exactly this
reason. Adding a corpus whose root already matches its vault `local_path`
layout needs no strip at all (the default is `""`).

## Usage

```bash
# Vault-selected ingestion (recommended): whatever the vault knows for a
# corpus that the target collection hasn't ingested yet.
rag-orchestrator vault --corpus central-bank

# Disk-fallback ingestion, bypassing the vault entirely (its own file ledger,
# no rag_ingestions read/write) — useful before a corpus is vault-onboarded.
rag-orchestrator cb_corpus --banks ecb --no-vault

# Facts probe: fills has_text_layer / page_count for a corpus's documents
# (feeds the OCR policy; safe to re-run, only unprobed rows are touched).
rag-orchestrator probe --corpus central-bank

# Force OCR handling explicitly instead of the engine's auto-detection.
rag-orchestrator vault --corpus central-bank --ocr always
```

> Equivalent module form: `python -m rag_orchestrator.cli vault --corpus central-bank`

### Useful flags

| flag | meaning |
|---|---|
| `--corpus NAME` | vault corpus to operate on (default `central-bank`) |
| `--no-vault` | use the local JSON-lines file ledger instead of the vault's `rag_ingestions` (no vault state is read or written); only valid with the `cb_corpus` source |
| `--ocr auto\|always\|never` | OCR fallback for scanned pages (default `auto`: defers to the engine's tesseract availability check) |
| `--collection NAME` | target Qdrant collection (default: `{corpus}-{model_tag}-v1` for `vault`, `cb_corpus` for the disk source) |
| `--source-codes a,b` | (vault source) only these `source_code` values, e.g. `ecb,fr` |
| `--doctypes C1,A3` | only these doc-type codes |
| `--languages en,fr` | (vault source) only these language codes |
| `--year-min` / `--year-max` | inclusive year bounds |
| `--banks a,b` | (cb_corpus source) only these bank codes |
| `--root PATH` | (cb_corpus source) corpus root override |
| `--limit N` | stop after N newly ingested docs |
| `--no-resume` | ignore the resume ledger, re-ingest everything |
| `--count-only` | (cb_corpus source) just count matching documents |

## The write protocol

Every document's Qdrant points are upserted **before** its `rag_ingestions`
record is written — never the other way round, and batching never spans
documents. That order makes the invariant **`rag_ingestions ⊆ Qdrant`** hold
at all times: if the process crashes between the two writes, the vault simply
under-claims that one document, and the next resume pass re-ingests it. Point
ids are deterministic (derived from `doc_id`, `page`, `chunk_number`), so a
re-ingested document's points overwrite in place instead of duplicating —
resume is a pure retry, not a special code path. A `reconcile` command
(drift audit + soft-deleted cleanup) is a specified future addition, not
built here.

## Collections & embedding model

Collections are named by the orchestrator, one per `(corpus, embedding
generation)`:

```
{corpus}-{model_tag}-v{n}
```

e.g. `central-bank-e5b-v1`. The embedding model defaults to
`intfloat/multilingual-e5-base` and is overridable via `RAGO_EMBEDDING_MODEL`
(used by CI to swap in a tiny model for integration tests). Bump `n` (or the
`EMBEDDING_VERSION` policy tag in `routing.py`) when chunking/embedding
*policy* changes, not just the model name.

## Qdrant payload

Every chunk's Qdrant point carries:

| field | source |
|---|---|
| `doc_id`, `corpus`, `source_code`, `doc_type`, `title`, `date`, `year`, `language`, `sha256`, `provenance` | vault (`documents`, merged with corpus-specific `extra`) |
| `filename`, `page`, `chunk_number`, `text` | pipeline (chunking output) |
| `ingestion_date` | pipeline — one ISO timestamp per document, shared by all of its points (eigenmind's date-range filter reads this) |

`chunk_number` (not `chunk_index`) matches what the eigenmind engine reads
everywhere (`vectordb/store.py`, `pipelines/rag.py`, `graph/singular.py`).

## Operational note: fresh database

On a fresh database, the **vault's own ingestion service must run at least
once before this orchestrator** — its DDL train is what creates the
`rag_ingestions` table this repo reads and writes. Running the orchestrator
against a database that has never seen the vault service fails at
`vault.connect()` / the first `rag_ingestions` query.

## Facts probe

`rag-orchestrator probe --corpus <corpus>` fills `has_text_layer` and
`page_count` for every document with `has_text_layer IS NULL`, feeding the
facts-driven OCR policy. It is the **only** writer of those two columns.
Only unprobed rows are selected, so the pass is resumable by construction and
converges to a no-op on re-run. Each document's `UPDATE` runs inside its own
`SAVEPOINT`, so one failing row (a corrupt PDF path, a constraint) can never
poison the whole batch's transaction — it is rolled back to the savepoint and
counted as an error, while every other document in the batch still commits.

## Testing

```bash
./venv/bin/python -m pytest -q                    # 46 unit tests
./venv/bin/python -m pytest -m integration -q      # 10 integration tests
```

Integration tests spin up throwaway Postgres and Qdrant containers via
`docker run` (skipped automatically if Docker is unavailable) and use a tiny
embedding model (`RAGO_EMBEDDING_MODEL=sentence-transformers/paraphrase-albert-small-v2`)
to keep CI fast. CI (`.github/workflows/tests.yml`) runs both suites on every
push/PR, checking out this repo and the public `jeulinmarc/eigenmind` fork
side by side.

## Adding a new corpus

Add one `CorpusRoute` entry to `ROUTING` in `rag_orchestrator/routing.py`
(root env key + optional `local_path_strip`) — no new connector code, as long
as the corpus is onboarded to the vault. The `vault` source works for any
corpus already in `documents`.

## Adding a new disk-fallback source

Create `rag_orchestrator/sources/<name>.py` exposing an `iter_items(...)`
generator that yields `core.SourceItem(doc_id, path, payload)`, then wire it
into `cli.py` (`source` choices + dispatch). The core engine needs no
changes.

## Dashboard

An optional Streamlit dashboard (`rag-dashboard`, `pip install -e ".[dashboard]"`)
reads corpus/RAG state live for the `cb_corpus` disk layout. It is out of
scope for the vault-driven ingestion chain described above and still reads
its own catalog internals directly.
