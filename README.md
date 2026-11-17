# data-orchestrator

Installable Python package (repo: `MyOpenFund/data-orchestrator`). It is the **policy
layer** that sits between three systems, each owning one thing:

| system | owns | this repo... |
|---|---|---|
| **vault** (Postgres) | facts, document selection, ingestion state (`documents`, `rag_ingestions`) | reads the selection, writes the ingestion record |
| **rag-engine** (library) | the RAG engine — chunking, embedding, point ids and payloads, Qdrant store | calls it once per document, never reimplements it |
| **Qdrant** | the derived vector projection | writes to it, treats it as rebuildable from vault + corpus |

> read whatever the vault says is new → hand it to rag-engine → land it in Qdrant

The orchestrator carries **no RAG logic of its own** — chunking, embedding,
the point ids and payloads, the collection checks and the Qdrant client all
live in [rag-engine](https://github.com/MyOpenFund/rag-engine), a normal
dependency pinned by commit in `pyproject.toml`. What this repo owns is orchestration
policy: which documents to pick up next, per-corpus path routing, resumability,
and the write protocol that keeps Qdrant and the vault consistent.

Until 2026-11 the engine was a fork (`MyOpenFund/eigenmind`) installed from a
separate checkout; an installation from that time follows
[MIGRATION.md](MIGRATION.md).

**The dependency is one-way.** This package *uses* rag-engine; rag-engine never
imports this one. It also *uses* the
vault's Postgres schema (`documents`, `rag_ingestions`) but never owns or
migrates it — the vault's own ingestion service runs that DDL.

## Layout

```
data-orchestrator/
  pyproject.toml          # packaging — installs the `data-orchestrator` CLI
  data_orchestrator/
    __init__.py            # public API: SourceItem, run_ingest, Ledger, …
    config.py              # the one .env loader (see "Configuration loading")
    core.py                # the run loop around the engine: settings, resume ledger, isolation
    policy.py              # run outcome rule (ok / degraded) + failure-ratio env
    routing.py              # per-corpus routing, collection naming, supported types
    vault.py                 # Postgres connection + the rag_ingestions-backed ledger
    probe.py                 # facts probe (has_text_layer / page_count) for OCR policy
    cli.py                    # console entrypoint (`data-orchestrator`)
    sources/
      vault.py                # source #1: vault-selected documents (any corpus)
      cb_corpus.py              # source #2: the central-bank PDF corpus (disk fallback)
      company_corpus.py         # source #3: SEC EDGAR company filings (disk fallback)
  state/                   # resume ledgers when running with --no-vault (gitignored)
```

The corpus family has two disk layers feeding the same RAG, each selectable
through the `vault` source once onboarded there, and both usable disk-only
via their own connector (`--no-vault`) in the meantime:

- **Macro** — `cb_corpus` (vault corpus `central-bank`): central-bank documents.
- **Micro** — `company_corpus` (vault corpus `company`): company filings from SEC EDGAR.

## Setup

1. **Install.** One command brings the engine with it (rag-engine with its
   parsers and its PyTorch backend, pinned by commit in `pyproject.toml`);
   there is no separate engine checkout. This repo must be an **editable**
   install for the repo-root `.env` fallback below to find its `.env`:

   ```bash
   pip install -e .
   ```

   - **Apple Silicon: install normally**, as above. The default PyTorch wheel
     uses the Metal backend (`mps`); check the log line of the first run,
     `Loading embedding model intfloat/multilingual-e5-base on mps`.
   - **Linux without an NVIDIA GPU (a NAS, a server), and CI**: install the
     CPU build of PyTorch first,

     ```bash
     pip install torch --index-url https://download.pytorch.org/whl/cpu
     pip install -e .
     ```

     Skipping that step on Linux pulls the CUDA packages with the default
     wheel (several gigabytes) for nothing.
   - **An installation that already holds ingested collections**: the
     versions of chunknorris, PyMuPDF, sentence-transformers, torch and
     transformers decide the text and the vectors of every point. Save
     `pip freeze` before changing the environment and see
     [MIGRATION.md](MIGRATION.md) ("Validating on real data after the
     switch") before relying on a re-run overwriting points in place.

   OCR of scanned pages needs Tesseract's language data on the host
   (`apt-get install tesseract-ocr`, `brew install tesseract`) and the two
   switches of the table below.

2. **`.env`.** Copy `.env.example` to `.env` (git-ignored, per-machine) and set:

   | key | meaning |
   |---|---|
   | `DATABASE_URL` | vault Postgres, e.g. `postgresql://user:pass@host:5432/documents` |
   | `CB_CORPUS_ROOT` | local root of the central-bank corpus (folder containing `raw/`) — required whenever the central-bank corpus is used (vault source, probe, or cb_corpus fallback) |
   | `COMPANY_CORPUS_ROOT` | local root of the company_corpus (SEC EDGAR) data dir — optional, only needed for the `company` disk fallback; unset falls back to company_corpus's own default data dir |
   | `QDRANT_HOST` / `QDRANT_PORT` | the Qdrant the engine writes to (read by rag-engine). **Always set both**: unset, the engine falls back to `localhost:6333`, the **legacy** store; every ingest run prints `Qdrant: <host>:<port> (api key: set|unset)` (the engine's store, as constructed) and warns on stderr when the port is 6333. A set `QDRANT_PORT` must be a plain integer in `1..65535` (surrounding whitespace ignored): empty, signed, decimal or followed by an inline `# comment` is an error on every ingest run (exit 2, before any work) |
   | `QDRANT_API_KEY` | API key of that Qdrant (read by rag-engine, sent as the `api-key` header; blank or unset = none). The start line says whether one is set, never its value. A 401/403 from Qdrant ends the run with **one** fatal error, at collection setup (before any document) or on a write in the middle of a run (the documents already recorded stay recorded, the one being written is not): `error: fatal: QdrantAuthError: Qdrant refused the request (HTTP 401): check QDRANT_API_KEY`, exit 1, same text in `runs.extra->>'fatal'`; the run report still counts the documents ingested before the refusal (see [Run reports](#run-reports-and-exit-codes)). See `QDRANT_HTTPS` for the plain-HTTP warning |
   | `QDRANT_HTTPS` | `true` only if that Qdrant is served over TLS (read by rag-engine; default `false`, plain HTTP). With a key over plain HTTP `qdrant-client` warns `Api key is used with an insecure connection`: expected inside the tailnet |
   | `RAG_ENGINE_QDRANT_TIMEOUT` | optional, seconds a request to Qdrant may take (default `60`) |
   | `ENABLE_OCR` / `TESSDATA_PREFIX` | the host's OCR switches, read by the engine **from the process environment each time a file is chunked**: with `--ocr auto` (the default) scanned pages are read by OCR only when `ENABLE_OCR=true` **and** `TESSDATA_PREFIX` is set (the folder of Tesseract's `*.traineddata` files, or its parent). Put both in this `.env` (or the real environment): left out, `auto` silently means `never`. Every ingest run says which it is: `OCR: available (languages: eng)` or `OCR: off (ENABLE_OCR not true or TESSDATA_PREFIX unset)`. Neither has a default (unset = off). `--ocr always` does not look at `ENABLE_OCR` but still needs `TESSDATA_PREFIX`: without it the run is refused (exit 2, before any work), since OCR would fail on every PDF |
   | `RAG_ENGINE_OCR_LANGUAGES` | optional, Tesseract languages used by OCR, codes joined by `+` (default `eng`; e.g. `eng+fra`). The engine uses exactly this list; it does not look at which language files are installed |
   | `DATA_ORCHESTRATOR_EMBEDDING_MODEL` | optional, overrides the embedding model (default `intfloat/multilingual-e5-base`; see [Collections](#collections--embedding-model)); `RAG_ENGINE_MODEL`, the engine's own name for it, is read when this one is unset |
   | `DATA_ORCHESTRATOR_EMBEDDING_DIM` | vector dimension of that model. Not needed for the default model (768); **required for any other model** (`RAG_ENGINE_DIM` is read when this one is unset), else exit 2. A dimension that is not the loaded model's is a fatal error before any document |
   | `RAG_ENGINE_BATCH_SIZE` | optional, points per write to Qdrant (default `100`) |
   | `RAG_ENGINE_BACKEND` | leave unset (default `torch`): ingestion uses the torch backend, the one this package installs. `onnx`, the engine's backend for search services, is not installed here and is refused on an ingest run (`error: RAG_ENGINE_BACKEND='onnx': ingestion uses the torch backend, …`, exit 2, before any work), like any other value |
| `RAG_ENGINE_SEARCH_LIMIT` / `RAG_ENGINE_SEARCH_MAX_LIMIT` / `RAG_ENGINE_PARITY_MIN` / `RAG_ENGINE_ONNX_THREADS` | leave unset (defaults `10`, `50`, `0.999`, none). Settings of the engine's search side: ingestion does not use them, but the engine reads all of its settings at once, so an invalid value (not a number, a limit below 1, a parity outside `(0, 1]`) stops an ingest run with exit 2 |
   | `DATA_ORCHESTRATOR_DOTENV` | optional (default: unset, the `.env` is searched as described below), **real environment only** (never read from a `.env`): explicit path of the `.env` to load instead of searching (a relative path is resolved against the current directory; the absolute path is what is read and printed); a missing file is an error (exit 2 from the CLI, `FileNotFoundError` for a library caller) |
   | `DATA_ORCHESTRATOR_LOG_LEVEL` | optional, level of the log lines the CLI sends to stderr (this package's and the engine's: model loaded, collection created, a text file with invalid bytes, …): `DEBUG`, `INFO` (default), `WARNING`, `ERROR`, `CRITICAL`; anything else is an error (exit 2) |
   | `DATA_ORCHESTRATOR_STATE_DIR` | optional, where the `--no-vault` file ledger and run reports are written; defaults to `<repo>/state` (`~` is expanded) |
   | `DATA_ORCHESTRATOR_SUPPORTED_MIME_TYPES` | optional comma list narrowing the types the vault selection offers to the engine (default: every type the pinned engine reads; see [Supported document types](#supported-document-types)); a type the engine cannot read is an error (exit 2) |
   | `DATA_ORCHESTRATOR_DEGRADED_FAILURE_RATIO` | optional plain decimal in `[0, 1]`, default `0.10` (see [Run reports](#run-reports-and-exit-codes)); anything else is an error (exit 2) |

3. **Qdrant** reachable at `QDRANT_HOST:QDRANT_PORT`.

### Configuration loading

- **One loader, first.** `data-orchestrator` (`cli.main`) loads the `.env`
  right after parsing its arguments, before anything reads the environment.
  It is the only loader: rag-engine loads no `.env` of its own and reads
  nothing at import. It reads the process environment when asked — its
  settings once per run, right after this load (an invalid one, e.g.
  `RAG_ENGINE_BATCH_SIZE=0`, is an `error:` and exit 2 before any work), and
  `ENABLE_OCR` / `TESSDATA_PREFIX` each time a file is chunked.
- **Which file.** `DATA_ORCHESTRATOR_DOTENV` if set — read from the real
  environment only, never from a `.env`, a relative path resolved against the
  current directory; a missing file is an `error:` and exit 2 before any work
  — else the nearest `.env` in the current directory or
  its parents, else the `.env` at this repo's root (found next to the
  installed package, hence the editable install).
- **Traceability.** Every run prints `Config: .env = <absolute path>` (or
  `Config: no .env file found`) first. Every ingest run (`vault`, `cb_corpus`,
  `company`, not `--count-only`) then prints the OCR line (`OCR: available
  (languages: eng)`, `OCR: off (ENABLE_OCR not true or TESSDATA_PREFIX
  unset)`, or `OCR: off (--ocr never)` / `OCR: forced (--ocr always,
  languages: eng)` for an explicit mode; `--ocr always` without
  `TESSDATA_PREFIX` is refused before this line, exit 2) and `Qdrant: <host>:<port> (api key:
  set|unset)` once the engine's store is constructed — read from that store —
  with a stderr `warning:` on port 6333 (the legacy store; compared
  numerically, `06333` is 6333). A `QDRANT_PORT` that is set but is not a
  plain integer in `1..65535` (empty included) stops an ingest run with an
  `error:` and exit 2 before any work,
  rather than let the engine fail mid-run. `--count-only` and `probe` never
  touch Qdrant or the engine, do not check `QDRANT_PORT` or the engine's
  settings and print no `Qdrant:` or `OCR:` line.
- **Literal values.** No `$VAR` expansion, surrounding quotes stripped, and
  **no inline comments**: a `#` after a value is part of the value. A key
  already set in the real environment always wins over the file.
- **Library callers** (`run_ingest` & co. without the CLI) get no loader
  *first*: `vault.connect()` and `routing.corpus_root()` call
  `config.load_dotenv()` lazily (raising `FileNotFoundError` when
  `DATA_ORCHESTRATOR_DOTENV` names a missing file), while `run_ingest` reads
  the engine's settings from the process environment as it starts. Call
  `data_orchestrator.config.load_dotenv()` yourself before `run_ingest` (or
  pass it `settings=`, `store=` and `embedder=`).

## Disk sources

Both disk sources below are selectable through the `vault` source once their
corpus is onboarded there (recommended — see [Usage](#usage)); each also has
its own disk-fallback connector (`--no-vault`) for use before that onboarding,
or standalone. Neither hard-codes a collection name: the default is always the
routed `{corpus}-{model_tag}-v1` (`central-bank-e5b-v1` / `company-e5b-v1`),
via `routing.collection_name`.

### `cb_corpus` — the central-bank corpus

Reads the corpus produced by the
[`central-bank-corpus`](https://github.com/MyOpenFund/central-bank-corpus)
project, laid out on disk as:

```
<root>/raw/<bank>/<doctype>/<year>/<doc_id>.pdf
<root>/manifest/<bank>.jsonl
```

`<root>` is the folder that contains `raw/`. The per-bank manifest files
supply exact publication dates, titles and checksums; a root without any
`manifest/*.jsonl` is rejected outright (mis-pointed `CB_CORPUS_ROOT`),
while a document not yet indexed is ingested with path-derived metadata
(`<year>-01-01`) and counted — the run summary warns and the run report
carries `totals.docs_path_metadata`.

`bank`, `doctype` and `year` are parsed from the path and attached to **every
chunk's payload**, so the vector DB can be filtered and cited per bank /
document-type / year.

| field | example | source |
|-------|---------|--------|
| `source` | `cb_corpus` | connector |
| `bank_code` | `ecb` | path |
| `doc_type` | `C1` | path |
| `doc_group` | `C` | path |
| `year` | `2019` | path |
| `doc_id` | `ecb/C1/2019/3c03….pdf` | path |
| `filename`, `page`, `chunk_number`, `ingestion_date`, `text` | — | engine |

### `company` — the company (SEC EDGAR) corpus

The **micro** layer: company filings from SEC EDGAR, produced by the
[`company-corpus`](https://github.com/MyOpenFund/company-corpus) project. That
project discovers filings, downloads and decomposes the complete submission,
extracts clean text and renders each filing's primary document to a
human-readable, **page-anchored PDF** — which flows through the same engine
PDF parser with no change. Cleaned text is the fallback when no PDF exists.

This connector is a thin shim over `company_corpus.rag.iter_items` — all
discovery/render logic stays in that project; nothing is vendored here. Each
chunk's Qdrant payload carries at least:

| field | example | source |
|-------|---------|--------|
| `source` | `company-corpus` | connector |
| `cik` | `320193` | company_corpus |
| `company` | `Apple Inc.` | company_corpus |
| `doc_type` | `A1` (10-K) | company_corpus |
| `year` | `2024` | company_corpus |
| `url` | `https://www.sec.gov/...` | company_corpus |
| `filename`, `page`, `chunk_number`, `ingestion_date`, `text` | — | engine |

The payload is company_corpus's: a key in it that the engine writes itself is
left out and counted (see [Reserved payload keys](#reserved-payload-keys)).

Default narrative scope is families **A** (10-K/10-Q/20-F) and **C** (proxy);
8-K/6-K and ownership forms are high-volume / low-narrative, so down-weight or
opt-in (see the family-weighting note in `company-corpus/docs/INGESTION_RAG.md`).

Install the dependency (it is not vendored) and point a root at the rendered
corpus:

```bash
pip install -e .[company]        # pulls company-corpus from GitHub
# then, with COMPANY_CORPUS_ROOT set in .env (or --root):
data-orchestrator vault --corpus company                # recommended, once onboarded
data-orchestrator company --ciks 320193 --no-vault       # disk fallback, routes to company-e5b-v1
```

`company_corpus` can also be used from a local checkout (`pip install -e .` in
that repo) or simply put on `PYTHONPATH`.

Its vault `local_path` convention isn't settled yet (no `company`-corpus
manifests feed the vault at the time of writing), so `ROUTING["company"]` in
`routing.py` currently carries an empty `local_path_strip` — revisit once
that lands (see the comment there).

### Path resolution convention

The vault stores each document's `local_path` **relative to its corpus repo
root**, e.g. `data/raw/us/C1/2010/<doc_id>.pdf` for `cb_corpus`. A corpus's
`*_ROOT` env key, however, may point at a folder that already sits one level
inside that repo (for `cb_corpus`, `CB_CORPUS_ROOT` is documented as "the
folder that contains `raw/`", i.e. the repo's `data/` dir itself). Joining
`root / local_path` naively would then double-nest into `<root>/data/raw/...`
and find nothing.

`routing.resolve_local_path(corpus, local_path)` reconciles this: each
`CorpusRoute` in `data_orchestrator/routing.py` carries an optional
`local_path_strip` — a **leading-prefix** stripped from `local_path` before
joining with the root. `central-bank` strips `"data/"` for exactly this
reason. Adding a corpus whose root already matches its vault `local_path`
layout needs no strip at all (the default is `""`).

## Usage

```bash
# Vault-selected ingestion (recommended): whatever the vault knows for a
# corpus that the target collection hasn't ingested yet.
data-orchestrator vault --corpus central-bank

# Disk-fallback ingestion, bypassing the vault entirely (its own file ledger,
# no rag_ingestions read/write) — useful before a corpus is vault-onboarded.
data-orchestrator cb_corpus --banks ecb --no-vault

# The company (SEC EDGAR) corpus, disk fallback — routes to company-e5b-v1
# via routing.collection_name(--corpus), not a "company_corpus"-named one.
data-orchestrator company --ciks 320193 --no-vault

# Facts probe: fills has_text_layer / page_count for a corpus's documents
# (feeds the OCR policy; safe to re-run, only unprobed rows are touched).
data-orchestrator probe --corpus central-bank

# Force OCR handling explicitly instead of the engine's auto-detection.
data-orchestrator vault --corpus central-bank --ocr always
```

> Equivalent module form: `python -m data_orchestrator.cli vault --corpus central-bank`

### Useful flags

| flag | meaning |
|---|---|
| `--corpus NAME` | vault corpus to operate on (default: per-source — `central-bank` for `vault`/`cb_corpus`/`probe`, `company` for `company`) |
| `--no-vault` | (cb_corpus/company) use the local JSON-lines file ledger instead of the vault's `rag_ingestions`, and append the run report to `state_dir()/runs.jsonl` (no vault state is read or written). Rejected for the `vault` source and for `probe` (exit 2): both always read and write the vault |
| `--ledger PATH` | (cb_corpus/company with `--no-vault`) path of the file ledger (default `state_dir()/<collection>.jsonl`); ignored by the other ingest runs, rejected for `probe` (exit 2) |
| `--ocr auto\|always\|never` | OCR for scanned pages, passed to the engine as is. `auto` (default): OCR runs page by page where needed, **only if the host switches are on** (`ENABLE_OCR=true` and `TESSDATA_PREFIX` set), else no OCR; `always`: every page, whatever `ENABLE_OCR` says — `TESSDATA_PREFIX` must be set, else the run is refused (exit 2); `never`. The run's `OCR:` line says which applies |
| `--collection NAME` | target Qdrant collection (default: `{corpus}-{model_tag}-v1` via `routing.collection_name` — the same resolution for every source, using `--corpus` or the source's implied corpus) |
| `--source-codes a,b` | (vault source) only these `source_code` values, e.g. `ecb,fr` |
| `--doctypes C1,A3` | only these doc-type codes (cb_corpus, vault); `A1` etc. for company |
| `--languages en,fr` | (vault source) only these language codes |
| `--year-min` / `--year-max` | inclusive year bounds (cb_corpus, company, vault) |
| `--banks a,b` | (cb_corpus source) only these bank codes |
| `--groups a,b` | (cb_corpus source) comma list of doc groups, e.g. A,C |
| `--ciks 320193,789019` | (company source) only these SEC CIK numbers |
| `--prefer pdf\|text` | (company source) rendered PDF (default) or cleaned text |
| `--root PATH` | (cb_corpus/company sources) corpus root override |
| `--limit N` | stop after N newly ingested docs; for `probe`, probe at most N documents (`N >= 1`; `0` or negative is an error, exit 2) |
| `--progress-every N` | progress line every N documents (`N >= 1`, default 25) |
| `--no-resume` | (disk sources) ignore the resume ledger, re-ingest everything. **Rejected for the `vault` source** (exit 2), by design: its resume *is* the `documents`/`rag_ingestions` anti-join, and a policy change means a fresh collection — re-ingest into a new `--collection` name. Re-ingesting into the same collection would leave stale chunks behind when the new chunking yields fewer chunks than the points already stored. Also rejected for `probe` (exit 2), which is resumable by construction |
| `--count-only` | just count matching documents, do not ingest (cb_corpus, company, vault; **rejected for `probe`**, exit 2); for `vault`, also the unsupported documents and their per-source split. A NULL `source_code` is counted under the corpus name — the label the run report's `sources[]` uses (`core.source_label`) — and a NULL `doc_type` as `unknown`. No run report, no `Qdrant:` line |

`--include-html` was removed (argparse rejects it, exit 2): the engine has no
HTML reader, so every such file was a per-document error. HTML is converted to
`.md` or PDF upstream before it reaches the engine.

## Supported document types

`routing.MIME_EXTENSIONS` is built from the extensions the pinned rag-engine
reads (`rag_engine.chunking.SUPPORTED_EXTENSIONS`; the engine dispatches on
the file extension) and this repo's mime type for each:

| mime type | extension |
|---|---|
| `application/pdf` | `.pdf` |
| `text/markdown` | `.md` |
| `text/plain` | `.txt` (read as markdown) |
| `text/csv` | `.csv` |
| DOCX (`application/vnd.openxmlformats-officedocument.wordprocessingml.document`) | `.docx` |
| XLSX (`…spreadsheetml.sheet`) | `.xlsx` |
| PPTX (`…presentationml.presentation`, via markitdown) | `.pptx` |

`text/html` is deliberately absent. The default list is the whole table;
`DATA_ORCHESTRATOR_SUPPORTED_MIME_TYPES` narrows it (entries trimmed,
lower-cased, de-duplicated; an unknown type → exit 2). When a new rag-engine
pin reads another format, `tests/test_routing.py` fails until
`routing._EXTENSION_MIME` names its mime type.

A vault row is selected iff its `local_path` ends (case-insensitive, trailing
whitespace trimmed) with a supported extension **and** its `mime_type` is NULL
or supported. Mime values are normalised first: lower-cased, parameters
dropped (`text/plain; charset=utf-8` → `text/plain`), whitespace trimmed, blank
= NULL. So a supported mime on another extension (`application/pdf` +
`.html`) and a row with no `local_path` are unsupported. Unsupported rows are
never offered to the engine and never degrade a run. A vault run reports them
three ways: `totals.docs_unsupported` (the sum); `docs_unsupported` on each
`sources[]` entry (`0` when none) — `sources[]` lists only the source codes
the run actually touched, so a source code with only unsupported documents is
not there; and the full per-source split, every source code included, as
`unsupported_by_source` (a JSON object, keys sorted in the report — Postgres JSONB keeps its own key order — stored in
`runs.extra->'unsupported_by_source'`, written on every vault run, `{}` when
there are none; a fatal run keeps it when the count was computed before the
failure). A NULL `source_code` is filed under the corpus name, as for
the ingested ones. Before each run that has any, the CLI prints
`Skipping N document(s)…` with the count. Disk sources report none of these keys (they cannot know).

## The write protocol

Every document's Qdrant points are upserted **before** its `rag_ingestions`
record is written — never the other way round, and batching never spans
documents. That order makes the invariant **`rag_ingestions ⊆ Qdrant`** hold
at all times: if the process crashes between the two writes, the vault simply
under-claims that one document, and the next resume pass re-ingests it. Point
ids are deterministic (derived from `doc_id`, `page`, `chunk_number` by
`rag_engine.schema.point_id`), so a
re-ingested document's points overwrite in place instead of duplicating —
resume is a pure retry, not a special code path. A failed write is not undone
either: the document is not recorded, and the next run writes it again. A `reconcile` command
(drift audit + soft-deleted cleanup) is a specified future addition, not
built here.

## Run reports and exit codes

Every ingest run (`vault`, `cb_corpus`, `company`) writes a telemetry row to
the vault's `runs` table (or appends `state_dir()/runs.jsonl` with
`--no-vault`): run id, `corpus`, per-source counters and the outcome — on a
fatal error too, best effort (a vault that cannot be reached cannot record it;
the failure is said on stderr, `warning: failed to write run report to vault:`
or, with `--no-vault`, `warning: failed to append run report:`). A fatal
error in the middle of a run (a refused API key on a write, a source that
stops answering) reports what was done before it: `totals` and `sources[]`
count the documents seen, ingested and failed up to the failure, as
`rag_ingestions` records them (the document being written when a key is
refused is counted as seen, neither new nor failed), with `outcome` `failed`,
`exit_code` 1 and the reason under `fatal`; the usual summary (`Done.` and its
counts) is printed, then the `error: fatal:` line. A fatal error before the
first document reports zeros and prints no summary. **No report**
is written by `--count-only`, by `probe`, or by a run stopped by a pre-flight
error (exit 1 or 2, before any work). Exit codes are honest:

- `0` **ok**;
- `3` **degraded** when (a) nothing was ingested while attempts failed, or
  (b) `docs_failed / docs_attempted` exceeds the ratio, where
  `docs_attempted = ingested + empty + failed` (resume skips excluded). The
  ratio is `DATA_ORCHESTRATOR_DEGRADED_FAILURE_RATIO`, default `0.10`; `0` =
  any failure degrades, `1` = rule (b) off. `docs_unsupported` never degrades;
- `1` **failed**: a fatal error (e.g. Qdrant unreachable; a refused API key,
  at setup or in the middle of a run; an embedding model whose dimension is
  not the declared one; a collection that is not a single unnamed vector of
  the model's dimension with cosine distance — another dimension, another
  distance, named vectors: refused before any document, whether it
  pre-existed or a concurrent run just created it, as
  `error: fatal: CollectionMismatchError: collection '<name>': expected a
  single unnamed 768-d vector with Cosine distance, found <what>: use another
  --collection`); also a
  pre-flight error (unknown `--corpus`, missing corpus root or `--root`, a
  cb_corpus root without manifest), with no report; also a vault failure
  (missing `DATABASE_URL`, unreachable database, failing query) under `probe`
  or `vault --count-only`, said as `error: fatal: <Type>: <message>` (no
  traceback, no report);
- `2` **usage/config error**, before any work and with no report: bad
  arguments (argparse), `--limit`/`--progress-every` below 1, `--no-vault` or
  `--no-resume` on the `vault` source, `--count-only`, `--no-vault`,
  `--no-resume` or `--ledger` on `probe`, a missing `DATA_ORCHESTRATOR_DOTENV`
  file, an invalid `DATA_ORCHESTRATOR_SUPPORTED_MIME_TYPES` or
  `DATA_ORCHESTRATOR_DEGRADED_FAILURE_RATIO`, and, on an ingest run, a set
  `QDRANT_PORT` that is not a plain integer in `1..65535`, an invalid engine
  setting (`RAG_ENGINE_*`, `QDRANT_HTTPS`, `DATA_ORCHESTRATOR_EMBEDDING_DIM`,
  a model other than the default without its dimension), a
  `RAG_ENGINE_BACKEND` other than `torch`, or `--ocr always` without
  `TESSDATA_PREFIX`.

A per-document failure is listed in the run summary as `<Type>: <message>`
with the engine's error types (`UnsupportedFormatError`, `ValueError` for a
reserved payload key, the parser's or the Qdrant client's own error
otherwise).

`runs.corpus` is now written on every report (success and fatal, all
sources). **Caveat:** the vault health views `runs_sources`,
`sources_without_cadence` and `source_health` group on `(corpus,
source_code)` without filtering on `tool`; until the vault follow-up filters
them on the crawler tools, orchestrator runs count there as crawler runs of
the sources they list. Only the source codes a run actually touched appear in
`runs.sources` (the full unsupported split lives in
`extra->'unsupported_by_source'`, which those views do not read), but a run
still touches every source code its selection covers: until the views filter
on `tool`, scope the first real runs with `--source-codes`.

Report fields that have no column of their own are swept into `runs.extra`
(JSONB, NULL when there are none), the same rule the vault's runs ingester
applies to central-bank-corpus's `runs.jsonl`. A `vault` run carries
`unsupported_by_source` (even `{}`) whenever its unsupported count was computed
— every successful one, so a successful `vault` run's `extra` is never NULL;
a disk source's successful run has a NULL `extra`. A fatal run's reason is
`extra->>'fatal'` (capped at 500 characters, the key central-bank-corpus
writes); rows written before 2026-11 carry it as `extra->>'error'`, so a reader
covering both uses `coalesce(extra->>'fatal', extra->>'error')`.

## Collections & embedding model

Collections are named by the orchestrator, one per `(corpus, embedding
generation)`:

```
{corpus}-{model_tag}-v{n}
```

e.g. `central-bank-e5b-v1`. The embedding model defaults to
`intfloat/multilingual-e5-base` (768-d) and is overridable via `DATA_ORCHESTRATOR_EMBEDDING_MODEL`
(used by CI to swap in a tiny model for integration tests). Bump `n` (or the
`EMBEDDING_VERSION` policy tag in `routing.py`) when chunking/embedding
*policy* changes, not just the model name.

One resolution serves the model that embeds, the collection name and the
`embedding_model` of `rag_ingestions`: `DATA_ORCHESTRATOR_EMBEDDING_MODEL`,
else the engine's `RAG_ENGINE_MODEL`, else the default. The engine wants the
vector dimension declared (it compares it with the loaded model and refuses a
difference): 768 for the default model, `DATA_ORCHESTRATOR_EMBEDDING_DIM`
(else `RAG_ENGINE_DIM`) for any other, which is then required — e.g.

```bash
DATA_ORCHESTRATOR_EMBEDDING_MODEL=sentence-transformers/paraphrase-albert-small-v2
DATA_ORCHESTRATOR_EMBEDDING_DIM=768
```

Passages are embedded with the `passage: ` prefix (the engine's
`RAG_ENGINE_PASSAGE_PREFIX`, leave unset: another prefix produces vectors that
do not match a collection built with this one).

A collection is created on first use as one unnamed cosine vector of the
model's dimension. An existing collection is checked before any document and
refused when it is anything else: another dimension, named vectors, or **a
distance other than cosine**.

## Qdrant payload

Every chunk's Qdrant point carries:

| field | source |
|---|---|
| `doc_id`, `corpus`, `source_code`, `doc_type`, `title`, `date`, `year`, `language`, `sha256`, `provenance` | vault (`documents`, merged with corpus-specific `extra`) |
| `filename`, `page`, `chunk_number`, `text` | engine (chunking output; `page` only for formats with pages) |
| `ingestion_date` | engine — one ISO timestamp per document, shared by all of its points |

This is rag-engine's payload schema v1 (`rag_engine.schema`), the payload this
repo assembled itself before 2026-11. What was shown about the switch: with
the same versions of the libraries that decide text and vectors (chunknorris
and its PDF library PyMuPDF, sentence-transformers, torch, transformers, the
revision of the embedding model) and the same device, the same documents give
the same point ids, the same payloads and the same vectors. It was checked (i)
on a committed fixture with a deterministic test embedder and OCR off
(`tests/test_engine_parity.py`), (ii) end to end, old code against new code,
on PostgreSQL and Qdrant with a real small model, and (iii) with OCR on, in
English, on generated documents. Two payload differences are declared: invalid
UTF-8 in a text file is replaced by U+FFFD (it was dropped), and a reserved
key found in a document's free metadata is dropped (a stray `page` on a
document without pages used to be kept; see below). Another version of one of
those libraries, or another device, is outside what was checked:
[MIGRATION.md](MIGRATION.md) says how to keep the versions and how to validate
on real data. The key is `chunk_number`, not `chunk_index`.

### Reserved payload keys

The engine writes `doc_id`, `filename`, `chunk_number`, `ingestion_date`,
`text` and `page` itself and **refuses** metadata that sets one of them (a
`doc_id` equal to the document's is accepted). Policy of this repo, applied
where metadata it does not own enters:

- **vault source**: such a key in `documents.extra` is left out of the payload
  (the vault's own columns are never dropped; `doc_id` always comes from its
  column);
- **company source**: such a key in company_corpus's payload is left out (a
  `doc_id` other than the item's included);
- **cb_corpus source**: builds its payload from fixed keys, none reserved.

Each key left out is counted: the run summary warns (`warning: N metadata
key(s) reserved for the engine were left out of the payloads (filename x2,
…)`) and the run report carries `totals.payload_keys_dropped` (present only
when it is not zero). It never changes the outcome or the exit code: a stray
key must not fail a document. A `SourceItem` built by hand with a reserved
key is not rewritten: the engine refuses that document (`ValueError`, one
per-document error).

## Operational note: fresh database

On a fresh database, the **vault's own ingestion service must run at least
once before this orchestrator** — its DDL train is what creates the
`rag_ingestions` table this repo reads and writes. Running the orchestrator
against a database that has never seen the vault service fails at
`vault.connect()` / the first `rag_ingestions` query.

## Facts probe

`data-orchestrator probe --corpus <corpus>` fills `has_text_layer` and
`page_count` for every document with `has_text_layer IS NULL`, feeding the
facts-driven OCR policy. It is the **only** writer of those two columns.
Only unprobed rows are selected, so the pass is resumable by construction and
converges to a no-op on re-run. Each document's `UPDATE` runs inside its own
`SAVEPOINT`, so one failing row (a corrupt PDF path, a constraint) can never
poison the whole batch's transaction — it is rolled back to the savepoint and
counted as an error, while every other document in the batch still commits.
`--count-only`, `--no-vault`, `--no-resume` and `--ledger` are rejected
(exit 2); `--limit N` probes at most N documents. A vault failure (missing
`DATABASE_URL`, unreachable database) is an `error: fatal:` line and exit 1.

## Testing

In the virtual environment of the installation, with the test tools
(`pip install -e ".[dev]"`):

```bash
python -m pytest -q                    # unit tests (no network, no service)
python -m pytest -m integration -q     # integration tests (need Docker)
python tools/check_engine_pin.py       # the rag-engine installed is the pinned commit
```

Integration tests spin up throwaway containers via `docker run` (skipped
automatically if Docker is unavailable): `postgres:16` and
`qdrant/qdrant:v1.20.0`, the Qdrant version production runs, pinned by the
`QDRANT_IMAGE` constant of `tests/integration/conftest.py` (change it when
production is upgraded). They use a tiny embedding model
(`DATA_ORCHESTRATOR_EMBEDDING_MODEL=sentence-transformers/paraphrase-albert-small-v2`,
`DATA_ORCHESTRATOR_EMBEDDING_DIM=768`, downloaded on first use) to keep CI
fast. CI (`.github/workflows/tests.yml`) runs both suites on pushes to `main`
and on pull requests — the unit suite on Python 3.12 and 3.13 (production
runs 3.13): the CPU build of PyTorch, then `pip install -e ".[dev]"`, which
brings rag-engine at the commit pinned in `pyproject.toml`. A step then
checks it: `tools/check_engine_pin.py` compares the commit pip recorded for
the installed rag-engine (`direct_url.json`, `vcs_info.commit_id`) with the
sha in `pyproject.toml` and fails on a difference (it also fails on an engine
installed from a local checkout, which records no commit).

**Bumping the rag-engine pin** changes what is written to Qdrant only if the
guards say so: `tests/test_engine_parity.py` (the points written for six
fixture documents must stay the ones captured from the code this repo ran
before it used rag-engine: ids, payloads, vectors) and `tests/test_routing.py`
(the supported extensions). The parity test runs with a deterministic test
embedder and OCR off: it does not see a change of sentence-transformers,
torch, transformers or the model. Never regenerate the parity fixture to make
a failing test pass.

## Adding a new corpus

Add one `CorpusRoute` entry to `ROUTING` in `data_orchestrator/routing.py`
(root env key + optional `local_path_strip`) — no new connector code, as long
as the corpus is onboarded to the vault. The `vault` source works for any
corpus already in `documents`.

## Adding a new disk-fallback source

Create `data_orchestrator/sources/<name>.py` exposing an `iter_items(...)`
generator that yields `core.SourceItem(doc_id, path, payload)`, then wire it
into `cli.py` (`source` choices + dispatch). The core needs no
changes. A payload copied from metadata the connector does not own goes
through `core.split_reserved_keys` (see [Reserved payload keys](#reserved-payload-keys)).

## Corpus and RAG-state views

Coverage, cadence and RAG-state views live in Metabase over the vault
([MyOpenFund/vault](https://github.com/MyOpenFund/vault)): `documents`,
`rag_ingestions`, `cadence` and `runs` are the tables to chart (see the
`runs.corpus` caveat under [Run reports](#run-reports-and-exit-codes)). The former
Streamlit dashboard shipped with this repo was removed (it read the extinct
single-file manifest and duplicated what the vault now records); its feature
inventory is kept in the project notes.
