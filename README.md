# data-orchestrator

Installable Python package (repo: `MyOpenFund/data-orchestrator`). It is the **policy
layer** that sits between three systems, each owning one thing:

| system | owns | this repo... |
|---|---|---|
| **vault** (Postgres) | facts, document selection, ingestion state (`documents`, `rag_ingestions`) | reads the selection, writes the ingestion record |
| **eigenmind** (fork) | the RAG engine — chunking, embedding, Qdrant client | drives it as a library, never reimplements it |
| **Qdrant** | the derived vector projection | writes to it, treats it as rebuildable from vault + corpus |

> read whatever the vault says is new → drive it through eigenmind → land it in Qdrant

The orchestrator carries **no RAG logic of its own** — chunking, embedding and
the Qdrant client all live in [eigenmind](https://github.com/MyOpenFund/eigenmind),
consumed as an editable dependency at a pinned fork commit (see [Testing](#testing)). What this repo owns is orchestration
policy: which documents to pick up next, per-corpus path routing, resumability,
and the write protocol that keeps Qdrant and the vault consistent.

**The dependency is one-way.** This package *uses* eigenmind; eigenmind never
imports this one and keeps working entirely on its own. It also *uses* the
vault's Postgres schema (`documents`, `rag_ingestions`) but never owns or
migrates it — the vault's own ingestion service runs that DDL.

## Layout

```
data-orchestrator/
  pyproject.toml          # packaging — installs the `data-orchestrator` CLI
  data_orchestrator/
    __init__.py            # public API: SourceItem, run_ingest, Ledger, …
    config.py              # the one .env loader (see "Configuration loading")
    core.py                # engine: chunk -> embed -> upsert + resume ledger
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

1. **The eigenmind engine.** Clone [MyOpenFund/eigenmind](https://github.com/MyOpenFund/eigenmind)
   locally, check out the commit CI pins, and install both editable (not from
   PyPI). This repo must itself be an **editable** install (`pip install -e .`)
   for the repo-root `.env` fallback below to find its `.env`:

   With `$EIGENMIND_PATH` = your eigenmind clone:

   ```bash
   git -C $EIGENMIND_PATH checkout 84d2ac4   # the CI pin, see Testing
   pip install -e $EIGENMIND_PATH
   pip install -e .
   ```

2. **`.env`.** Copy `.env.example` to `.env` (git-ignored, per-machine) and set:

   | key | meaning |
   |---|---|
   | `EIGENMIND_PATH` | local clone of the eigenmind fork |
   | `DATABASE_URL` | vault Postgres, e.g. `postgresql://user:pass@host:5432/documents` |
   | `CB_CORPUS_ROOT` | local root of the central-bank corpus (folder containing `raw/`) — required whenever the central-bank corpus is used (vault source, probe, or cb_corpus fallback) |
   | `COMPANY_CORPUS_ROOT` | local root of the company_corpus (SEC EDGAR) data dir — optional, only needed for the `company` disk fallback; unset falls back to company_corpus's own default data dir |
   | `QDRANT_HOST` / `QDRANT_PORT` | the Qdrant the engine writes to (read by eigenmind). **Always set both**: unset, eigenmind falls back to `localhost:6333`, the **legacy** store; every ingest run prints `Qdrant: <host>:<port>` (the engine's store, as constructed) and warns on stderr when the port is 6333. A set `QDRANT_PORT` must be a plain integer in `1..65535` (surrounding whitespace ignored): empty, signed, decimal or followed by an inline `# comment` is an error on every ingest run (exit 2, before any work) |
   | `DATA_ORCHESTRATOR_EMBEDDING_MODEL` | optional, overrides the embedding model (see [Collections](#collections--embedding-model)) |
   | `DATA_ORCHESTRATOR_DOTENV` | optional, **real environment only** (never read from a `.env`): explicit path of the `.env` to load instead of searching (a relative path is resolved against the current directory; the absolute path is what is read and printed); a missing file is an error (exit 2 from the CLI, `FileNotFoundError` for a library caller) |
   | `DATA_ORCHESTRATOR_STATE_DIR` | optional, where the `--no-vault` file ledger and run reports are written; defaults to `<repo>/state` (`~` is expanded) |
   | `DATA_ORCHESTRATOR_SUPPORTED_MIME_TYPES` | optional comma list narrowing the types the vault selection offers to the engine (see [Supported document types](#supported-document-types)); a type the engine cannot read is an error (exit 2) |
   | `DATA_ORCHESTRATOR_DEGRADED_FAILURE_RATIO` | optional plain decimal in `[0, 1]`, default `0.10` (see [Run reports](#run-reports-and-exit-codes)); anything else is an error (exit 2) |

3. **Qdrant** reachable at `QDRANT_HOST:QDRANT_PORT`.

### Configuration loading

- **One loader, first.** `data-orchestrator` (`cli.main`) loads the `.env`
  right after parsing its arguments, before anything imports eigenmind.
  eigenmind's own python-dotenv (`override=False`) runs at its first import and
  can then only *add* keys this file did not set: ours wins.
- **Which file.** `DATA_ORCHESTRATOR_DOTENV` if set — read from the real
  environment only, never from a `.env`, a relative path resolved against the
  current directory; a missing file is an `error:` and exit 2 before any work
  — else the nearest `.env` in the current directory or
  its parents, else the `.env` at this repo's root (found next to the
  installed package, hence the editable install).
- **Traceability.** Every run prints `Config: .env = <absolute path>` (or
  `Config: no .env file found`) first. Every ingest run (`vault`, `cb_corpus`,
  `company`, not `--count-only`) then prints `Qdrant: <host>:<port>` once the
  engine's store is constructed — read from that store, i.e. after
  eigenmind's own python-dotenv ran at its first import (which can still set
  a `QDRANT_*` key our `.env` left unset) — with a stderr `warning:` on port
  6333 (the legacy store; compared numerically, `06333` is 6333). A
  `QDRANT_PORT` that is set but is not a plain integer in `1..65535` (empty
  included) stops an ingest run with an `error:` and exit 2 before any work,
  rather than let the engine fail mid-run. `--count-only` and `probe` never
  touch Qdrant, do not check `QDRANT_PORT` and print no `Qdrant:` line.
- **Literal values.** No `$VAR` expansion, surrounding quotes stripped, and
  **no inline comments**: a `#` after a value is part of the value. A key
  already set in the real environment always wins over the file.
- **Library callers** (`run_ingest` & co. without the CLI) get no loader
  *first*: `vault.connect()` and `routing.corpus_root()` call
  `config.load_dotenv()` lazily (raising `FileNotFoundError` when
  `DATA_ORCHESTRATOR_DOTENV` names a missing file), which may be after
  eigenmind's python-dotenv already ran. Call
  `data_orchestrator.config.load_dotenv()` yourself before importing
  anything that reaches eigenmind.

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
| `filename`, `page`, `chunk_number`, `text` | — | pipeline |

### `company` — the company (SEC EDGAR) corpus

The **micro** layer: company filings from SEC EDGAR, produced by the
[`company-corpus`](https://github.com/MyOpenFund/company-corpus) project. That
project discovers filings, downloads and decomposes the complete submission,
extracts clean text and renders each filing's primary document to a
human-readable, **page-anchored PDF** — which flows through the same eigenmind
PDF loader with no change. Cleaned text is the fallback when no PDF exists.

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
| `filename`, `page`, `chunk_number`, `text` | — | pipeline |

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
| `--ocr auto\|always\|never` | OCR fallback for scanned pages (default `auto`: defers to the engine's tesseract availability check) |
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

`--include-html` was removed (argparse rejects it, exit 2): eigenmind has no
HTML reader, so every such file was a per-document error. HTML is converted to
`.md` or PDF upstream before it reaches the engine.

## Supported document types

`routing.MIME_EXTENSIONS` is the one declaration of what the pinned eigenmind
commit can chunk (it dispatches on the file extension):

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
lower-cased, de-duplicated; an unknown type → exit 2). Bump the table together
with the eigenmind pin.

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
`unsupported_by_source` (a JSON object sorted by key, stored in
`runs.extra->'unsupported_by_source'`, written on every vault run, `{}` when
there are none; a fatal run keeps it when the count was computed before the
failure). A NULL `source_code` is filed under the corpus name, as for
the ingested ones. Before each run the CLI prints `Skipping N document(s)…`
with the count. Disk sources report none of these keys (they cannot know).

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

## Run reports and exit codes

Every ingest run (`vault`, `cb_corpus`, `company`) writes a telemetry row to
the vault's `runs` table (or appends `state_dir()/runs.jsonl` with
`--no-vault`): run id, `corpus`, per-source counters and the outcome — on a
fatal error too, best effort (a vault that cannot be reached cannot record it;
the failure is said on stderr, `warning: failed to write run report to vault:`
or, with `--no-vault`, `warning: failed to append run report:`). **No report**
is written by `--count-only`, by `probe`, or by a run stopped by a pre-flight
error (exit 1 or 2, before any work). Exit codes are honest:

- `0` **ok**;
- `3` **degraded** when (a) nothing was ingested while attempts failed, or
  (b) `docs_failed / docs_attempted` exceeds the ratio, where
  `docs_attempted = ingested + empty + failed` (resume skips excluded). The
  ratio is `DATA_ORCHESTRATOR_DEGRADED_FAILURE_RATIO`, default `0.10`; `0` =
  any failure degrades, `1` = rule (b) off. `docs_unsupported` never degrades;
- `1` **failed**: a fatal error (e.g. Qdrant unreachable, a collection stored
  with another vector dimension, or a named-vector collection, whose
  dimension cannot be verified: either one is refused before any document,
  whether it pre-existed or a concurrent run just created it); also a
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
  `QDRANT_PORT` that is not a plain integer in `1..65535`.

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
`intfloat/multilingual-e5-base` and is overridable via `DATA_ORCHESTRATOR_EMBEDDING_MODEL`
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

```bash
./venv/bin/python -m pytest -q                    # unit tests
./venv/bin/python -m pytest -m integration -q      # integration tests (Docker)
```

Integration tests spin up throwaway Postgres and Qdrant containers via
`docker run` (skipped automatically if Docker is unavailable) and use a tiny
embedding model (`DATA_ORCHESTRATOR_EMBEDDING_MODEL=sentence-transformers/paraphrase-albert-small-v2`)
to keep CI fast. CI (`.github/workflows/tests.yml`) runs both suites on pushes
to `main` and on pull requests, checking out this repo and the `MyOpenFund/eigenmind` fork side by
side, the fork **pinned by commit** (`ref:
84d2ac41fb06704bdbc012afd1a010e758bcb5a1`; upstream rewrote its history on
2026-11-08). To match CI locally:
`git -C $EIGENMIND_PATH checkout 84d2ac4 && pip install -e $EIGENMIND_PATH`.
Bump the pin (both checkouts) together with `routing.MIME_EXTENSIONS`.

## Adding a new corpus

Add one `CorpusRoute` entry to `ROUTING` in `data_orchestrator/routing.py`
(root env key + optional `local_path_strip`) — no new connector code, as long
as the corpus is onboarded to the vault. The `vault` source works for any
corpus already in `documents`.

## Adding a new disk-fallback source

Create `data_orchestrator/sources/<name>.py` exposing an `iter_items(...)`
generator that yields `core.SourceItem(doc_id, path, payload)`, then wire it
into `cli.py` (`source` choices + dispatch). The core engine needs no
changes.

## Corpus and RAG-state views

Coverage, cadence and RAG-state views live in Metabase over the vault
([MyOpenFund/vault](https://github.com/MyOpenFund/vault)): `documents`,
`rag_ingestions`, `cadence` and `runs` are the tables to chart (see the
`runs.corpus` caveat under [Run reports](#run-reports-and-exit-codes)). The former
Streamlit dashboard shipped with this repo was removed (it read the extinct
single-file manifest and duplicated what the vault now records); its feature
inventory is kept in the project notes.
