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

**The `documents` contract.** What a corpus writes on each manifest line, and
what the vault stores from it, is the vault's `documents` contract 1.1.0
([`vault-contract`](https://github.com/MyOpenFund/vault/tree/master/contract),
standard library only), a runtime dependency pinned by commit in
`pyproject.toml` next to the engine. This repo uses it three ways:

- a manifest row is split into the vault's columns and `extra` by
  `vault_contract.split_row`, the split the vault ingests with
  (`payload.manifest_row_to_vault_row`), never by a copy of its field list;
- a document's identity is `(corpus, doc_id)`, as in the vault: `rag_ingestions`
  is read and written on `(corpus, doc_id, collection)` (`ON CONFLICT (corpus,
  doc_id, collection)`), the selection's anti-join and `sync-metadata`'s join
  match the corpus too;
- `local_path` is relative to the corpus data root (the directory holding
  `manifest/` and `raw/`), joined with that corpus's root env key; a path the
  contract refuses (`vault_contract.is_bad_local_path` with
  `transition=False`: absolute, `..`, a drive, a backslash, a leading `data/`,
  ...), or one leaving the root through a symbolic link,
  is never read: the document is skipped, counted in the run report
  (`docs_bad_local_path`) and as a failure for the outcome.

This version needs a vault on the contract (vault `12ffd11` or later, after its
ingestion ran once: that run is the DDL train), and reads only the contract
form: since the contract's transition cleanup (vault `2859fa6`) no pre-contract
form is read (see [Deploying with the documents contract](#deploying-with-the-documents-contract)). Against an older vault every
run that uses it (the vault source, `sync-metadata`, a disk source with the
vault ledger) stops before any Qdrant write with `VaultSchemaTooOld`, naming
what is missing (one catalog read, `vault.require_contract_schema`). See [Deploying with the documents contract](#deploying-with-the-documents-contract).

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
    sync_metadata.py         # sync-metadata: vault date metadata -> Qdrant payloads, no re-embedding
    cli.py                    # console entrypoint (`data-orchestrator`)
    sources/
      vault.py                # source #1: vault-selected documents (any corpus)
      cb_corpus.py              # source #2: the central-bank PDF corpus (disk fallback)
  state/                   # resume ledgers when running with --no-vault (gitignored)
```

The corpus family has two layers feeding the same RAG, both selected through
the `vault` source:

- **Macro** — `central-bank-corpus` (vault corpus `central-bank`): central-bank
  documents; its disk connector `cb_corpus` (`--no-vault`) remains as a fallback.
- **Micro** — `company-corpus` (vault corpus `company`): company filings from
  SEC EDGAR, read from the vault only (`vault --corpus company`).

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
     ```

     then the package:

     ```bash
     pip install -e .
     ```

     Skipping the first command on Linux pulls the CUDA packages with the
     default wheel (several gigabytes) for nothing.
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
   | `CB_CORPUS_ROOT` | data root of the central-bank corpus on this machine (the folder holding `manifest/` and `raw/`) — required whenever the central-bank corpus is used (vault source, probe, or cb_corpus fallback) |
   | `COMPANY_CORPUS_ROOT` | data root of the company corpus on this machine (the folder holding `manifest/` and `raw/`, e.g. `~/company-corpus-data`) — required by `vault --corpus company` and `probe --corpus company`; the vault's `local_path` (`raw/<cik>/<doc_type>/<year>/<doc_id>.pdf`) is joined with it |
   | `QDRANT_HOST` / `QDRANT_PORT` | the Qdrant the engine writes to (read by rag-engine). **Always set both**: unset, the engine falls back to `localhost:6333`, the **legacy** store; every ingest run prints `Qdrant: <host>:<port> (api key: set\|unset)` (the engine's store, as constructed) and warns on stderr when the port is 6333. A set `QDRANT_PORT` must be a plain integer in `1..65535` (surrounding whitespace ignored): empty, signed, decimal or followed by an inline `# comment` is an error on every ingest run (exit 2, before any work) |
   | `QDRANT_API_KEY` | API key of that Qdrant (read by rag-engine, sent as the `api-key` header; blank or unset = none). The start line says whether one is set, never its value. A 401/403 from Qdrant ends the run with **one** fatal error, at collection setup (before any document) or on a write in the middle of a run (the documents already recorded stay recorded, the one being written is not): `error: fatal: QdrantAuthError: Qdrant refused the request (HTTP 401): check QDRANT_API_KEY`, exit 1, same text in `runs.extra->>'fatal'`; the run report still counts the documents ingested before the refusal (see [Run reports](#run-reports-and-exit-codes)). See `QDRANT_HTTPS` for the plain-HTTP warning |
   | `QDRANT_HTTPS` | `true` only if that Qdrant is served over TLS (read by rag-engine; default `false`, plain HTTP). With a key over plain HTTP `qdrant-client` warns `Api key is used with an insecure connection`: expected inside the tailnet |
   | `RAG_ENGINE_QDRANT_TIMEOUT` | optional, seconds a request to Qdrant may take (default `60`) |
   | `ENABLE_OCR` / `TESSDATA_PREFIX` | the host's OCR switches, read by the engine **from the process environment each time a file is chunked**: with `--ocr auto` (the default) scanned pages are read by OCR only when `ENABLE_OCR=true` **and** `TESSDATA_PREFIX` is set (the folder of Tesseract's `*.traineddata` files, or its parent). Put both in this `.env` (or the real environment): left out, `auto` silently means `never`. Every ingest run says which it is: `OCR: available (languages: eng)` or `OCR: off (ENABLE_OCR not true or TESSDATA_PREFIX unset)`. Neither has a default (unset = off). `--ocr always` does not look at `ENABLE_OCR` but still needs `TESSDATA_PREFIX`: forcing OCR without the language data is a configuration error, refused before any work (`error: forced OCR needs TESSDATA_PREFIX; set it or use --ocr auto/never`, exit 2) |
   | `RAG_ENGINE_OCR_LANGUAGES` | optional, Tesseract languages used by OCR, codes joined by `+` (default `eng`; e.g. `eng+fra`). The engine uses exactly this list; it does not look at which language files are installed |
   | `DATA_ORCHESTRATOR_EMBEDDING_MODEL` (else `RAG_ENGINE_MODEL`) | optional, overrides the embedding model (default `intfloat/multilingual-e5-base`; see [Collections](#collections--embedding-model)); `RAG_ENGINE_MODEL`, the engine's own name for it, is read when this one is unset |
   | `DATA_ORCHESTRATOR_EMBEDDING_DIM` (else `RAG_ENGINE_DIM`) | vector dimension of that model. Not needed for the default model (768); **required for any other model** (`RAG_ENGINE_DIM` is read when this one is unset), else exit 2. A dimension that is not the loaded model's is a fatal error before any document |
   | `RAG_ENGINE_BATCH_SIZE` | optional, points per write to Qdrant (default `100`) |
   | `RAG_ENGINE_BACKEND` | leave unset (default `torch`): ingestion uses the torch backend, the one this package installs. `onnx`, the engine's backend for search services, is not installed here and is refused on an ingest run (`error: RAG_ENGINE_BACKEND='onnx': ingestion uses the torch backend, …`, exit 2, before any work), like any other value |
   | `RAG_ENGINE_SEARCH_LIMIT` / `RAG_ENGINE_SEARCH_MAX_LIMIT` / `RAG_ENGINE_PARITY_MIN` / `RAG_ENGINE_ONNX_THREADS` | leave unset (defaults `10`, `50`, `0.999`, none). Settings of the engine's search side: ingestion does not use them, but the engine reads all of its settings at once, so an invalid value (not a number, a limit below 1, a parity outside `(0, 1]`) stops an ingest run with exit 2 |
   | `DATA_ORCHESTRATOR_DOTENV` | optional (default: unset, the `.env` is searched as described below), **real environment only** (never read from a `.env`): explicit path of the `.env` to load instead of searching (a relative path is resolved against the current directory; the absolute path is what is read and printed); a missing file is an error (exit 2 from the CLI, `FileNotFoundError` for a library caller) |
   | `DATA_ORCHESTRATOR_PG_KEEPALIVES_IDLE` / `_INTERVAL` / `_COUNT` | optional, TCP keepalives of the vault connection (libpq `keepalives_idle`, `keepalives_interval`, `keepalives_count`; always on; defaults `30` s, `10` s, `3`): while the connection is **idle**, a dead peer (a laptop asleep, a dropped link) is noticed within idle + interval x count seconds. Keepalives do not fire while a request waits for an answer: see the next line. A whole number >= 1, else exit 2 |
   | `DATA_ORCHESTRATOR_PG_TCP_USER_TIMEOUT_MS` | optional, libpq `tcp_user_timeout` of the vault connection, in milliseconds (default `30000`; `0` = the system default): a request sent into a dead link (data never acknowledged) fails after that long instead of waiting for TCP's retransmissions (about 15 minutes on Linux); the run then stops with `VaultSessionLost`. **Linux only**: libpq ignores it where the system has no `TCP_USER_TIMEOUT` (macOS), where such a request waits for the system's retransmission limit. A whole number >= 0, else exit 2 |
   | `DATA_ORCHESTRATOR_LOG_LEVEL` | optional, level of the log lines the CLI sends to stderr (this package's and the engine's: model loaded, collection created, a text file with invalid bytes, …): `DEBUG`, `INFO` (default), `WARNING`, `ERROR`, `CRITICAL`; anything else is an error (exit 2) |
   | `DATA_ORCHESTRATOR_STATE_DIR` | optional, where the `--no-vault` file ledger and run reports are written; defaults to `<repo>/state` (`~` is expanded) |
   | `DATA_ORCHESTRATOR_SUPPORTED_MIME_TYPES` | optional comma list narrowing the types the vault selection offers to the engine (default: every type the pinned engine reads; see [Supported document types](#supported-document-types)); a type the engine cannot read is an error (exit 2) |
   | `DATA_ORCHESTRATOR_PAYLOAD_DROP_EXTRA_KEYS` | optional comma list of the keys of the vault's `extra` (and of a manifest row's fields outside the vault's columns) left out of every payload by design, silently and without being counted (default `html_path,alt_urls,alt_source_urls,local_path,pdf_path,text_path,primary_path,submission_path`: a path on the producer's machine, other addresses of the document, paths in the corpus, the last four the company corpus's other artefacts; see [Qdrant payload](#qdrant-payload)). `none` drops no key; unset or blank is the default. A vault column (`title`, `document_url`, `date_precision`, `date_kind`, `entity_scheme`, `entity_value`, ...: the column always wins) or a key the engine writes or derives itself is refused (`pdf_url` and `bank_code`, former column names, are ordinary keys of `extra` since the contract's transition cleanup and may be named; the `cb_corpus` disk source refuses a manifest row carrying either, `legacy_key`), as is a list with no entry (exit 2) |
   | `DATA_ORCHESTRATOR_DEGRADED_FAILURE_RATIO` | optional plain decimal in `[0, 1]`, default `0.10` (see [Run reports](#run-reports-and-exit-codes)); anything else is an error (exit 2) |
   | `DATA_ORCHESTRATOR_REPORT_DOC_IDS` | optional whole number >= 0, default `20`: how many `doc_id`s of the documents skipped, never read, a run report names per kind (`bad_local_path_doc_ids`, `refused_contract_doc_ids`; the counts beside them are always whole; `0` names none). Anything else is an error (exit 2) |

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
  not `--count-only`) then prints the OCR line (`OCR: available
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

The `vault` source is the recommended way for every corpus (see
[Usage](#usage)); the central-bank corpus also has a disk-fallback connector
(`cb_corpus`, usable with `--no-vault`). It hard-codes no collection name: the
default is always the routed `{corpus}-{model_tag}-v1` (`central-bank-e5b-v1`),
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

The payload is in the **vault's vocabulary**: a document gets the same
payload keys and values from this source as from the `vault` source
(`payload.manifest_row_to_payload`). Both sources build it with the same
`payload.row_to_payload`, and the split of a manifest row into the vault's
columns and `extra` is the vault's own: `vault_contract.split_row` (the
`documents` contract, with `transition=False` as the vault ingests), through
`payload.manifest_row_to_vault_row`. Its 18 columns are `doc_id`, `corpus`,
`source_code`, `doc_type`, `title`, `date`, `year`, `date_precision`,
`date_kind`, `language`, `provenance`, `document_url`, `source_url`,
`mime_type`, `sha256`, `local_path`, `entity_scheme`, `entity_value`; every
other field is `extra`. A blank optional value is absent; an undated row has
no `date_precision` nor `date_kind`. No old name is mapped any more: in
this split `bank_code` and `pdf_url` are ordinary fields of `extra`, and
`local_path` is kept as given (the disk source never gets there with such a
row: since contract 1.1.0 it is refused `legacy_key`, below).
`tests/test_cb_corpus.py::test_the_disk_and_the_vault_sources_give_the_same_payload`
compares it with a vault row written out by hand in the test, and
`tests/test_payload.py` checks the split against the contract's golden rows.
A manifest row the contract refuses (`vault_contract.check_row` without
vocabulary, `transition=False` as the vault ingests: the vault keeps no row for
it) is skipped, never read, as the vault skips it: the run warns once with
their number and the first doc_ids and reasons (also when it stops early), and
the run report counts them (`docs_refused_contract`, the first doc_ids in
`refused_contract_doc_ids`: 20 by default, `DATA_ORCHESTRATOR_REPORT_DOC_IDS`;
every reason with its count in
`refused_contract_reasons`; `runs.extra`) and as failures for the outcome,
like `docs_bad_local_path` (a run where they pass the degraded ratio is
`degraded`, exit 3). `--count-only` leaves them out of its count and says how
many there are. A row in the pre-contract form (`bank_code`, `pdf_url`,
`local_path` under `data/`, no `corpus`) is such a row (`missing_corpus`,
`missing_source_code`, `bad_local_path`, `legacy_key`): nothing is mapped.
Since contract 1.1.0 a row carrying `bank_code` or `pdf_url` is refused
`legacy_key` whatever its value, even beside its contract key (`source_code`,
`document_url`), as the vault refuses it. Migrate the manifest first
(central-bank-corpus `migrate-contract`). Every row is checked
against `--corpus`: a value that is not a corpus code (`[A-Za-z0-9_.-]{1,32}`,
`vault_contract.CODE_PATTERN`) is a usage error, exit 2, before any work.
**Change with the contract:** `date` and `year` are taken as the row gives
them; this source used to cut a `date` with a time to its day
(`2015-03-01T09:00:00Z` -> `2015-03-01`) and read a `year` of digits as an
integer, the way the vault's columns cast them. The contract refuses such
rows (`bad_date`, `year_mismatch`): the vault has none, and the producers
check their rows against it in CI. Then the rules of the vault
source apply (see [Qdrant payload](#qdrant-payload)): columns win over
`extra`, an absent or null field is left out, the keys of `extra` dropped by
design (`DATA_ORCHESTRATOR_PAYLOAD_DROP_EXTRA_KEYS`) are left out silently, a
key the engine writes itself is left out and counted. `corpus` is the row's
own, else the run's `--corpus` (default `central-bank`, the vault's name for
this corpus). This source adds one key, `metadata_source`:

| field | indexed document (`metadata_source: manifest`) | path-only document (`metadata_source: path`) |
|-------|-------|-------|
| `doc_id` | manifest | filename stem |
| `corpus` | manifest, else `--corpus` | `--corpus` (default `central-bank`) |
| `source_code` | manifest | bank folder (`ecb`) |
| `doc_type` | manifest | doc-type folder, upper-cased (`C1`) |
| `year` | manifest | year folder (`2019`; absent for a non-numeric folder) |
| `date` | manifest | `<year>-01-01` (absent without a year) |
| `date_precision` | manifest (absent on an undated row) | `year` (absent without a year) |
| `title`, `document_url`, `source_url`, `mime_type`, `sha256`, `language`, `provenance`, `date_kind`, `entity_scheme`, `entity_value` | manifest, each absent when the row has none (a blank one is none) | absent (never an empty string) |
| any other field of the row (`date_source`, `repec_handle`, …) | manifest (`extra`) | — |
| `metadata_source` | `manifest` | `path` |
| `filename`, `page`, `chunk_number`, `ingestion_date`, `text` | engine | engine |

The selection flags of this source (`--banks`, `--doctypes`, `--groups`,
`--year-min`/`--year-max`) still read the **path** (bank, doc-type and year
folders), while the payload values come from the manifest: a document filed
under `2015/` whose manifest says `year: 2014` is selected by
`--year-min 2015` and ingested with `year: 2014` (the same for a doc type or a
bank that differ from their folder).

A path-only date follows the central-bank corpus convention: a coarse date
is the first day of its period, and `date_precision` says so (the engine
derives the interval of the whole year).

**Migration note.** Before this vocabulary, this source wrote `source`,
`bank_code`, `publication_date`, `url`, `doc_type_label`, `doc_group`, `ext`,
`rel_path` and `date_granularity` (and empty strings for a path-only
document's title, URL and checksum). Points written that way keep these keys
until the document is ingested again (`--no-resume`, or a fresh
`--collection`); `sync-metadata` does not rename them. The run report and the
vault ledger no longer read `bank_code`: a payload without `source_code` is
counted under its corpus.

### The company (SEC EDGAR) corpus: from the vault

Company filings, produced by the
[`company-corpus`](https://github.com/MyOpenFund/company-corpus) project,
reach the RAG through the vault like central-bank documents: its manifests
follow the `documents` contract (`corpus: company`, `source_code: sec`, the
form family as `doc_type`, `entity_scheme: cik`, `entity_value` the 10-digit
CIK, `local_path` the rendered PDF, else the cleaned text), the vault ingests
them, and

```bash
data-orchestrator vault --corpus company
```

ingests the vault's selection into `company-e5b-v1`, reading each file under
`COMPANY_CORPUS_ROOT`. The payload is the vault row's
([Qdrant payload](#qdrant-payload)), `entity_scheme` / `entity_value` and the
company extras (`cik`, `sec_form`, `accession`, `company`, `ticker`, ...)
included. The former disk connector (`data-orchestrator company`, the
`company` extra, `company_corpus.rag.iter_items`, its payload with `source:
company-corpus`, `url`) is gone: `data-orchestrator company` exits 2 naming
the command above.

### Path resolution convention

The vault stores each document's `local_path` **relative to its corpus data
root** (the directory holding `manifest/` and `raw/`; documents contract), e.g.
`raw/ecb/A1/2022/<doc_id>.pdf` or `raw/0000320193/A1/2024/<doc_id>.pdf`. A
corpus's root env key (`CB_CORPUS_ROOT`, `COMPANY_CORPUS_ROOT`) names that
directory on this machine, and `routing.resolve_local_path(corpus,
local_path)` joins the two.

Before joining, a `local_path` the contract refuses
(`vault_contract.is_bad_local_path` with `transition=False`, as the vault
ingests: absolute, a `..`, `.` or empty segment, a drive or a scheme, a
backslash, an invisible character, a leading `~`, a leading `data/`) raises
`routing.BadLocalPath`: it is never joined with a root nor read. So does a
plain path that leaves the root all the same through a **symbolic link**: the
joined path is resolved (every link followed, the root's own included) and
must stay under the resolved root. The vault source skips such a row, names
its `doc_id` in a warning, counts it in the run report (`docs_bad_local_path`,
`bad_local_path_doc_ids`; [Run reports](#run-reports-and-exit-codes)) and
counts it as a failure for the outcome; the probe skips it. The vault refuses
a bad path at ingestion, so the first check is a second line of defence; the
link check is this machine's.

**Before the first run on a machine** (the NAS, the Mac), check that nothing
under `CB_CORPUS_ROOT` or `COMPANY_CORPUS_ROOT` that the vault's paths go
through (`raw/` and its subfolders) is a symbolic link pointing outside that
root, e.g. `find "$COMPANY_CORPUS_ROOT/raw" -maxdepth 3 -type l`: every
document behind such a link would be skipped and the run degraded. A root that
is itself a link is fine (it is resolved first).

Central-bank manifests gave `local_path` relative to the repository
(`data/raw/...`) before the contract. That form is refused now (no `data/` is
stripped): every live vault row was rewritten on the contract (0 rows under
`data/` on 2026-11-20).

## Usage

Each block below is one command, without comments, so that it can be pasted
as it is into any shell (the default shell of macOS, zsh, does not read a `#`
typed at the prompt as a comment).

Vault-selected ingestion (recommended): whatever the vault knows for a corpus
that the target collection has not ingested yet.

```bash
data-orchestrator vault --corpus central-bank
```

Disk-fallback ingestion, bypassing the vault entirely (its own file ledger, no
`rag_ingestions` read or write) — useful before a corpus is onboarded to the
vault.

```bash
data-orchestrator cb_corpus --banks ecb --no-vault
```

The company (SEC EDGAR) corpus, from the vault — routes to `company-e5b-v1`
via `routing.collection_name(--corpus)`; files are read under
`COMPANY_CORPUS_ROOT`.

```bash
data-orchestrator vault --corpus company
```

Facts probe: fills `has_text_layer` / `page_count` for a corpus's documents
(feeds the OCR policy; safe to re-run, only unprobed rows are touched).

```bash
data-orchestrator probe --corpus central-bank
```

Date metadata of the ingested documents, vault to Qdrant, without
re-embedding (dry run; `--write` applies; see
[Correcting dates in Qdrant](#correcting-dates-in-qdrant-sync-metadata)).

```bash
data-orchestrator sync-metadata --collection central-bank-e5b-v1
```

Forced OCR instead of the engine's page-by-page detection (needs
`TESSDATA_PREFIX`, see the flags below).

```bash
data-orchestrator vault --corpus central-bank --ocr always
```

> Equivalent module form: `python -m data_orchestrator.cli vault --corpus central-bank`

### Useful flags

| flag | meaning |
|---|---|
| `--corpus NAME` | vault corpus to operate on (default `central-bank`; `company` for the company corpus) |
| `--no-vault` | (cb_corpus) use the local JSON-lines file ledger instead of the vault's `rag_ingestions`, and append the run report to `state_dir()/runs.jsonl` (no vault state is read or written). Rejected for the `vault` source and for `probe` (exit 2): both always read and write the vault |
| `--ledger PATH` | (cb_corpus with `--no-vault`) path of the file ledger (default `state_dir()/<collection>.jsonl`); ignored by the other ingest runs, rejected for `probe` (exit 2) |
| `--ocr auto\|always\|never` | OCR for scanned pages, passed to the engine as is. `auto` (default): OCR runs page by page where needed, **only if the host switches are on** (`ENABLE_OCR=true` and `TESSDATA_PREFIX` set), else no OCR; `always`: every page, whatever `ENABLE_OCR` says — `TESSDATA_PREFIX` must be set, else the run is refused before any work (forced OCR needs the language data: set it, or use `auto` or `never`; exit 2); `never`. The run's `OCR:` line says which applies |
| `--collection NAME` | target Qdrant collection (default: `{corpus}-{model_tag}-v1` via `routing.collection_name` — the same resolution for every source, using `--corpus` or the source's implied corpus) |
| `--source-codes a,b` | (vault source) only these `source_code` values, e.g. `ecb,fr` |
| `--doctypes C1,A3` | only these doc-type codes of the corpus (cb_corpus, vault): a code means something only with its corpus (`A1` is a rate decision in `central-bank`, a 10-K in `company`) |
| `--languages en,fr` | (vault source) only these language codes |
| `--year-min` / `--year-max` | inclusive year bounds (cb_corpus, vault) |
| `--banks a,b` | (cb_corpus source) only these bank codes |
| `--groups a,b` | (cb_corpus source) comma list of doc groups, e.g. A,C |
| `--root PATH` | (cb_corpus source) corpus root override |
| `--limit N` | stop after N newly ingested docs; for `probe`, probe at most N documents; for `sync-metadata`, examine the first N ledger documents in `doc_id` order (`N >= 1`; `0` or negative is an error, exit 2) |
| `--progress-every N` | progress line every N documents (`N >= 1`, default 25) |
| `--no-resume` | (disk sources) ignore the resume ledger, re-ingest everything. **Rejected for the `vault` source** (exit 2), by design: its resume *is* the `documents`/`rag_ingestions` anti-join, and a policy change means a fresh collection — re-ingest into a new `--collection` name. Re-ingesting into the same collection would leave stale chunks behind when the new chunking yields fewer chunks than the points already stored. Also rejected for `probe` (exit 2), which is resumable by construction |
| `--count-only` | just count matching documents, do not ingest (cb_corpus, vault; **rejected for `probe`**, exit 2); for `vault`, also the unsupported documents and their per-source split. A NULL `source_code` is counted under the corpus name — the label the run report's `sources[]` uses (`core.source_label`) — and a NULL `doc_type` as `unknown`. No run report, no `Qdrant:` line |

The flags of `sync-metadata` (`--write`, `--doc-ids`, `--csv`,
`--page-size`, `--doc-batch`, `--include-title`, `--include-links`,
`--create-indexes`) are
listed in [its section](#correcting-dates-in-qdrant-sync-metadata); any of
them on another command is an error (exit 2).

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

Every ingest run (`vault`, `cb_corpus`) writes a telemetry row to
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
`exit_code` 1 and the reason under `fatal`. The report is written first; then
the counts are printed under `Interrupted by a fatal error. Before it:`
(`Done.` opens the summary of a run that reached its end, and only that one),
then the `error: fatal:` line. When standard output is itself what failed (the
command piped into a reader that has gone, e.g. `| head`), the counts cannot
be printed: the report is written all the same and the exit code stays 1. A
fatal error before the first document reports zeros and prints no summary. **No report**
is written by `--count-only`, by `probe`, by `sync-metadata`, or by a run stopped by a pre-flight
error (exit 1 or 2, before any work). Exit codes are honest:

- `0` **ok**;
- `3` **degraded** when (a) nothing was ingested while attempts failed, or
  (b) `docs_failed / docs_attempted` exceeds the ratio, where
  `docs_attempted = ingested + empty + failed` (resume skips excluded). The
  ratio is `DATA_ORCHESTRATOR_DEGRADED_FAILURE_RATIO`, default `0.10`; `0` =
  any failure degrades, `1` = rule (b) off. `docs_unsupported` never degrades;
  the documents the `vault` source skipped for a bad `local_path`
  (`docs_bad_local_path`, below) and the documents the `cb_corpus` source
  skipped for a manifest row the contract refuses (`docs_refused_contract`,
  below) count as failed in (a) and (b), so a run where they pass the ratio is
  at least `degraded`;
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
  `TESSDATA_PREFIX` (forced OCR needs the language data, whatever the
  selection holds).

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
a disk source's successful run has a NULL `extra` (unless it skipped refused
manifest rows, below). A fatal run's reason is
`extra->>'fatal'` (capped at 500 characters, the key central-bank-corpus
writes); rows written before 2026-11 carry it as `extra->>'error'`, so a reader
covering both uses `coalesce(extra->>'fatal', extra->>'error')`. A `vault` run
that skipped documents for their `local_path` ([Path resolution
convention](#path-resolution-convention)) carries `docs_bad_local_path` (how
many, on success and on a fatal error alike) and `bad_local_path_doc_ids` (the
first `doc_id`s: 20 by default, `DATA_ORCHESTRATOR_REPORT_DOC_IDS`) in `extra`,
and says so once on stderr (`warning: N document(s) skipped, never read: …`);
they are absent from a run without any. Likewise a
`cb_corpus` run that skipped documents for a manifest row the documents
contract refuses ([cb_corpus](#cb_corpus--the-central-bank-corpus)) carries
`docs_refused_contract` (how many, on success and on a fatal error alike),
`refused_contract_doc_ids` (the first `doc_id`s, as many as for
`bad_local_path_doc_ids`) and `refused_contract_reasons` (each
`vault_contract` reason with the number of rows it refused; contract 1.1.0
added `legacy_key`, so a reader of these reports may meet a reason a 1.0.0
list does not name) in `extra`, with the same stderr warning.
`totals.docs_failed` stays the engine's count.

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
| `document_url`, `source_url`, `mime_type` | vault (`documents`): the addresses a citation links back to (`document_url`: the document itself, whatever its type; `pdf_url` until 2026-11), and the document's type |
| `date_precision`, `date_kind`, `entity_scheme`, `entity_value` | vault (`documents`), columns of the documents contract, after every earlier column (`date_precision`: `day`/`month`/`year`, absent = not stated; `date_kind`: `meeting`/`publication`/`period`; `entity_*`: the company a document is about, e.g. `cik` / `0000320193`). Central-bank rows carried the first two in `extra` before the contract, under the same names and values: the points ingested then need no rewrite, with two exceptions by the contract: the precision of an undated row is dropped (7 public central-bank rows) and a blank title is absent (110 public rows; `title` is compared only with `sync-metadata --include-title`). See [SYNC-METADATA.md](SYNC-METADATA.md#after-the-documents-contract-vault-12ffd11-or-later). A NULL one removes the same key of `extra`, like any column |
| `date_source`, any other key of `extra` | vault `extra`, copied as given, except the keys dropped by design (below) |
| `filename`, `page`, `chunk_number`, `text` | engine (chunking output; `page` only for formats with pages) |
| `ingestion_date` | engine — one ISO timestamp per document, shared by all of its points |
| `date_start`, `date_end` | engine, derived from `date` and `date_precision` (absent without a readable `date`): the first and the last day the document is dated to, RFC 3339 at midnight UTC (`2016-07-01T00:00:00Z` / `2016-07-31T00:00:00Z` for a `month` date of July 2016; a precision that is absent or not one of the three is read as a day) |

A NULL column is left out (and removes the same key of `extra`: a column
always wins).
An empty or whitespace-only value of any column but the identity and the codes
(`doc_id`, `corpus`, `source_code`, `doc_type`, `language`, `provenance`)
counts as NULL, as the contract reads it (`split_row`); a vault row stored
before the contract may still hold one (2,294 central-bank rows have
`source_url: ""`, which is no link). A key of
`extra` whose value is null is left out (not counted, even a reserved one). The keys of `extra` named by
`DATA_ORCHESTRATOR_PAYLOAD_DROP_EXTRA_KEYS` (default `html_path`, a path on
the producer's machine; `alt_urls` and `alt_source_urls`, lists of other
addresses of the document; `local_path`) are **dropped by design**: never
sent, never counted as reserved keys. The `cb_corpus` disk source builds the
same payload from the manifest (see [its section](#cb_corpus--the-central-bank-corpus)).
Points ingested before the links were sent (no `document_url`, an
`html_path`) are aligned by `sync-metadata --include-links`.

**`pdf_url` became `document_url` (2026-11).** The vault source and
`sync-metadata` read `document_url` only (every vault has run the rename), and
a manifest row's address is its `document_url` only. Since the contract's
transition cleanup `pdf_url` and `bank_code` are ordinary keys for the vault
source and the payload functions: a vault row that still held one in `extra`
would carry it onto its points as given. No new one can appear: since contract
1.1.0 the vault refuses a manifest row carrying either (`legacy_key`), and so
does the `cb_corpus` disk source.

This is rag-engine's payload schema v2 (`rag_engine.schema`): schema v1, the
payload this repo assembled itself before 2026-11, plus `date_start` and
`date_end`, written after every key of v1. Points written under v1 (the whole
of `central-bank-e5b-v1` before `sync-metadata`) have neither; a search in
the engine's `auto` mode keeps reading their `date` until they get them (see
[Correcting dates in Qdrant](#correcting-dates-in-qdrant-sync-metadata)).
What was shown about the switch: with
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
document without pages used to be kept; see below). Since the schema v2 pin
the parity test checks the captured v1 points plus exactly `date_start` and
`date_end` (computed by the test itself, not by the engine), and nothing
else: same ids, same vectors, every other key with its value and position.
Another version of one of
those libraries, or another device, is outside what was checked:
[MIGRATION.md](MIGRATION.md) says how to keep the versions and how to validate
on real data. The key is `chunk_number`, not `chunk_index`.

### Reserved payload keys

The engine writes `doc_id`, `filename`, `chunk_number`, `ingestion_date`,
`text` and `page` itself, derives `date_start` and `date_end`, and
**refuses** metadata that sets one of them (a `doc_id` equal to the
document's is accepted). Policy of this repo, applied where metadata it does
not own enters:

- **vault source**: such a key in `documents.extra` is left out of the payload
  (the vault's own columns are never dropped; `doc_id` always comes from its
  column);
- **cb_corpus source**: such a field in a manifest row (outside the vault's
  columns, so in what the vault would store as `extra`) is left out, as in the
  vault source.

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

## Correcting dates in Qdrant (sync-metadata)

The vault takes corrected metadata at its next ingestion of the manifests
(its upsert rewrites `date`, `year`, `title` and `extra`), but a document
already recorded in `rag_ingestions` is never offered to the engine again, so
its points keep the payload they were written with. `sync-metadata` brings
the date metadata of those points in line with the vault **without
re-embedding**:

1. it reads the documents `rag_ingestions` records for the collection and the
   corpus (`doc_id` order), each joined with its `documents` row;
2. it builds the metadata an ingestion would write today **through the
   ingestion code path**: `payload.row_to_payload` (the vault columns
   over `extra`, NULL columns dropped, the keys dropped by design and the
   engine's reserved and derived keys dropped from `extra`), then rag-engine's `build_payload`, which derives
   `date_start` / `date_end` from `date` and `date_precision`;
3. it reads the owned keys of **every point** of these documents (a scroll
   filtered on `doc_id`, `--doc-batch` documents per filter, `--page-size`
   points per request, payload restricted to the owned keys, never a vector);
4. where a point differs, with `--write`, it calls rag-engine's
   `QdrantStore.set_payload(collection, doc_id, fields)` with
   `date_fields(date, date_precision, year=year)` plus `date_kind` and
   `date_source` (a `None` removes a key: a `date_kind` the vault no longer
   has, a withdrawn date and its interval). The fields are checked to be exactly what an ingestion would
   write before they are sent.

**Owned keys**: `date`, `year`, `date_precision`, `date_kind`, `date_source`,
`date_start`, `date_end`. `date_source` (where the producer read the date:
`title`, `url`, …) is in `extra`, so a fresh ingestion copies it: it is owned
so that a corrected point carries exactly the date keys a new ingestion of
the same row would write. Nothing else is compared or written: no vector (not read, not
sent), no point id, no other payload key, no point created or deleted, no
`rag_ingestions` or `runs` row. A key absent from the vault (a NULL column, a
key not in `extra`) must be absent from the points; a stored null counts as
absent. So does an explicit null in `extra` (`"date_kind": null`): an
ingestion leaves it out, and `set_payload` cannot store one anyway (Qdrant
drops a key set to null, and every filter reads a null as a missing key). A
point holding such a null (written before nulls of `extra` were left out) and
one without the key are therefore both `unchanged`.

**`title` is not owned by default.** The vault may hold corrected titles as
well (its upsert rewrites them), but this command exists for the dates: a
title change is shown to every reader of a citation, it is not part of the
policy-dates correction, and mixing it into the dry run would hide the date
differences among title ones. `--include-title` adds `title` to the compared
and written keys, explicitly, for a run whose CSV has been read for titles.

**After the documents contract** `date_precision` and `date_kind` are columns
of the vault (they were keys of `extra`, same names, same values), read from
there; the first dry run after the vault deploy is expected to find nothing to
write but the stated rows ([SYNC-METADATA.md](SYNC-METADATA.md#after-the-documents-contract-vault-12ffd11-or-later)).

**The links are not owned by default either.** `--include-links` adds
`document_url`, `source_url` and `mime_type` (set from the vault's columns; a NULL
or blank column removes the key) and the keys of `extra` dropped by design
(`DATA_ORCHESTRATOR_PAYLOAD_DROP_EXTRA_KEYS`, default `html_path`,
`alt_urls`, `alt_source_urls`, `local_path`, `pdf_path`, `text_path`,
`primary_path`, `submission_path`): an ingestion never sends these,
so they are expected absent and removed from the points that still carry
them, the way an absent owned key is removed. They appear in the summary and
the CSV like the other keys (a list as JSON). A document written for its
links gets its full date set (re)written too: the write is always every owned
key. No point carries `mime_type` before this, so the first run writes every
ledger document that has one in the vault. This is how a collection
ingested before the links were sent is aligned (see
[SYNC-METADATA.md](SYNC-METADATA.md#aligning-the-links-include-links)).

**Every point is read.** rag-engine copies the same metadata into every point
of one ingestion (`ingest_document` hands one mapping to `build_payload` for
each chunk), so the points of a document agree unless something else wrote
them: a stale point left by an earlier ingestion that made more chunks, an
interrupted correction. The read is the guard: a document whose points do not
all carry the same values is reported (`points disagreeing`, a `points
disagree` note in the CSV, its values listed) and `--write` corrects all of
its points (`set_payload` filters on `doc_id`). The read costs the owned keys
of each point, about 33,600 small payloads for `central-bank-e5b-v1`.

**Payload indexes.** `--write` first creates **every** payload index of
rag-engine (`PAYLOAD_INDEXES`: `doc_id`, `source_code`, `doc_type`,
`language`, `corpus`, `year`, `date`, `date_start`, `date_end`) that the
collection lacks (`ensure_payload_indexes`; on production, which has the
seven of schema v1, the two date ones), then corrects the points: the
order of the rag-engine README, which states that creating the two indexes
before the points have their interval is safe (in `auto` mode a search keeps
the legacy condition on `date` as a second way in, so a half-upgraded
collection loses no document). `--create-indexes` does only that step and
exits, for an operator who wants it separate, or for a collection in Qdrant's
strict mode without an index on `doc_id` (there the reads themselves are
refused). A dry run never creates anything; it says which indexes are
missing.

**Dry run by default**, with a read-only Qdrant key: it reads the vault and
Qdrant and writes the CSV only. `--write` needs a key allowed to write; with
a read-only key the first write is refused (`QdrantAuthError`, exit 1, nothing
written). **Idempotent and resumable**: a document is either corrected or not
(one `set_payload` per document, applied when it returns); the next run
recomputes everything from the vault and Qdrant and writes only what still
differs, so a second `--write` writes 0 documents.

**Never alongside an ingestion of the same collection.** `set_payload` counts
then writes, and a batch of an ingestion that started before the correction
writes whole payloads with the old metadata. The two now exclude each other
through a Postgres advisory lock on the collection, taken by the run's vault
connection and released when it closes (or the process dies): an ingestion
holds it shared (ingestions do not exclude each other: the `vault` source,
and `cb_corpus` in vault mode, with or without `--no-resume`),
`sync-metadata --write` holds it exclusive. A dry run holds it shared: it
only reads, so it neither stops an ingestion from starting nor waits for one
(a nightly ingestion is never lost to a dry run), but it does not run beside
a `--write`. A dry run made while an ingestion writes may report, for the
documents that ingestion rewrites, a state already gone: run it again
afterwards if that matters. Whichever comes second stops with `error: fatal:
CollectionBusyError: … holds collection '<name>' (vault advisory lock 7468;
held by pid 4242 (ShareLock, user …, application data-orchestrator@<host>,
client …, session started …)); wait for it to end, then run again. …` and
exit 1. The holder's user, application (`data-orchestrator@<host>`: the
vault connection names its machine), client address and session start come
from `pg_stat_activity`; a role without `pg_read_all_stats` sees the user and
the application of another role's session but not its client nor its start,
and the message says so. **The lock only knows runs of this version**: an
ingestion started by an older checkout (another machine), or a disk-source
run with `--no-vault`, takes no lock; check that none runs before `--write`.

**A stale lock.** A run whose machine went to sleep or lost its network
stops (see below), but the server does not know: it keeps that session, and
its lock, until its own TCP keepalive ends it (by default after about two
hours on Linux), and every run until then is refused although nothing runs.
Clear it once the refusal's holder is known to be dead (its `application`
names the machine: check there that no `data-orchestrator` process is left,
`pgrep -fl 'data[-_]orchestrator'`): with the vault's admin role, the
statement the refusal prints, `SELECT pg_terminate_backend(<pid>);`. Or wait
for the server to drop the session. All the holders, with their sessions:
`SELECT l.pid, l.mode, a.application_name, a.client_addr, a.backend_start FROM
pg_locks l LEFT JOIN pg_stat_activity a USING (pid) WHERE l.locktype =
'advisory' AND l.classid = 7468;`. **Recommended on the vault's Postgres**
(the NAS): server-side keepalives, so that a dead client's session goes
within minutes, e.g. `tcp_keepalives_idle = 60`, `tcp_keepalives_interval =
10`, `tcp_keepalives_count = 6` in `postgresql.conf` (or `ALTER SYSTEM SET`,
then a reload).

**The lock is checked during the run, not only at its start.** It lives in
the run's vault session, which a server restart, `pg_terminate_backend` or a
dropped network link (a laptop asleep) ends silently, releasing the lock.
So `sync-metadata` checks, on its own session, that it still holds the lock
(`pg_locks` for its backend) before each batch it reads and before each
document it writes, and an ingestion before each document. A lost session
stops the run at the next check: the document in flight (a `set_payload`
already sent, a document being ingested) may still be written, then nothing
more: `error: fatal: VaultSessionLost: … the run stops here and can be run
again`, exit 1 (for an ingestion, a failure to record a document because the
session is gone is this same fatal error, not one failed document; its points
are written again by the next run). Both commands are resumable: run the
same command again.

How fast a dead link is noticed: while the connection is idle, by the TCP
keepalives (`DATA_ORCHESTRATOR_PG_KEEPALIVES_*`, about 60 s by default);
while a check or a ledger write waits for an answer, keepalives do not fire,
and `DATA_ORCHESTRATOR_PG_TCP_USER_TIMEOUT_MS` (30 s by default) bounds the
wait instead, on Linux only: on macOS libpq ignores it and the wait is the
system's retransmission limit. A server-side `statement_timeout` would not
help: on a dead link the server never receives the statement.

### Options

| flag | meaning |
|---|---|
| `--corpus NAME` | the corpus of the ledger rows (`rag_ingestions.corpus`; default `central-bank`) |
| `--collection NAME` | the collection (default: the routed `{corpus}-{model_tag}-v1`) |
| `--write` | apply the corrections (creates every missing payload index of rag-engine first). Without it, nothing is written to Qdrant |
| `--csv PATH` | the CSV of differences (default `<state dir>/sync-metadata-<collection>-<UTC time>.csv`, `DATA_ORCHESTRATOR_STATE_DIR` or `<repo>/state`) |
| `--doc-ids a,b` | only these ledger documents (a trial); an id not in the ledger for that collection and corpus is listed in a warning |
| `--limit N` | only the first N ledger documents, in `doc_id` order |
| `--page-size N` | points per Qdrant scroll request (default 256) |
| `--doc-batch N` | documents per Qdrant read, one `doc_id` match-any filter (default 100) |
| `--include-title` | also compare and write `title` (off by default, see above) |
| `--include-links` | also compare and write `document_url`, `source_url`, `mime_type`, and remove the keys of `extra` dropped by design (off by default, see above) |
| `--create-indexes` | only create the missing payload indexes, print them, exit; alone (with `--corpus`/`--collection`) |
| `--progress-every N` | a progress line every N documents (default 25) |

The selection flags of the ingest runs (`--source-codes`, `--doctypes`,
`--languages`, `--year-min`/`--year-max`, `--banks`, `--groups`,
`--root`) and `--count-only`, `--no-vault`, `--no-resume`, `--ledger` are
refused (exit 2). The Qdrant and vault settings are the ones of an ingest run
(`DATABASE_URL`, `QDRANT_HOST`, `QDRANT_PORT`, `QDRANT_API_KEY`,
`QDRANT_HTTPS`); `CB_CORPUS_ROOT` is not needed (no file is read).

### Actions, counts and CSV

Each ledger document gets one action:

| action | meaning | written by `--write` |
|---|---|---|
| `unchanged` | dated in the vault, every point already carries the expected owned keys | nothing |
| `updated` | dated in the vault, at least one point differs | its points |
| `undated` | no `date` in the vault (`date`, `date_start`, `date_end` must then be absent; a search finds the document only with `include_undated`) | its points, when they differ |
| `missing-in-qdrant` | recorded in `rag_ingestions`, no point in the collection | nothing |
| `missing-in-vault` | no `documents` row, or the row has `deleted_at` set | nothing (its points are left as they are) |

The CSV has one row per document and differing key: `doc_id`, `action`,
`field`, `qdrant_value`, `vault_value`, `note` (a value is empty when the key
is absent, a string as it is, anything else as JSON; when the points of a
document disagree, `qdrant_value` is the JSON list of their values). A
`missing-in-*` document has one row with an empty `field`. An `unchanged`
document has none.

Example (a dry run on a small collection):

```text
Config: .env = /path/to/data-orchestrator/.env
Qdrant: <host>:<port> (api key: set)
→ sync-metadata: collection 'central-bank-e5b-v1', corpus 'central-bank' (dry run: nothing is written to Qdrant)
  keys compared: date, year, date_precision, date_kind, date_source, date_start, date_end
  payload indexes missing: date_start, date_end (--write creates them first, or --create-indexes)
────────────────────────────────────────────────────────────
Done (dry run: nothing was written to Qdrant).
  documents          : 6
  unchanged          : 0
  updated            : 5
  undated            : 1
  missing-in-qdrant  : 0
  missing-in-vault   : 0
  documents to write : 5
  points read        : 22
  points disagreeing : 0 document(s)
  keys changing      : date x2, year x1, date_precision x5, date_kind x5, date_start x5, date_end x5
  time               : 0.0s
  CSV                : /path/to/data-orchestrator/state/sync-metadata-central-bank-e5b-v1-20261118T091330Z.csv (23 row(s))
```

and the first rows of its CSV:

```text
doc_id,action,field,qdrant_value,vault_value,note
au-e1-2016-08-smp,updated,date_precision,,month,
au-e1-2016-08-smp,updated,date_kind,,period,
au-e1-2016-08-smp,updated,date_start,,2016-08-01T00:00:00Z,
au-e1-2016-08-smp,updated,date_end,,2016-08-31T00:00:00Z,
gb-a3-2004-minutes,updated,date,2004-01-01,2003-12-10,
gb-a3-2004-minutes,updated,year,2004,2003,
```

With `--write` the summary says `Done.`, `documents written` and `points
written` instead of `documents to write`, and lists the indexes it created.
`keys changing` counts documents per key. A warning follows the summary for a
document whose points disagree, and for one whose point count changed between
the read and the write (an ingestion that did not take the lock).

**Exit codes**: `0` the run reached its end (dry run or write); `1` a fatal
error, said as `error: fatal: <Type>: <message>` with what to do: the vault
or Qdrant unreachable, a refused key (`QdrantAuthError`, with which key the
step needs), the collection lock held (`CollectionBusyError`), a collection
that does not exist, a strict-mode collection without the `doc_id` index
(`MissingIndexError`, with the `--create-indexes` command); the CSV of the
documents done before the error is kept. `2` a usage or configuration error.

The validation procedure for the production collection (a scratch copy
first, vectors byte-identical before and after, then production) is in
[SYNC-METADATA.md](SYNC-METADATA.md).

## Deploying with the documents contract

This version follows the vault's `documents` contract 1.1.0 (vault `12ffd11` or later
for the schema; the pinned `vault-contract` is 1.1.0, whose `legacy_key` the
`cb_corpus` disk source applies).
Old and new versions of the two do not mix:

| | vault before the contract (`2d0457e`) | vault on the contract (`12ffd11` or later), after its first ingestion |
|---|---|---|
| **this repo before** (`81ead10`) | works | **ingestion works** until the vault's transition cleanup (`2859fa6` drops the transitional unique index on `rag_ingestions (doc_id, collection)` it relies on: after it, its ledger writes fail) but its payloads lose `date_precision` / `date_kind` (moved out of `extra`): **no ingestion and no `sync-metadata` with it** (a `--write` would strip both keys from every point) |
| **this repo now** | **refused** before any Qdrant write: the vault source (and its `--count-only`), `sync-metadata` and `cb_corpus` with the vault ledger read the catalog first (`vault.require_contract_schema`: the four contract columns, the key `(corpus, doc_id, collection)`) and stop with `VaultSchemaTooOld`, exit 1 | works |

So: deploy the vault and let its ingestion run once (the DDL train, then the
rows in contract form), then update this repository on every machine that runs
it, before any ingestion or `sync-metadata`. Between the two steps, run
neither: **nothing in the old version stops it**. Its `sync-metadata` never
writes `rag_ingestions` and its `doc_id` join still works on the new schema,
so an old `sync-metadata --write` runs cleanly and strips the keys (replayed on
the 39,664 public central-bank rows: `date_precision` lost on 14,659 rows,
`date_kind` on 2,555). The deploy order is the only protection. Then:

1. `git pull` and `pip install -e .` (it installs `vault-contract` at the
   pinned commit), then `python tools/check_engine_pin.py` (both pins);
2. `.env`: `CB_CORPUS_ROOT` and `COMPANY_CORPUS_ROOT` name each corpus's data
   root (the folder holding `manifest/` and `raw/`); `COMPANY_CORPUS_ROOT` was
   optional, it is now required to ingest the company corpus;
3. `sync-metadata` dry run on `central-bank-e5b-v1`: nothing to write except
   the rows stated in [SYNC-METADATA.md](SYNC-METADATA.md#after-the-documents-contract-vault-12ffd11-or-later);
4. a pilot of the company corpus into a scratch collection (it creates the
   collection and its `rag_ingestions` rows; remove both after):
   `data-orchestrator vault --corpus company --collection zz-company-pilot --limit 5`.

**Legacy removed.** The `company` disk source (`data-orchestrator company`,
`--ciks`, `--prefer`, the `company` extra and `company_corpus.rag.iter_items`)
is gone; points written by it (payload `source: company-corpus`, `url`,
`publication_date`) are not in the vault's vocabulary and are not rewritten:
re-ingest the company corpus from the vault into a fresh collection. A vault
row of another corpus with the same `doc_id` no longer counts as ingested for
this one.

**Transition code: removed** (the contract's transition cleanup, with the
vault's: vault `2859fa6`, which ingests with `transition=False` and drops the
transitional `rag_ingestions (doc_id, collection)` index). Gates measured on
2026-11-20 before the removal: the query below returned 0, no live row had a
`local_path` under `data/`, the last ingestions of both corpora logged no
`corpus_missing` / `legacy_alias` / `legacy_data_prefix` notice, and no point
of `central-bank-e5b-v1` carried `pdf_url` (2026-11-19).

```sql
SELECT count(*) FROM documents WHERE deleted_at IS NULL AND ((date_precision IS NULL AND extra ? 'date_precision') OR (date_kind IS NULL AND extra ? 'date_kind') OR (entity_scheme IS NULL AND extra ? 'entity_scheme') OR (entity_value IS NULL AND extra ? 'entity_value'));
```

Before deploying this version, a second gate, on the vault: no live row keeps
a pre-contract name in `extra` (the vault source reads `pdf_url` and
`bank_code` as ordinary keys, so a row that still had one would carry it onto
its points as such; the vault on contract 1.1.0 refuses any new one). Expected: 0.

```sql
SELECT count(*) FROM documents WHERE deleted_at IS NULL AND extra ?| array['pdf_url','bank_code'];
```

On deploy day, with this version installed and before any ingestion: re-run
both gate queries (each 0), then two `sync-metadata` dry runs on
`central-bank-e5b-v1` (no `--write`; each expects `documents to write : 0`):

- `data-orchestrator sync-metadata --collection central-bank-e5b-v1`: the
  dates this version builds from the vault's contract columns (`date`,
  `year`, `date_precision`, `date_kind`, `date_source` and the interval) are
  the ones on the points, so no `date_precision` / `date_kind` is lost now
  that only the columns are read;
- `data-orchestrator sync-metadata --collection central-bank-e5b-v1
  --include-links`: the links this version builds (`document_url` from its
  column only, `source_url`, `mime_type`) are the ones on the points, and no
  point carries a key of `extra` dropped by design. It does not look at
  `pdf_url` (no longer owned): its absence from the points is the
  2026-11-19 measurement above.

Any other result stops the deploy: nothing in this version maps the old names
back.

What went, and what replaces it:

- `payload.LEGACY_COLUMN_NAMES`: in the vault source and the payload
  functions (`payload.manifest_row_to_vault_row`, `row_to_payload`), `pdf_url`
  and `bank_code` are ordinary keys of `extra` (carried as given; the drop
  setting may name them); the `cb_corpus` disk source never reads them: since
  contract 1.1.0 a manifest row carrying either is refused `legacy_key`, as
  the vault refuses it;
- `payload.CONTRACT_COLUMNS_FROM_EXTRA`: a NULL `date_precision`,
  `date_kind`, `entity_scheme` or `entity_value` removes the same key of
  `extra`, like any column;
- `routing.TRANSITION_LOCAL_PATH_STRIP` (and `CorpusRoute.local_path_strip`):
  `vault_contract.is_bad_local_path` runs with `transition=False` in
  `routing.resolve_local_path`, so a leading `data/` is a bad local path
  (skipped, counted, never read);
- the transition mode of `vault_contract.check_row` in
  `sources/cb_corpus.py::_refusals` and of `split_row` in
  `payload.manifest_row_to_vault_row`: both pass `transition=False`, so the
  `cb_corpus` disk source skips a manifest row in the pre-contract form
  (`missing_corpus`, `missing_source_code`, `bad_local_path`, `legacy_key`) or
  any other row the contract refuses, never reads or maps it, and counts it in the
  run report (`docs_refused_contract`, `refused_contract_doc_ids`,
  `refused_contract_reasons`) and as a failure for the outcome; a `--corpus`
  that is not a corpus code is a usage error (exit 2);
- `sync_metadata.LEGACY_LINK_KEYS`: `--include-links` no longer owns
  `pdf_url` (neither compared nor removed);
- `vault.document_url_column` / `LEGACY_DOCUMENT_URL_COLUMN` /
  `document_url_select`: `document_url` is selected directly;
- the copy of the transitional index in `tests/integration/conftest.py`
  (`tests/test_contract.py` now checks the pinned vault DDL drops it).

The preflight is unchanged (`vault.require_contract_schema`). Deploy order
with the vault's cleanup: either is safe. This version needs no transitional
index (it writes `ON CONFLICT (corpus, doc_id, collection)`, as `2695b26`
did), and a vault on the contract stores every row in contract form whichever
mode it ingests with, so it reads the vault before and after its cleanup the
same way; the vault's cleanup in turn only needs every orchestrator to be at
`2695b26` or later. The `cb_corpus` disk source, unlike a vault still on
`transition=True`, refuses and skips a pre-contract manifest row: run it on
migrated manifests only.

## Testing

In the virtual environment of the installation, with the test tools
(`pip install -e ".[dev]"`). The unit tests (no network, no service):

```bash
python -m pytest -q
```

The integration tests (they need Docker):

```bash
python -m pytest -m integration -q
```

The contract test (`tests/test_contract.py`): the `vault-contract` installed
is the `contract/` directory of a vault checkout file for file, the vault
ingests with it, and the CI workflow checks out the vault at the commit
`pyproject.toml` pins. Skipped without a checkout, except under CI (`CI` set),
where a missing checkout fails:

```bash
DATA_ORCHESTRATOR_VAULT_CHECKOUT=~/myopenfund/vault python -m pytest -q tests/test_contract.py
```

The check that the rag-engine and the vault-contract installed are the pinned
commits:

```bash
python tools/check_engine_pin.py
```

Without Docker, the integration suite runs against services already started
on the machine, named by three variables: `DATA_ORCHESTRATOR_IT_DATABASE_URL`
(an empty throwaway Postgres database: the suite drops and re-creates its
tables), `DATA_ORCHESTRATOR_IT_QDRANT` (`host:port` of a throwaway Qdrant
without a key) and `DATA_ORCHESTRATOR_IT_QDRANT_KEYED` (`host:port` of a
throwaway Qdrant started with `QDRANT__SERVICE__API_KEY=it-key-not-a-secret`
and `QDRANT__SERVICE__READ_ONLY_API_KEY=it-ro-key-not-a-secret`). The suite
deletes every collection of both Qdrants: never point them at one that holds
data.

Integration tests spin up throwaway containers via `docker run` (skipped
automatically if Docker is unavailable and the variables above are unset): `postgres:16` and
`qdrant/qdrant:v1.20.0`, the Qdrant version production runs, pinned by the
`QDRANT_IMAGE` constant of `tests/integration/conftest.py` (change it when
production is upgraded). They use a tiny embedding model
(`DATA_ORCHESTRATOR_EMBEDDING_MODEL=sentence-transformers/paraphrase-albert-small-v2`,
`DATA_ORCHESTRATOR_EMBEDDING_DIM=768`, downloaded on first use) to keep CI
fast. CI (`.github/workflows/tests.yml`) runs both suites on pushes to `main`
and on pull requests — the unit suite on Python 3.12 and 3.13, the integration
suite on 3.13, the version production runs (the libraries `pip` resolves
depend on the Python version; on 3.13 they are the ones production has): the
CPU build of PyTorch, then `pip install -e ".[dev]"`, which brings rag-engine
at the commit pinned in `pyproject.toml`. A step then checks it: `tools/check_engine_pin.py` compares the commit pip recorded for
the installed rag-engine (`direct_url.json`, `vcs_info.commit_id`) with the
sha in `pyproject.toml` and fails on a difference (it also fails on an engine
installed from a local checkout, which records no commit).

**Bumping the rag-engine pin** changes what is written to Qdrant only if the
guards say so: `tests/test_engine_parity.py` (the points written for six
fixture documents must stay the ones captured from the code this repo ran
before it used rag-engine: ids, payloads, vectors; since the schema v2 pin,
plus exactly `date_start` and `date_end`, which the test computes itself) and
`tests/test_routing.py` (the supported extensions). The parity test runs with a deterministic test
embedder and OCR off: it does not see a change of sentence-transformers,
torch, transformers or the model. Never regenerate the parity fixture to make
a failing test pass.

## Adding a new corpus

Add one `CorpusRoute` entry to `ROUTING` in `data_orchestrator/routing.py`
(the env key of its data root) — no new connector code, as long as the corpus
writes the `documents` contract and the vault ingests it. The `vault` source
works for any corpus already in `documents`. Collections are per corpus, so a
Qdrant point id from `doc_id` alone is unique; a collection mixing corpora
would have to put `corpus` in its point ids (`doc_id` is unique only within
its corpus).

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
