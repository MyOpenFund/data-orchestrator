# Migrating from the engine fork to rag-engine

Until 2026-11 this repository drove a fork (`MyOpenFund/eigenmind`) installed
from a separate checkout. It now depends on
[rag-engine](https://github.com/MyOpenFund/rag-engine), pinned by commit in
`pyproject.toml`. This note lists what an existing installation must do and
what behaves differently.

**Nothing has to be ingested again.** Point ids, payloads and vectors are
unchanged for every valid document (evidence at the end).

## What to do

1. **Reinstall the virtual environment.** The engine is now a dependency:

   ```bash
   git pull
   pip install torch --index-url https://download.pytorch.org/whl/cpu   # machines without an NVIDIA GPU only
   pip install -e .
   pip uninstall eigenmind        # the fork is no longer imported; its checkout can go
   ```

   A fresh venv is the safe way. The engine expects `chunknorris>=1.3.7,<1.6`
   and `sentence-transformers>=6,<7`; `pip` enforces both.

2. **`.env`: remove `EIGENMIND_PATH`.** It is no longer read by anything.
   Every other key keeps its name and meaning (`DATABASE_URL`, `*_ROOT`,
   `QDRANT_HOST`, `QDRANT_PORT`, `QDRANT_API_KEY`, `QDRANT_HTTPS`,
   `DATA_ORCHESTRATOR_*`).

3. **`.env`: make sure the OCR switches are in it** if this machine runs OCR:
   `ENABLE_OCR=true` and `TESSDATA_PREFIX=<folder of the *.traineddata files>`.
   The engine reads both from the process environment each time a file is
   chunked, and this repository's `.env` is the only one loaded: a value that
   lived only in a `.env` next to the fork's checkout is no longer seen, and
   `--ocr auto` then silently means `never`. Check the line every run now
   prints: `OCR: available (languages: eng)` or
   `OCR: off (ENABLE_OCR not true or TESSDATA_PREFIX unset)`.

4. **OCR languages are explicit.** The fork ran OCR with every language file
   found in the tessdata folder; the engine uses `RAG_ENGINE_OCR_LANGUAGES`
   (default `eng`). On a host whose tessdata holds only English (the
   production host), nothing changes. Where OCR ran in several languages, set
   it, e.g. `RAG_ENGINE_OCR_LANGUAGES=eng+fra`.

5. **Another embedding model than the default: declare its dimension.** With
   `DATA_ORCHESTRATOR_EMBEDDING_MODEL` set to anything but
   `intfloat/multilingual-e5-base`, set `DATA_ORCHESTRATOR_EMBEDDING_DIM` too
   (the fork read the dimension from the loaded model; the engine wants it
   declared and refuses a model that does not match). Without it an ingest run
   stops with exit 2. The default model needs nothing.

6. **Before the first run, look at the free-form metadata** (optional,
   read-only): the keys of `documents.extra` in the vault,

   ```sql
   SELECT k, count(*) FROM documents, jsonb_object_keys(extra) AS k
   WHERE k IN ('filename', 'chunk_number', 'ingestion_date', 'text', 'page')
   GROUP BY k;
   ```

   Any row found is handled (see "Reserved payload keys" below), but a `page`
   there changes one payload value.

Then one small real run (`--limit 5`) and a look at its first lines
(`Config:`, `OCR:`, `Qdrant:`).

## What changes

| Before (fork) | Now (rag-engine) |
|---|---|
| Engine installed from a checkout at `EIGENMIND_PATH`, pinned by hand | Installed with the package, pinned by commit in `pyproject.toml` |
| The fork loaded a `.env` of its own at import (python-dotenv) and froze `ENABLE_OCR` then | The engine loads no `.env` and reads nothing at import. One loader (`cli.main`); `ENABLE_OCR` / `TESSDATA_PREFIX` are read at each file |
| OCR languages: every `*.traineddata` in the tessdata folder | `RAG_ENGINE_OCR_LANGUAGES`, default `eng` |
| `--ocr always` left a `TESSDATA_PREFIX` pointing at the parent of `tessdata/` as it was unless `ENABLE_OCR=true` | The parent folder is corrected whenever OCR may run, `--ocr always` included |
| Invalid UTF-8 bytes in a `.txt` / `.md` file were dropped silently | Each invalid sequence becomes `�` (U+FFFD), with a warning in the log. Only files with invalid bytes are concerned: their chunks differ from the ones stored (re-ingest those documents into a fresh collection if it matters) |
| A metadata key named `filename`, `chunk_number`, `ingestion_date`, `text` or `page` was overwritten without a word | The engine refuses it; the sources drop and count it (below) |
| An existing collection was checked for its dimension and for named vectors | Also for its distance: **a collection that is not cosine is refused** (`CollectionMismatchError`), before any document |
| A wrong collection: `RuntimeError: collection 'x' stores 384-d vectors but the embedding model produces 768-d: use another --collection` | `CollectionMismatchError: collection 'x': expected a single unnamed 768-d vector with Cosine distance, found 384-d vectors: use another --collection` |
| A refused API key: this repository's `QdrantAuthError` (a `RuntimeError`) | rag-engine's `QdrantAuthError` (not a `RuntimeError`); same name and same message in the logs and in `runs.extra->>'fatal'` |
| HTTP 401/403 on a write in the middle of a run: one failure per remaining document (a "degraded" run, exit 3) | **One fatal error**, exit 1: `QdrantAuthError: Qdrant refused the request (HTTP 401): check QDRANT_API_KEY (while writing document '…', after N document(s) ingested in this run)`. Documents already recorded stay recorded; the next run resumes after them |
| An unreadable extension: `ValueError: format '.html' is not supported by ChunkNorris` | `UnsupportedFormatError: format '.html' is not supported: '<path>' (supported: …)` (still one per-document error) |
| An invalid `QDRANT_HTTPS`: fatal at store construction (exit 1, with a run report) | Exit 2 before any work, no report, like every invalid setting (`RAG_ENGINE_BATCH_SIZE`, `RAG_ENGINE_OCR_LANGUAGES`, `RAG_ENGINE_QDRANT_TIMEOUT`, …) |
| The model's dimension was read from the loaded model | Declared: 768 for the default model, `DATA_ORCHESTRATOR_EMBEDDING_DIM` otherwise; a mismatch is fatal before any document |
| Importing the fork configured logging (INFO to stderr) as a side effect | The CLI configures it itself, same format; level in `DATA_ORCHESTRATOR_LOG_LEVEL` (default `INFO`). The lines now come from `rag_engine.*` loggers instead of `eigenmind.*`. A library caller of `run_ingest` configures its own logging |
| — | New start-up line `OCR: …`; new optional settings `RAG_ENGINE_OCR_LANGUAGES`, `RAG_ENGINE_BATCH_SIZE`, `RAG_ENGINE_QDRANT_TIMEOUT`; `RAG_ENGINE_MODEL` / `RAG_ENGINE_DIM` read as fallbacks of the two `DATA_ORCHESTRATOR_EMBEDDING_*` keys |

Code that catches `RuntimeError` to detect a refused key or a wrong collection
must catch `rag_engine.store.QdrantAuthError` /
`rag_engine.store.CollectionMismatchError` (or their base,
`rag_engine.errors.RagEngineError`) instead. `core.QdrantAuthError`,
`core._point_id`, `core._ensure_collection`, `core.OCR_TO_ENGINE` and the lazy
wrappers `core.chunk_with_chunknorris`, `core.EmbeddingModel`,
`core.detect_device`, `core.QdrantStore` no longer exist; `run_ingest` takes
`store=`, `embedder=` and `settings=`, and builds its defaults with
`core.build_store` / `core.build_embedder`.

Exit codes, the run report (`runs` table or `runs.jsonl`) and the
`rag_ingestions` rows are otherwise unchanged. The run report gains one key,
`totals.payload_keys_dropped`, written only when it is not zero.

## Reserved payload keys

The engine writes `doc_id`, `filename`, `chunk_number`, `ingestion_date`,
`text` and `page` itself and refuses metadata that sets one of them. So that
one stray key cannot fail a document:

- the **vault source** leaves such a key of `documents.extra` out of the
  payload (the vault's columns are never dropped);
- the **company source** leaves such a key of company_corpus's payload out;
- each one is counted: a `warning:` in the run summary and
  `totals.payload_keys_dropped` in the run report. The outcome and the exit
  code do not depend on it.

For `filename`, `chunk_number`, `ingestion_date` and `text` the payload
written is exactly the one written before (the engine's value replaced the
stray one then, too). Two cases differ, both of them metadata errors:

- a stray `page` on a document **without pages** (markdown, text, CSV, …) was
  kept in the payload; it is now left out, as the schema says (no `page` for
  such a document);
- a company_corpus payload carrying a `doc_id` **other than the item's** put
  that other id in the payload (while point ids and the ledger used the
  item's); the payload now carries the item's `doc_id`.

A `SourceItem` built by hand with a reserved key is not rewritten: that
document fails with `ValueError: metadata may not set reserved payload key(s):
…`.

## Evidence that nothing has to be re-ingested

- `tests/test_engine_parity.py`: for six documents (markdown, text, CSV, a
  3-page PDF, a PDF with a blank page, a Word file) described by vault-like
  rows, the points written today — 23 of them — are compared with the points
  captured from this repository at `7196bfc` driving the fork at `6e9093a`:
  same ids, same payloads (every key and value, `ingestion_date` aside), same
  vectors. `tests/fixtures/engine_parity/README.md` says how the capture was
  made.
- rag-engine's own golden test compares `ingest_document` with points captured
  through this repository's previous `ingest_item`; its PyTorch embedder was
  measured against the fork's on the real model (bit-identical vectors) and
  passes the texts to sentence-transformers exactly as the fork did.
- Point ids depend only on `doc_id`, the page and the chunk position
  (`rag_engine.schema.point_id`, the algorithm this repository had): a
  document ingested again overwrites its points in place.
