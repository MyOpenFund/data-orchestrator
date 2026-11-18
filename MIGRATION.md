# Migrating from the engine fork to rag-engine

Until 2026-11 this repository drove a fork (`MyOpenFund/eigenmind`) installed
from a separate checkout. It now depends on
[rag-engine](https://github.com/MyOpenFund/rag-engine), pinned by commit in
`pyproject.toml`. This note lists what an existing installation must do and
what behaves differently.

**What is known about the points already stored.** With the same versions of
the libraries that decide text and vectors (chunknorris and its PDF library
PyMuPDF, sentence-transformers, torch, transformers, the revision of the
embedding model) and the same device, the same documents give the same point
ids, the same payloads and the same vectors as before the switch, with two
declared payload differences:

- invalid UTF-8 bytes in a `.txt` / `.md` file are replaced by `�` (U+FFFD);
  they used to be dropped;
- a reserved key found in a document's free metadata is dropped (a stray
  `page` on a document without pages used to be kept).

This is what was checked, and how, in ["What was checked"](#what-was-checked)
at the end. It was not checked with other versions of those libraries, nor on
another device than the one of the checks (CPU). So a re-run that overwrites
points in place can be relied on once the versions are known to be the same
(step 1) and a few already-ingested documents have been compared on the real
installation
(["Validating on real data after the switch"](#validating-on-real-data-after-the-switch)).

## What to do

1. **Save the installed versions, then install in place.** The engine is now
   a dependency. In the existing virtual environment, **before** anything is
   installed, save the versions the stored points were made with (each
   block below is one command, without comments: the default shell of macOS,
   zsh, does not read a `#` typed at the prompt as a comment):

   ```bash
   pip freeze > ~/data-orchestrator-freeze-before-switch.txt
   ```

   Update the checkout:

   ```bash
   git pull
   ```

   Install; this brings rag-engine at the pinned commit:

   ```bash
   pip install -e .
   ```

   The fork is still installed in the environment after that: it was an
   editable install, so its `.pth` entry (which points at its checkout) and
   its console script `eigenmind-ingest` are still there. Nothing in this
   repository imports it any more, so it changes nothing to a run; remove it
   all the same, so that the environment holds one engine (`pip` asks for
   confirmation):

   ```bash
   pip uninstall eigenmind
   ```

   This removes the installation only. The fork's checkout directory is left
   alone; delete it by hand once it is no longer wanted.

   Prefer this in-place install to a fresh environment: `pip` leaves an
   installed library alone when its version satisfies the requirement, so
   chunknorris, PyMuPDF, sentence-transformers, torch and transformers stay at
   the versions the stored points were made with. The engine expects
   `chunknorris>=1.3.7,<1.6` and `sentence-transformers>=6,<7`; a version
   outside these ranges is replaced, and `pip` says so. Then compare:

   ```bash
   pip freeze | grep -i -E '^(chunknorris|pymupdf|sentence-transformers|torch|transformers)=' \
     | diff - <(grep -i -E '^(chunknorris|pymupdf|sentence-transformers|torch|transformers)=' \
                ~/data-orchestrator-freeze-before-switch.txt) && echo "same versions"
   ```

   **If a fresh environment is made instead**, `pip` installs the latest
   versions the ranges allow: run the comparison above against the saved
   freeze, install the saved versions where they differ
   (`pip install chunknorris==… PyMuPDF==… …`), and validate on real data
   (section below) before relying on a re-run overwriting points in place.
   Keep the model cache (`~/.cache/huggingface`, or `RAG_ENGINE_MODEL_CACHE`):
   the folder name under
   `models--intfloat--multilingual-e5-base/snapshots/` is the model revision
   in use.

   Which PyTorch build, for a fresh environment:

   - **Apple Silicon (the production Mac): install normally**, as above. The
     default wheel uses the Metal backend (`mps`); the first run must log
     `Loading embedding model intfloat/multilingual-e5-base on mps`.
   - **Linux without an NVIDIA GPU, and CI**: install the CPU build first,

     ```bash
     pip install torch --index-url https://download.pytorch.org/whl/cpu
     ```

     then the package:

     ```bash
     pip install -e .
     ```

     Without the first command the default Linux wheel pulls the CUDA
     packages (several gigabytes) for nothing.

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
   SELECT k, count(*)
   FROM documents,
        jsonb_object_keys(CASE WHEN jsonb_typeof(extra) = 'object'
                               THEN extra ELSE '{}'::jsonb END) AS k
   WHERE k IN ('filename', 'chunk_number', 'ingestion_date', 'text', 'page')
   GROUP BY k;
   ```

   (The `jsonb_typeof` guard skips an `extra` that is a JSON `null`, an array
   or a scalar: `jsonb_object_keys` fails on those.) Any row found is handled
   (see "Reserved payload keys" below), but a `page` there changes one payload
   value.

7. **`--ocr always` needs `TESSDATA_PREFIX`.** Forcing OCR without the
   language data is a configuration error: the run is refused before any work
   (`error: forced OCR needs TESSDATA_PREFIX; set it or use --ocr auto/never`,
   exit 2), whatever the selection holds. Set it in the `.env` (step 3), or
   use `--ocr auto` or `--ocr never`.

Then validate on real data, as described next.

## Validating on real data after the switch

**No step of this section writes to the production collection: every command
that ingests names the scratch collection explicitly (`--collection
zz-switch-check`), and everything else only reads.**

The names below are examples: `central-bank-e5b-v1` is the production
collection, `zz-switch-check` the scratch one, `ecb` / `A1` a source and a
document type production already holds. Replace them by yours, and keep
`--collection zz-switch-check` in every `data-orchestrator` command of this
section: without `--collection` a run writes to the routed collection, which
is the production one.

What the section does write: the scratch collection in Qdrant, the
`rag_ingestions` rows of that collection name, and one `runs` row per run. The
last step removes the first two. Every block is one command, without comments,
to be pasted as it is; run them from the repository directory, in its virtual
environment.

The vault source selects the documents that have no `rag_ingestions` row
**for the target collection**. A fresh collection name has its own ledger
rows — none yet — so documents already ingested into production are selected
again for it, and the production rows and points are not touched.

### 1. A small run into the scratch collection, and its first lines

Count first (this reads the vault only): the documents of a source and a type
that production holds.

```bash
data-orchestrator vault --corpus central-bank --collection zz-switch-check --source-codes ecb --doctypes A1 --count-only
```

Then ingest the first three of them, in `doc_id` order, into the scratch
collection. Use the OCR mode of the documents' original run: the same `--ocr`
value and, for `auto`, the same `ENABLE_OCR` / `TESSDATA_PREFIX`.

```bash
data-orchestrator vault --corpus central-bank --collection zz-switch-check --source-codes ecb --doctypes A1 --limit 3
```

Check its first lines, in this order:

- `Config: .env = <the .env of this repository>` — not `no .env file found`;
- `Resume ledger (vault): rag_ingestions/zz-switch-check (0 docs already
  done)` (0 on the first run) and `→ Ingesting vault selection
  (corpus=central-bank) into 'zz-switch-check'`: the scratch name, in both;
- `OCR: available (languages: eng)` on a machine that runs OCR (if it says
  `OCR: off (ENABLE_OCR not true or TESSDATA_PREFIX unset)`, see step 3 of
  "What to do"): the line must read as it did when the documents were first
  ingested;
- `Qdrant: <host>:<port> (api key: set)` — the host and port of the production
  Qdrant, no warning about port 6333;
- in the log lines (stderr), `Loading embedding model
  intfloat/multilingual-e5-base on mps` on an Apple machine (`on cpu` there
  means PyTorch does not see the Metal backend: vectors may then differ from
  the stored ones in their last digits).

The run must end with `docs ingested  : 3` and `errors         : 0`.

### 2. Compare these documents with their production points

Only a document whose production points were written **before the switch** is
worth comparing: one that production never had has nothing to compare with,
and one ingested after the switch was written by the new code on both sides.
For the documents now in the scratch collection, their production row, its
chunk count and the time it was written:

```sql
SELECT s.doc_id,
       s.chunk_count AS scratch_chunks,
       p.chunk_count AS production_chunks,
       p.ingested_at AS production_ingested_at
FROM rag_ingestions s
LEFT JOIN rag_ingestions p
       ON p.doc_id = s.doc_id AND p.collection = 'central-bank-e5b-v1'
WHERE s.collection = 'zz-switch-check'
ORDER BY s.doc_id;
```

Read each row:

- `production_chunks` is NULL: production has no row for this document. Skip
  it.
- `production_ingested_at` is later than the switch: skip it too. The time of
  the switch is the time of the file saved in step 1 of "What to do"
  (`ls -l ~/data-orchestrator-freeze-before-switch.txt`).
- otherwise the document qualifies; `scratch_chunks` and `production_chunks`
  should already be equal.

If no document qualifies, run the second command of step 1 again (it takes
the next three: the first three are now recorded for the scratch collection),
or narrow it to years ingested long ago with `--year-min` / `--year-max`.

Create the file `/tmp/compare_doc.py` — outside the repository, so that
nothing is left in the checkout — with this content. It only reads, from both
collections; the Qdrant settings and the API key come from the `.env`, and the
key is never printed.

```python
# compare_doc.py — usage: python /tmp/compare_doc.py <production> <scratch> <doc_id>
import os
import sys

import numpy as np
from qdrant_client import QdrantClient, models

from data_orchestrator import config

production, scratch, doc_id = sys.argv[1:4]
config.load_dotenv()
client = QdrantClient(
    host=os.environ["QDRANT_HOST"], port=int(os.environ["QDRANT_PORT"]),
    api_key=(os.environ.get("QDRANT_API_KEY") or "").strip() or None,
    https=(os.environ.get("QDRANT_HTTPS") or "").strip().lower() in ("1", "true", "yes", "on"),
    timeout=120)
of_doc = models.Filter(must=[
    models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))])

new, offset = {}, None
while True:  # every point of the document in the scratch collection
    points, offset = client.scroll(scratch, scroll_filter=of_doc, limit=256, offset=offset,
                                   with_payload=True, with_vectors=True)
    new.update({p.id: p for p in points})
    if offset is None:
        break
old = {p.id: p for p in client.retrieve(production, ids=list(new),
                                        with_payload=True, with_vectors=True)}
stored = client.count(production, count_filter=of_doc, exact=True).count

def cosine(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

def without_date(payload):
    return {k: v for k, v in payload.items() if k != "ingestion_date"}

common = sorted(set(new) & set(old), key=str)
print(f"points: scratch {len(new)}, production {stored}")
print("ids equal     :", bool(new) and set(new) == set(old) and stored == len(new))
print("texts equal   :", bool(common) and all(new[i].payload["text"] == old[i].payload["text"] for i in common))
print("payloads equal:", bool(common) and all(without_date(new[i].payload) == without_date(old[i].payload) for i in common))
print("min cosine    :", min((cosine(new[i].vector, old[i].vector) for i in common), default=None))
```

Then, for each document that qualifies, with its `doc_id` in place of
`<doc_id>` (keep the single quotes):

```bash
python /tmp/compare_doc.py central-bank-e5b-v1 zz-switch-check '<doc_id>'
```

Expected: `ids equal: True`, `texts equal: True`, `payloads equal: True`
(`ingestion_date`, the time of each run, is left out) and a minimum cosine of
1 to within rounding (0.9999 or more). The count of the production points
scans the collection when `doc_id` has no payload index: slow on a large
collection, not wrong. If the ids or the texts differ, the text extraction is
not the one of the stored points (a library version, the OCR mode): stop and
compare the versions with the saved freeze. If only the cosine is lower, the
embedding differs (torch, transformers, the model revision, the device).

### 3. The production collection still matches its ledger

Both statements only read. The documents and the points the ledger expects in
production:

```sql
SELECT count(*) AS documents, sum(chunk_count) AS points_expected
FROM rag_ingestions
WHERE collection = 'central-bank-e5b-v1';
```

The points Qdrant holds for it:

```bash
python -c "
import os
from qdrant_client import QdrantClient
from data_orchestrator import config
config.load_dotenv()
print(QdrantClient(host=os.environ['QDRANT_HOST'], port=int(os.environ['QDRANT_PORT']),
                   api_key=(os.environ.get('QDRANT_API_KEY') or '').strip() or None,
                   https=(os.environ.get('QDRANT_HTTPS') or '').strip().lower() in ('1', 'true', 'yes', 'on'),
                   timeout=120).count('central-bank-e5b-v1', exact=True).count)"
```

The two numbers must be equal, and the same as before this section was run;
check them again after the first real run. More points than expected means a
document was written and not recorded (a run interrupted between the two
writes; the next run writes it again over the same ids) or was re-ingested
with fewer chunks than before; fewer points than expected means recorded
documents are missing from Qdrant.

### 4. Remove what the check created

The scratch collection (the name is written in the command: it deletes that
collection and no other):

```bash
python -c "
import os
from qdrant_client import QdrantClient
from data_orchestrator import config
config.load_dotenv()
QdrantClient(host=os.environ['QDRANT_HOST'], port=int(os.environ['QDRANT_PORT']),
             api_key=(os.environ.get('QDRANT_API_KEY') or '').strip() or None,
             https=(os.environ.get('QDRANT_HTTPS') or '').strip().lower() in ('1', 'true', 'yes', 'on'),
             timeout=120).delete_collection('zz-switch-check')"
```

Its ledger rows, with the vault's admin role (the orchestrator's role may not
delete from `rag_ingestions`):

```sql
DELETE FROM rag_ingestions WHERE collection = 'zz-switch-check';
```

The helper script:

```bash
rm /tmp/compare_doc.py
```

Do the first two together: ledger rows left behind for a collection that no
longer exists would make a later run into that name skip those documents. The
`runs` rows of the check stay (telemetry; nothing reads them for resume).

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
| HTTP 401/403 on a write in the middle of a run: one failure per remaining document (a "degraded" run, exit 3) | **One fatal error**, exit 1: `QdrantAuthError: Qdrant refused the request (HTTP 401): check QDRANT_API_KEY (while writing document '…', after N document(s) ingested in this run)`. Documents already recorded stay recorded; the next run resumes after them. The run report counts them (`totals`, `sources`); it is written first, then the counts are printed under `Interrupted by a fatal error. Before it:` (not `Done.`), then the fatal line |
| `RAG_ENGINE_BACKEND` did not exist | Leave it unset. `onnx` (the engine's backend for search services, not installed here) is refused with exit 2 before any work: ingestion uses the torch backend |
| `--ocr always` without `TESSDATA_PREFIX`: one error per PDF (a selection without PDFs went through) | A configuration error, refused with exit 2 before any work, whatever the selection holds: `error: forced OCR needs TESSDATA_PREFIX; set it or use --ocr auto/never` |
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
written has the same keys and the same values as before (the engine's value
replaced the stray one then, too); the order of the keys inside the payload
may differ, which no reader depends on. Two cases differ, both of them
metadata errors:

- a stray `page` on a document **without pages** (markdown, text, CSV, …) was
  kept in the payload; it is now left out, as the schema says (no `page` for
  such a document);
- a company_corpus payload carrying a `doc_id` **other than the item's** put
  that other id in the payload (while point ids and the ledger used the
  item's); the payload now carries the item's `doc_id`.

A `SourceItem` built by hand with a reserved key is not rewritten: that
document fails with `ValueError: metadata may not set reserved payload key(s):
…`.

## What was checked

All three checks ran on CPU, the old and the new code on the same installed
libraries (the fixture records chunknorris 1.5.0 and PyMuPDF 1.27.2):

1. **A committed fixture, deterministic test embedder, OCR off**
   (`tests/test_engine_parity.py`, run by CI). Six documents (markdown, text,
   CSV, a 3-page PDF, a PDF with a blank page, a Word file) described by
   vault-like rows: the 23 points written today are compared with the points
   captured from this repository at `7196bfc` driving the fork at `6e9093a` —
   same ids, same payloads (every key and value, `ingestion_date` aside), same
   vectors. The embedder is the engine's test embedder, not a model: this
   shows that the same texts reach the embedder with the same prefix and that
   ids and payloads are assembled the same way. `tests/fixtures/engine_parity/README.md`
   says how the capture was made.
2. **End to end, old code against new code, on PostgreSQL and Qdrant, with a
   real small model** (`sentence-transformers/paraphrase-albert-small-v2`):
   the same vault selection ingested by the code before the switch (with the
   fork) and by the code after it, each into a fresh database and a fresh
   Qdrant — same points (ids, payloads, vectors), same `rag_ingestions` rows.
   Done once, by hand, at the switch; not a test of this repository.
3. **With OCR on, in English, on generated documents**: the old and the new
   code give the same points. Done once, by hand.

Not covered by these checks: the production model on the production device
(`intfloat/multilingual-e5-base` on `mps`), OCR in another language, real
scanned documents, and any other version of the libraries above. That is what
["Validating on real data after the switch"](#validating-on-real-data-after-the-switch)
is for.

Point ids depend only on `doc_id`, the page and the chunk position
(`rag_engine.schema.point_id`, the algorithm this repository had): a document
ingested again with the same chunking overwrites its points in place.
