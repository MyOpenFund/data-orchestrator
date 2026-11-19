# Correcting dates in Qdrant: validation, then the production run

`sync-metadata` (README, ["Correcting dates in Qdrant"](README.md#correcting-dates-in-qdrant-sync-metadata))
brings the date metadata of the points of `central-bank-e5b-v1` in line with
the vault, without re-embedding: `date`, `year`, `date_precision`, `date_kind`,
`date_source` and the interval rag-engine derives from them, `date_start` / `date_end`
(payload schema v2). This note is the procedure for a first run on a
collection written before schema v2.

**Production is not written before the end.** Steps 1 to 4 write only a
scratch collection, `zz-sync-check`, made of copies of a few production
documents, and its `rag_ingestions` rows; steps 5 and 6 only read production;
step 7 is the first and only write to production. Every block is one command,
without comments, to be pasted as it is into zsh (the default shell of macOS,
which does not read a `#` typed at the prompt as a comment) or bash; run them
from the repository directory, in its virtual environment. The names are the
ones of the central-bank installation: `central-bank-e5b-v1` the production
collection, `zz-sync-check` the scratch one.

What the procedure writes, apart from the scratch collection and production
in step 7: four files under `/tmp` (a helper script, two snapshots of the
scratch collection, the CSVs) and two snapshots of production under `~`
(about 100 MB together: they hold the payloads). Step 9 removes them.

## 0. Before starting

The vault must have ingested the corrected manifests (the policy-dates
backfill of central-bank-corpus, then the vault's next ingestion). With
`psql` on the vault, this reads only:

```sql
SELECT count(*) FILTER (WHERE extra ? 'date_kind') AS with_kind,
       count(*) FILTER (WHERE extra ? 'date_precision') AS with_precision,
       count(*) FILTER (WHERE date IS NULL) AS undated
FROM documents d
JOIN rag_ingestions r ON r.doc_id = d.doc_id AND r.collection = 'central-bank-e5b-v1';
```

Expected: `with_kind` close to 2,554 (the backfill gave a kind to 2,554 rows;
misfiled, lastmod-dated and undated rows have none), `with_precision` close to
the number of ingested documents, `undated` 3 (the three Bank of Canada index
pages). `with_kind` at 0 means the vault has not taken the corrected
manifests yet: stop here.

Update the checkout and install; the second command brings rag-engine at the
pinned commit, the third checks it:

```bash
git pull
```

```bash
pip install -e .
```

```bash
python tools/check_engine_pin.py
```

No ingestion of `central-bank-e5b-v1` may run while `sync-metadata` writes. An
ingestion of this version and `sync-metadata --write` exclude each other (a
vault advisory lock: whichever comes second stops with `CollectionBusyError`;
a dry run takes the lock shared and runs beside an ingestion), but
an ingestion started by an older checkout, on any machine, takes no lock.
Check that none runs, on every machine that ingests:

```bash
pgrep -fl 'data[-_]orchestrator'
```

(no output: none runs here; the pattern matches the console script and
`python -m data_orchestrator.cli` alike).

If a later step stops with `CollectionBusyError` although nothing runs (a
machine that went to sleep or lost its network during a run: the server
keeps its session, and the lock, until its own keepalive ends it), the
message names the session (`held by pid …`, its application
`data-orchestrator@<host>`, and, when the role may read them, its client and
start). Check on that machine that no process is left (the command above),
then end the session with the vault's admin role, using the pid of the
message:

```sql
SELECT pg_terminate_backend(<pid>);
```

or wait for the server to drop it. Server-side keepalives on the vault's
Postgres (`tcp_keepalives_idle = 60`, `tcp_keepalives_interval = 10`,
`tcp_keepalives_count = 6`) make that wait a few minutes (README, "A stale
lock").

## 1. A scratch collection made of a few production documents

The scratch ledger: one document of each source, document type, date
precision and dated/undated group that production holds (about forty
documents). This writes `rag_ingestions` rows for `zz-sync-check` only:

```sql
INSERT INTO rag_ingestions (doc_id, collection, corpus, source_code, embedding_model, embedding_version, chunk_count)
SELECT doc_id, 'zz-sync-check', corpus, source_code, embedding_model, embedding_version, chunk_count
FROM (
  SELECT r.*, row_number() OVER (
           PARTITION BY d.source_code, d.doc_type, d.extra->>'date_precision', d.date IS NULL
           ORDER BY r.doc_id) AS n
  FROM rag_ingestions r
  JOIN documents d ON d.doc_id = r.doc_id
  WHERE r.collection = 'central-bank-e5b-v1' AND d.deleted_at IS NULL
) AS picked
WHERE n = 1;
```

Create the file `/tmp/sync_check.py` — outside the repository, so that
nothing is left in the checkout — with this content. `copy` reads production
and writes the scratch collection only (it refuses a scratch collection that
already exists); `snapshot` and `diff` only read. The Qdrant settings and the
key come from the `.env`; the key is never printed.

```python
# sync_check.py — usage:
#   python /tmp/sync_check.py copy <production> <scratch>
#   python /tmp/sync_check.py snapshot <collection> <file>
#   python /tmp/sync_check.py diff <before-file> <after-file>
# copy reads <production> only; snapshot only reads; diff reads two files.
import hashlib
import json
import os
import struct
import sys

from qdrant_client import QdrantClient, models

from data_orchestrator import config, vault
from data_orchestrator.sync_metadata import OWNED_KEYS

config.load_dotenv()
client = QdrantClient(
    host=os.environ["QDRANT_HOST"], port=int(os.environ["QDRANT_PORT"]),
    api_key=(os.environ.get("QDRANT_API_KEY") or "").strip() or None,
    https=(os.environ.get("QDRANT_HTTPS") or "").strip().lower() in ("1", "true", "yes", "on"),
    timeout=120)


def points_of(collection, doc_ids=None):
    selector = None
    if doc_ids is not None:
        selector = models.Filter(must=[models.FieldCondition(
            key="doc_id", match=models.MatchAny(any=sorted(doc_ids)))])
    offset = None
    while True:
        page, offset = client.scroll(collection, scroll_filter=selector, limit=256,
                                     offset=offset, with_payload=True, with_vectors=True)
        yield from page
        if offset is None:
            return


def copy(production, scratch):
    conn = vault.connect()
    with conn.cursor() as cur:
        cur.execute("SELECT doc_id FROM rag_ingestions WHERE collection = %s", (scratch,))
        doc_ids = [row[0] for row in cur.fetchall()]
    conn.close()
    if not doc_ids:
        sys.exit(f"no rag_ingestions row for {scratch}: insert them first")
    info = client.get_collection(production)
    if client.collection_exists(scratch):
        sys.exit(f"{scratch} already exists: delete it first")
    params = info.config.params.vectors
    client.create_collection(scratch, vectors_config=models.VectorParams(
        size=params.size, distance=params.distance))
    copied = [models.PointStruct(id=p.id, vector=p.vector, payload=p.payload)
              for p in points_of(production, doc_ids)]
    client.upsert(scratch, points=copied, wait=True)
    for field, schema in (info.payload_schema or {}).items():
        client.create_payload_index(scratch, field, field_schema=schema.data_type, wait=True)
    print(f"{len(doc_ids)} documents, {len(copied)} points copied to {scratch}; "
          f"indexes: {', '.join(sorted(info.payload_schema or {}))}")


def snapshot(collection, path):
    out = {}
    for p in points_of(collection):
        vector = struct.pack(f"<{len(p.vector)}f", *p.vector)
        out[str(p.id)] = {"vector_sha256": hashlib.sha256(vector).hexdigest(),
                          "payload": p.payload}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    print(f"{len(out)} points of {collection} saved to {path}")


def diff(before_path, after_path):
    with open(before_path, encoding="utf-8") as fh:
        before = json.load(fh)
    with open(after_path, encoding="utf-8") as fh:
        after = json.load(fh)
    same_ids = set(before) == set(after)
    common = set(before) & set(after)
    vectors = all(before[i]["vector_sha256"] == after[i]["vector_sha256"] for i in common)
    changed, documents = {}, set()
    for i in common:
        old, new = before[i]["payload"], after[i]["payload"]
        for key in set(old) | set(new):
            if old.get(key) != new.get(key):
                changed[key] = changed.get(key, 0) + 1
                documents.add(new.get("doc_id"))
    others = sorted(set(changed) - set(OWNED_KEYS))
    print(f"points: before {len(before)}, after {len(after)}")
    print("same point ids       :", same_ids)
    print("vectors byte-identical:", vectors)
    print("documents changed    :", len(documents))
    print("keys changed (points):", dict(sorted(changed.items())))
    print("other keys changed   :", others or "none")


if __name__ == "__main__":
    command, *args = sys.argv[1:]
    {"copy": copy, "snapshot": snapshot, "diff": diff}[command](*args)
```

Copy the points of those documents, vectors included, and the payload
indexes production has (the seven of schema v1):

```bash
python /tmp/sync_check.py copy central-bank-e5b-v1 zz-sync-check
```

Expected: `N documents, M points copied to zz-sync-check; indexes: corpus,
date, doc_id, doc_type, language, source_code, year`.

## 2. Dry run on the scratch collection

```bash
data-orchestrator sync-metadata --collection zz-sync-check --csv /tmp/zz-sync-check-dry.csv
```

Check: `(dry run: nothing is written to Qdrant)` in the first lines,
`payload indexes missing: date_start, date_end`, then in the summary
`missing-in-qdrant : 0`, `missing-in-vault : 0`, `points disagreeing : 0
document(s)`, `unchanged : 0` and every dated document under `updated` (no
point of production has its interval yet). Read the CSV
(`open /tmp/zz-sync-check-dry.csv` opens it in a spreadsheet on a Mac): for
each document, `date_start` / `date_end` follow `date_precision` (a `month`
date ends on the last day of its month), and a `date` or `year` row shows the
old placeholder and the corrected value.

With a read-only key the same command works too. To prove it, a copy of the
`.env` whose `QDRANT_API_KEY` is the read-only key (the one of the search
service), named by `DATA_ORCHESTRATOR_DOTENV` (read from the real environment
only):

```bash
DATA_ORCHESTRATOR_DOTENV=~/data-orchestrator-readonly.env data-orchestrator sync-metadata --collection zz-sync-check --csv /tmp/zz-sync-check-ro.csv
```

The same summary is expected. With `--write` added, the same command must stop
with `error: fatal: QdrantAuthError: … a read-only key is refused`, exit 1,
having written nothing.

## 3. Write on the scratch collection; vectors byte-identical

Snapshot the scratch collection (point ids, a SHA-256 of each vector as
float32 bytes, the payloads):

```bash
python /tmp/sync_check.py snapshot zz-sync-check /tmp/zz-before.json
```

Write:

```bash
data-orchestrator sync-metadata --collection zz-sync-check --write --csv /tmp/zz-sync-check-write.csv
```

Expected: `payload indexes created: date_start, date_end`, `Done.`,
`documents written` equal to `documents to write` of step 2, `points written`
equal to `points read` minus the points of the `undated` documents that had
nothing to change. Snapshot again and compare:

```bash
python /tmp/sync_check.py snapshot zz-sync-check /tmp/zz-after.json
```

```bash
python /tmp/sync_check.py diff /tmp/zz-before.json /tmp/zz-after.json
```

Expected, exactly:

```text
same point ids       : True
vectors byte-identical: True
other keys changed   : none
```

and under `keys changed (points)` only `date`, `year`, `date_precision`,
`date_kind`, `date_source`, `date_start`, `date_end`. Any other line means stop: do not go to
step 7.

## 4. A second write finds nothing

```bash
data-orchestrator sync-metadata --collection zz-sync-check --write --csv /tmp/zz-sync-check-again.csv
```

Expected: `documents written  : 0`, `points disagreeing : 0 document(s)`.

## 5. Dry run on production

This reads only (the vault and Qdrant; the CSV is written under `~`):

```bash
data-orchestrator sync-metadata --collection central-bank-e5b-v1 --csv ~/sync-metadata-dry.csv
```

A summary of the CSV, per key, then the date and year changes one by one:

```bash
python -c "
import csv, collections, sys
rows = list(csv.DictReader(open(sys.argv[1], encoding='utf-8')))
print(collections.Counter(r['action'] for r in rows if r['field'] in ('date', '')))
print(collections.Counter(r['field'] for r in rows))
for r in rows:
    if r['field'] in ('date', 'year'):
        print(r['doc_id'], r['field'], r['qdrant_value'], '->', r['vault_value'])
" ~/sync-metadata-dry.csv
```

**What to expect** (the collection: 33,601 points of 2,566 documents written
with payload v1, the seven indexes of v1; the vault: the 2,567 policy-core rows
corrected by the policy-dates backfill, whose `--write` changed `date_kind` on
2,554 rows, `date_precision` on 2,433, `date_source` on 2,421, `date` on 339
and `year` on 13):

| line | expected | why |
|---|---|---|
| `documents` | the number of `rag_ingestions` rows of `central-bank-e5b-v1`, at most 2,566 | the ledger is the list |
| `missing-in-qdrant`, `missing-in-vault` | 0 | `rag_ingestions ⊆ Qdrant`; no policy-core row is deleted. Anything else: read those CSV rows before going on |
| `undated` | 3, with nothing to write | the three Bank of Canada E1 index pages, the only policy-core rows left without a date: no `date`, no interval, no kind |
| `unchanged` | 0 | no point has `date_start` / `date_end` yet |
| `updated`, `documents to write` | every dated document: the documents minus the undated ones, about 2,563 | each one gains its interval |
| `points disagreeing` | 0 | one ingestion per document |
| `keys changing`: `date_start`, `date_end` | the number of `updated` documents | |
| `date_precision` | nearly every `updated` document (each one whose vault row has a precision) | no point carries a precision yet |
| `date_kind` | at most 2,554 | the rows the backfill gave a kind, among the ingested ones (the misfiled, lastmod-dated and undated rows have none) |
| `date` | at most 339, including the six Bank of Japan 2026 minutes that were undated | the corrected placeholders among the ingested documents (gb A3, gb E1, ca E1, jp A3) |
| `year` | at most 13 | dates moved to another year |
| `date_source` | at most 2,421 | the rows the backfill gave a source, among the ingested ones |

Read the `date` and `year` lines printed above: each pair must be a
placeholder (`YYYY-01-01`, or a first of the month) becoming the date the
policy-dates CSV of the backfill gave for that `doc_id`. A difference with
that CSV, or a count outside the table, is a reason to stop and look before
step 7.

## 6. Snapshot production

This only reads (every point with its vector: a few minutes, a file of about
50 MB):

```bash
python /tmp/sync_check.py snapshot central-bank-e5b-v1 ~/sync-metadata-before.json
```

## 7. Write on production

Check again that no ingestion runs (step 0), then:

```bash
data-orchestrator sync-metadata --collection central-bank-e5b-v1 --write --csv ~/sync-metadata-write.csv
```

Expected: `payload indexes created: date_start, date_end` (the two indexes
are created first: from then on a search in `auto` mode uses the intervals,
and keeps the legacy condition on `date` for the points not corrected yet, so
no document is lost while the run goes on), then the counts of step 5 with
`documents written` in place of `documents to write`, and `points written`
close to 33,601 (all of them but the points of the undated documents).

An interrupted run (a network failure, Ctrl-C) leaves each document either
corrected or not: run the same command again, it writes only what still
differs.

## 8. Check production

```bash
python /tmp/sync_check.py snapshot central-bank-e5b-v1 ~/sync-metadata-after.json
```

```bash
python /tmp/sync_check.py diff ~/sync-metadata-before.json ~/sync-metadata-after.json
```

Expected: `same point ids : True`, `vectors byte-identical: True`, `other
keys changed : none`, `documents changed` equal to `documents written`.

A dry run must now find nothing:

```bash
data-orchestrator sync-metadata --collection central-bank-e5b-v1 --csv ~/sync-metadata-again.csv
```

Expected: `documents to write : 0`, `unchanged` equal to the documents minus
the undated ones.

The points still without an interval are the points of the undated documents
only:

```bash
python -c "
import os
from qdrant_client import QdrantClient, models
from data_orchestrator import config
config.load_dotenv()
client = QdrantClient(host=os.environ['QDRANT_HOST'], port=int(os.environ['QDRANT_PORT']),
                      api_key=(os.environ.get('QDRANT_API_KEY') or '').strip() or None,
                      https=(os.environ.get('QDRANT_HTTPS') or '').strip().lower() in ('1', 'true', 'yes', 'on'),
                      timeout=120)
empty = models.IsEmptyCondition(is_empty=models.PayloadField(key='date_start'))
print(client.count('central-bank-e5b-v1', count_filter=models.Filter(must=[empty]), exact=True).count)"
```

Then re-run the evaluation baseline of rag-engine: the date-filtered
questions are the check of the search (spec of the chantier, section 5).

## 9. Remove what the procedure created

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
             timeout=120).delete_collection('zz-sync-check')"
```

Its ledger rows, with the vault's admin role (the orchestrator's role may not
delete from `rag_ingestions`):

```sql
DELETE FROM rag_ingestions WHERE collection = 'zz-sync-check';
```

Do these two together: ledger rows left behind for a collection that no
longer exists would make a later run into that name skip those documents.

The files (keep the CSVs of production if you want a record of the run: no
`runs` row is written by `sync-metadata`):

```bash
rm /tmp/sync_check.py /tmp/zz-before.json /tmp/zz-after.json /tmp/zz-sync-check-dry.csv /tmp/zz-sync-check-write.csv /tmp/zz-sync-check-again.csv
```

```bash
rm -f /tmp/zz-sync-check-ro.csv ~/sync-metadata-before.json ~/sync-metadata-after.json
```

The read-only copy of the `.env` of step 2, if you made one: it holds the
database password and a Qdrant key.

```bash
rm -f ~/data-orchestrator-readonly.env
```

## Aligning the links (`--include-links`)

Points ingested before the vault source sent `document_url`, `source_url`
and `mime_type` have none of them, and may carry keys of `extra` that are now
dropped by design (`html_path`, a path on the producer's machine;
`alt_urls`, `alt_source_urls`; README, "Qdrant payload"), or the address
under its former name `pdf_url` (see "The `document_url` rename" below).
`--include-links` sets the three links from the vault and removes those keys. **It also
(re)writes the full date set** of every document it writes: the write of a
document is always the whole set of owned keys (`date`, `year`,
`date_precision`, `date_kind`, `date_source`, `date_start`, `date_end`, then
the links and the dropped keys), with the vault's values, even when only a
link differs. Once the dates are aligned this rewrites the values the points
already hold; before, it corrects them too. The vectors and every key outside
the owned ones are untouched.

**The first run touches every ledger document** that has a link or a
`mime_type` in the vault: no point carries `mime_type` yet (no ingestion has
sent it), so on production that is every document of the collection, each
written with its full date set. Expect `documents to write` close to the
ledger count in the dry run.
Same procedure and same guarantees as above: dry run first, then `--write`,
then a second `--write` that finds nothing. It can run on its own once the
dates are aligned (the date keys then show no difference), on a machine that
reaches the vault and the production Qdrant.

**Pre-flight, before the production write.** The vault keeps `document_url`
(formerly `pdf_url`), `source_url` and `mime_type` in their columns, never in
`extra`; the write sets each of them from the column, so a value found only in
`extra` would be **removed** from the points (a NULL column removes the key;
a `pdf_url` of `extra` is never sent). Check that no row holds one, with
`psql` on the vault (it reads only):

```sql
SELECT count(*) FROM documents WHERE extra ?| array['document_url','pdf_url','source_url','mime_type'];
```

Expected: `0`. If it is not 0, **stop**: those keys would be removed from the
points; bring the rows back to the vault's split first.

The dry run:

```bash
data-orchestrator sync-metadata --collection central-bank-e5b-v1 --include-links --csv ~/sync-links-dry.csv
```

What to read in the CSV: `document_url` / `source_url` / `mime_type` rows
with an empty `qdrant_value` (the links the points lack), `pdf_url` /
`html_path` / `alt_urls` / `alt_source_urls` rows with an empty `vault_value`
(the keys to remove), and no date row. A row of another field means the dates are not aligned yet: run
the date correction first.

```bash
python -c "
import csv, collections, sys
rows = list(csv.DictReader(open(sys.argv[1], encoding='utf-8')))
print(collections.Counter(r['field'] for r in rows))
" ~/sync-links-dry.csv
```

Then, with a key allowed to write, and no ingestion running:

```bash
data-orchestrator sync-metadata --collection central-bank-e5b-v1 --include-links --write --csv ~/sync-links-write.csv
```

```bash
data-orchestrator sync-metadata --collection central-bank-e5b-v1 --include-links --write --csv ~/sync-links-again.csv
```

Expected of the last one: `documents written  : 0`. Then remove the CSVs:

```bash
rm -f ~/sync-links-dry.csv ~/sync-links-write.csv ~/sync-links-again.csv
```

### The `document_url` rename (2026-11)

The vault's column `pdf_url` became `document_url` (the address of the
document itself, whatever its type), and the payload key with it. A fresh
ingestion writes `document_url` and never `pdf_url`. With `--include-links`,
`sync-metadata` owns both: `document_url` is set from the vault, `pdf_url` is
expected absent, so **one `--include-links --write` renames the key** on every
point that carries it (same value, new name), and a second run finds nothing.

Transition: the vault source and `sync-metadata` read the column under
whichever name the vault has (`information_schema`, once per connection:
`document_url`, else `pdf_url`), always as `document_url`. A run before the
vault's DDL renamed the column therefore gives the same answer as a run after
it. To remove once every vault has run the rename.

**Production step** (Mac session, after the vault stack runs its DDL train,
deploy step 6 of the cross-repo spec): the same procedure as above (pre-flight,
dry run, `--write`, second `--write`), with the CSV names of your choice.

Expected on `central-bank-e5b-v1` as it stood on 2026-11-19 (2,566 ledger
documents, 33,601 points, dates and links aligned, `pdf_url` on every point):

| | dry run | `--write` | second `--write` |
|---|---|---|---|
| documents | 2,566 (2,563 updated, 3 undated, 0 unchanged, 0 missing, 0 disagreeing) | 2,566 | 2,566 (2,563 unchanged, 3 undated) |
| to write / written | 2,566: every ledger document, once | 2,566 documents, 33,601 points | 0 |
| keys changing | `document_url` x2566, `pdf_url` x2566, nothing else | | none |

In the CSV: one `document_url` row per document (empty `qdrant_value`, the
address as `vault_value`) and one `pdf_url` row (the same address as
`qdrant_value`, empty `vault_value`), no other field. Another field, or a
`document_url` value that differs from the `pdf_url` value of the same
document, means the points or the vault changed since: stop and look before
`--write`. A document ingested after the orchestrator update already carries
`document_url` and is not written. Vectors and every other key are untouched.

## Checked before this note was written

On PostgreSQL 16 and Qdrant 1.20.0 (`tests/integration/test_sync_metadata_e2e.py`,
and this procedure followed step by step on a small copy of the production
layout: v1 points, the seven v1 indexes, corrected vault rows): the dry run
writes nothing, with a read-only key too; `--write` with a read-only key is
refused before anything is written; `--write` changes only the owned keys,
the vectors are byte-identical (SHA-256 of every vector before and after),
`rag_ingestions` is untouched; a second run writes nothing; a document whose
points disagree is corrected on all of them; a document deleted in the vault
is left alone; an undated vault row loses its date and interval; a date moved
to another year moves `year`; a `date_precision` the engine does not know is
kept and read as a day; a strict-mode collection without the `doc_id` index
is refused with the `--create-indexes` command, which fixes it; an ingestion
and `sync-metadata` exclude each other. After the write, a search in `auto`
mode finds a month-precision document anywhere in its month and an undated
one only with `include_undated`.
