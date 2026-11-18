# Parity fixture: the points written before the switch to rag-engine

`expected.json` holds the Qdrant points (id, payload, vector) that `core.ingest_item` wrote
for the six documents of `corpus/` **while it still drove the engine fork**: the
orchestrator's own point ids and payload assembly, the fork's chunker. It was captured
before `ingest_item` became a call to `rag_engine.ingest.ingest_document`.

`tests/test_engine_parity.py` ingests the same documents with today's `ingest_item` and
asserts that the points are the same: same number, same order, same ids, same payloads (keys
and values, `ingestion_date` aside, which is the time of the run), same vectors. It shows
that, for these documents, with a deterministic test embedder and OCR off, the switch changed
nothing in what is written; `MIGRATION.md` ("What was checked") says what that does and does
not cover.

**Schema v2 (rag-engine `6be21ac`, 2026-11).** The engine now appends `date_start` and
`date_end` to every point of a dated document. The fixture was **not** regenerated: it
keeps the v1 points as captured, and the test adds exactly those two keys to them
(`_v2` in the test, computed from the row's `date` without calling the engine) and
requires everything else to be unchanged. The undated row (`jp-f1-2023-rates.csv`) gets
nothing. A second test gives one row a `date_precision` and a `date_kind` in `extra` and
checks that only those two keys and their month interval are added.

**Do not regenerate `expected.json` to make a failing test pass.** A difference means the
orchestrator no longer writes what is stored.

## What is in it

| Document | Format | Points | Why |
|---|---|---|---|
| `ecb-a1-2025-10-30-statement.md` | markdown, French and English | 7, no `page` | headed sections, accents (ChunkNorris stores them decomposed, NFKD) |
| `us-a3-2024-06-12-minutes.txt` | plain text | 2, no `page` | its `extra` holds `filename`, `text`, `chunk_number`, `ingestion_date` |
| `jp-f1-2023-rates.csv` | CSV | 2, no `page` | `title`, `date`, `language` NULL, `extra` NULL |
| `gb-e1-2024-08-01-report.pdf` | PDF, 3 pages | 3, pages 0, 1, 2 | its `extra` repeats `year` and `doc_id` with other values (the columns win) |
| `au-e1-2022-02-statement.pdf` | PDF, 4 pages, the second one blank | 3, pages 0, 2, 3 | a page without text |
| `ca-a3-2023-01-25-deliberations.docx` | Word | 6, no `page` | `source_code` NULL, trailing blanks in `local_path` |

The payloads are built as in production: each document is a fabricated row of the vault's
`documents` table (`ROWS` in `tools/capture_parity_fixture.py`, copied into `expected.json`
as `row`), turned into a `SourceItem` by `sources.vault.row_to_item`. So the metadata holds
the repeated `doc_id`, the vault columns and the keys of `extra`.

`expected.json`: at the top, the commits and versions of the capture, the fake embedder
(`rag_engine.embedding.FakeEmbedder`, 8 dimensions, seed 0, prefixes `passage: ` /
`query: `) and the OCR mode (`never`, so that the result does not depend on the OCR setup of
the host); then, per document, `row`, `written` (what `ingest_item` returned) and `points`
in the order they were written.

The documents are stored byte for byte (`.gitattributes`): the `sha256` of each payload is
the hash of the file as committed.

## How it was captured

On a checkout of this repository at `7196bfc` (the last commit before the switch), with the
engine fork (`MyOpenFund/eigenmind` at `6e9093a`) on `PYTHONPATH`, Python 3.13.18,
chunknorris 1.5.0, PyMuPDF 1.27.2.2, qdrant-client 1.20.0, and rag-engine `fbbe09b`
installed for its `FakeEmbedder` only:

```bash
# Once: writes the documents, then captures.
PYTHONPATH=<fork checkout> <venv>/bin/python tools/capture_parity_fixture.py --make-documents
```

The script only works on a checkout from before the switch: it hands `ingest_item` a
recording store with the shape the previous code wrote to (`store.client.upsert`). To
compare a new capture without touching the fixture, write it elsewhere (`--output`).
PyMuPDF does not write the same bytes twice, so the documents are committed and never
generated again.
