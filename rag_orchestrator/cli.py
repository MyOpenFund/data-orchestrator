"""rag_orchestrator CLI.

Installed as the ``rag-orchestrator`` console script. Standalone repo, sitting
next to ``mvp-graph-rag``::

    rag-orchestrator cb_corpus --count-only
    rag-orchestrator cb_corpus --banks ecb --doctypes C1 --limit 20
    rag-orchestrator cb_corpus --year-min 2015 --collection cb_corpus

(equivalently ``python -m rag_orchestrator.cli cb_corpus ...``)

The default collection is ``cb_corpus`` (kept separate from the demo
``documents`` collection). Ingestion is resumable: a JSON-lines ledger under
``<repo>/state/`` records every ingested document so re-runs skip work already
done. Use ``--no-resume`` to ignore it.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from .core import Ledger, SourceItem, run_ingest, IngestStats
from .sources import cb_corpus as cb_corpus_source

# Ledger lives at the repo root (one level above this package), or wherever
# ``RAGO_STATE_DIR`` points (useful once installed site-wide).
STATE_DIR = Path(
    os.environ.get("RAGO_STATE_DIR", Path(__file__).resolve().parents[1] / "state")
)
DEFAULT_COLLECTION = "cb_corpus"


def _csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


def _build_cb_corpus_items(args: argparse.Namespace):
    return cb_corpus_source.iter_items(
        root=Path(args.root) if args.root else None,
        banks=_csv(args.banks),
        doctypes=_csv(args.doctypes),
        groups=_csv(args.groups),
        year_min=args.year_min,
        year_max=args.year_max,
        include_html=args.include_html,
    )


def _make_progress(every: int):
    t0 = time.time()

    def on_progress(stats: IngestStats, item: SourceItem, status: str) -> None:
        done = stats.docs_ingested + stats.docs_skipped_resume + stats.docs_empty + stats.docs_error
        if status == "error":
            print(f"    ! error  {item.doc_id} — {stats.errors[-1][1]}", file=sys.stderr)
        if done % every == 0 or status == "error":
            rate = stats.docs_ingested / max(time.time() - t0, 1e-6)
            print(
                f"  seen={stats.docs_seen} ingested={stats.docs_ingested} "
                f"skip={stats.docs_skipped_resume} empty={stats.docs_empty} "
                f"err={stats.docs_error} chunks={stats.chunks_written} "
                f"({rate:.2f} docs/s)",
                flush=True,
            )

    return on_progress


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="RAGDataOrchestrator",
        description="Read data from a source and move it into the vector DB.",
    )
    parser.add_argument("source", choices=["cb_corpus"], help="data source to ingest")
    parser.add_argument("--root", help="corpus root (folder containing raw/)")
    parser.add_argument("--banks", help="comma list of bank codes, e.g. ecb,fr")
    parser.add_argument("--doctypes", help="comma list of doc-type codes, e.g. C1,A3")
    parser.add_argument("--groups", help="comma list of doc groups, e.g. A,C")
    parser.add_argument("--year-min", type=int, help="inclusive lower year bound")
    parser.add_argument("--year-max", type=int, help="inclusive upper year bound")
    parser.add_argument("--include-html", action="store_true",
                        help="also ingest .html with no .pdf sibling")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION,
                        help=f"Qdrant collection (default: {DEFAULT_COLLECTION})")
    parser.add_argument("--limit", type=int, help="stop after N newly ingested docs")
    parser.add_argument("--ocr", choices=["auto", "always", "never"], default="auto",
                        help="OCR fallback mode for scanned pages (default: auto)")
    parser.add_argument("--no-resume", action="store_true",
                        help="ignore the resume ledger (re-ingest everything)")
    parser.add_argument("--ledger", help="path to the resume ledger JSONL")
    parser.add_argument("--no-vault", action="store_true",
                        help="use the local file ledger instead of the vault "
                             "(no rag_ingestions state is read or written)")
    parser.add_argument("--corpus", default="central-bank",
                        help="vault corpus this run belongs to (default: central-bank)")
    parser.add_argument("--count-only", action="store_true",
                        help="just count matching documents, do not ingest")
    parser.add_argument("--progress-every", type=int, default=25,
                        help="print progress every N documents (default: 25)")
    args = parser.parse_args(argv)

    if args.source == "cb_corpus":
        items = _build_cb_corpus_items(args)
    else:  # pragma: no cover — argparse restricts choices
        parser.error(f"unknown source: {args.source}")

    if args.count_only:
        by_bank: dict[str, int] = {}
        by_type: dict[str, int] = {}
        total = 0
        for it in items:
            total += 1
            by_bank[it.payload["bank_code"]] = by_bank.get(it.payload["bank_code"], 0) + 1
            by_type[it.payload["doc_type"]] = by_type.get(it.payload["doc_type"], 0) + 1
        print(f"Matching documents: {total}")
        print("  by bank:", dict(sorted(by_bank.items(), key=lambda kv: -kv[1])))
        print("  by doc_type:", dict(sorted(by_type.items())))
        return 0

    ledger = None
    vault_conn = None
    if not args.no_resume:
        if args.no_vault:
            ledger_path = Path(args.ledger) if args.ledger else STATE_DIR / f"{args.collection}.jsonl"
            ledger = Ledger(ledger_path)
            print(f"Resume ledger (file): {ledger_path} ({len(ledger)} docs already done)")
        else:
            from . import vault as vault_mod
            from .routing import EMBEDDING_VERSION, embedding_model_name

            vault_conn = vault_mod.connect()
            ledger = vault_mod.VaultLedger(
                vault_conn, collection=args.collection, corpus=args.corpus,
                embedding_model=embedding_model_name(),
                embedding_version=EMBEDDING_VERSION,
            )
            print(f"Resume ledger (vault): rag_ingestions/{args.collection} "
                  f"({len(ledger)} docs already done)")

    print(f"→ Ingesting source '{args.source}' into collection '{args.collection}' "
          f"(ocr={args.ocr}, limit={args.limit})")
    t0 = time.time()
    stats = run_ingest(
        items,
        collection=args.collection,
        ledger=ledger,
        ocr=args.ocr,
        limit=args.limit,
        on_progress=_make_progress(args.progress_every),
    )
    elapsed = time.time() - t0

    if vault_conn is not None:
        vault_conn.close()

    print("─" * 60)
    print("Done.")
    print(f"  docs seen      : {stats.docs_seen}")
    print(f"  docs ingested  : {stats.docs_ingested}")
    print(f"  skipped(resume): {stats.docs_skipped_resume}")
    print(f"  empty          : {stats.docs_empty}")
    print(f"  errors         : {stats.docs_error}")
    print(f"  chunks written : {stats.chunks_written}")
    print(f"  time           : {elapsed:.1f}s")
    if stats.errors:
        print(f"  first errors   :")
        for path, msg in stats.errors[:10]:
            print(f"    - {path}: {msg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
