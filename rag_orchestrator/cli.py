"""rag_orchestrator CLI.

Installed as the ``rag-orchestrator`` console script. Standalone repo, sitting
next to ``mvp-graph-rag``::

    rag-orchestrator cb_corpus --count-only
    rag-orchestrator cb_corpus --banks ecb --doctypes C1 --limit 20
    rag-orchestrator cb_corpus --year-min 2015 --collection cb_corpus

    rag-orchestrator bottom_up_corpus --ciks 320193 --collection bottom_up_corpus
    rag-orchestrator bottom_up_corpus --ciks 320193 --doctypes A1 --year-min 2024

(equivalently ``python -m rag_orchestrator.cli <source> ...``)

The default collection matches the source name (e.g. ``cb_corpus``,
``bottom_up_corpus``), kept separate from the demo ``documents`` collection.
Ingestion is resumable: a JSON-lines ledger under ``<repo>/state/`` records every
ingested document so re-runs skip work already done. Use ``--no-resume`` to
ignore it.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from .core import Ledger, SourceItem, run_ingest, IngestStats
from .sources import cb_corpus as cb_corpus_source
from .sources import bottom_up_corpus as bottom_up_corpus_source

# Ledger lives at the repo root (one level above this package), or wherever
# ``RAGO_STATE_DIR`` points (useful once installed site-wide).
STATE_DIR = Path(
    os.environ.get("RAGO_STATE_DIR", Path(__file__).resolve().parents[1] / "state")
)

# Selectable sources. The default Qdrant collection for each is its own name.
SOURCES = ("cb_corpus", "bottom_up_corpus")

# For ``--count-only``: the two payload fields each source is summarised by.
COUNT_KEYS = {
    "cb_corpus": ("bank_code", "doc_type"),
    "bottom_up_corpus": ("cik", "doc_type"),
}


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


def _build_bottom_up_corpus_items(args: argparse.Namespace):
    return bottom_up_corpus_source.iter_items(
        root=Path(args.root) if args.root else None,
        ciks=_csv(args.ciks),
        doctypes=_csv(args.doctypes),
        year_min=args.year_min,
        year_max=args.year_max,
        prefer=args.prefer,
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
    parser.add_argument("source", choices=list(SOURCES), help="data source to ingest")
    parser.add_argument("--root", help="corpus root (folder containing the source's data)")
    parser.add_argument("--doctypes", help="comma list of doc-type codes, e.g. C1,A3 (cb_corpus) or A1 (bottom_up_corpus)")
    parser.add_argument("--year-min", type=int, help="inclusive lower year bound")
    parser.add_argument("--year-max", type=int, help="inclusive upper year bound")
    # cb_corpus-specific selectors
    cb = parser.add_argument_group("cb_corpus options")
    cb.add_argument("--banks", help="comma list of bank codes, e.g. ecb,fr")
    cb.add_argument("--groups", help="comma list of doc groups, e.g. A,C")
    cb.add_argument("--include-html", action="store_true",
                    help="also ingest .html with no .pdf sibling")
    # bottom_up_corpus-specific selectors
    bu = parser.add_argument_group("bottom_up_corpus options")
    bu.add_argument("--ciks", help="comma list of SEC CIK numbers, e.g. 320193,789019")
    bu.add_argument("--prefer", choices=["pdf", "text"], default="pdf",
                    help="artifact to ingest: rendered PDF (default) or cleaned text")
    parser.add_argument("--collection", default=None,
                        help="Qdrant collection (default: the source name)")
    parser.add_argument("--limit", type=int, help="stop after N newly ingested docs")
    parser.add_argument("--ocr", choices=["auto", "always", "never"], default="auto",
                        help="OCR fallback mode for scanned pages (default: auto)")
    parser.add_argument("--no-resume", action="store_true",
                        help="ignore the resume ledger (re-ingest everything)")
    parser.add_argument("--ledger", help="path to the resume ledger JSONL")
    parser.add_argument("--count-only", action="store_true",
                        help="just count matching documents, do not ingest")
    parser.add_argument("--progress-every", type=int, default=25,
                        help="print progress every N documents (default: 25)")
    args = parser.parse_args(argv)

    # Default collection matches the source name (cb_corpus, bottom_up_corpus, ...).
    if args.collection is None:
        args.collection = args.source

    if args.source == "cb_corpus":
        items = _build_cb_corpus_items(args)
    elif args.source == "bottom_up_corpus":
        items = _build_bottom_up_corpus_items(args)
    else:  # pragma: no cover — argparse restricts choices
        parser.error(f"unknown source: {args.source}")

    if args.count_only:
        key1, key2 = COUNT_KEYS[args.source]
        by_key1: dict[str, int] = {}
        by_key2: dict[str, int] = {}
        total = 0
        for it in items:
            total += 1
            v1 = it.payload.get(key1)
            v2 = it.payload.get(key2)
            by_key1[v1] = by_key1.get(v1, 0) + 1
            by_key2[v2] = by_key2.get(v2, 0) + 1
        print(f"Matching documents: {total}")
        print(f"  by {key1}:", dict(sorted(by_key1.items(), key=lambda kv: -kv[1])))
        print(f"  by {key2}:", dict(sorted(by_key2.items(), key=lambda kv: (str(kv[0])))))
        return 0

    ledger = None
    if not args.no_resume:
        ledger_path = Path(args.ledger) if args.ledger else STATE_DIR / f"{args.collection}.jsonl"
        ledger = Ledger(ledger_path)
        print(f"Resume ledger: {ledger_path} ({len(ledger)} docs already done)")

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
