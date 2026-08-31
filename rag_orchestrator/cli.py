"""rag_orchestrator CLI.

Installed as the ``rag-orchestrator`` console script. Standalone repo that
drives the eigenmind engine (a separate editable install, see ``README.md``)::

    rag-orchestrator vault --corpus central-bank
    rag-orchestrator cb_corpus --banks ecb --doctypes C1 --limit 20
    rag-orchestrator cb_corpus --year-min 2015 --collection central-bank-e5b-v1

    rag-orchestrator bottom_up_corpus --ciks 320193 --collection company-e5b-v1
    rag-orchestrator bottom_up_corpus --ciks 320193 --doctypes A1 --year-min 2024

(equivalently ``python -m rag_orchestrator.cli <source> ...``)

The default collection is routed per corpus by
``routing.collection_name`` — ``{corpus}-{model_tag}-v1``, e.g.
``central-bank-e5b-v1`` for the ``cb_corpus``/``vault`` sources or
``company-e5b-v1`` for ``bottom_up_corpus`` — for every source, never a
hard-coded legacy name. Each disk source implies its own vault corpus
(``cb_corpus`` -> ``central-bank``, ``bottom_up_corpus`` -> ``company``), so
``--corpus`` defaults per-source; an explicit ``--corpus`` always wins.

The vault (``rag_ingestions``) is the *default* resume mechanism: a re-run
skips documents already recorded there. Pass ``--no-vault`` to fall back to a
local JSON-lines ledger under ``<repo>/state/`` instead (disk sources —
``cb_corpus``, ``bottom_up_corpus`` — only; the vault source's resume *is*
the vault anti-join in ``sources/vault.py``, so ``--no-vault`` is rejected
there). ``--no-resume`` ignores the ledger and re-ingests everything; it is
likewise rejected for the vault source (its anti-join always filters on the
target collection, so a fresh ``--collection`` name is the way to re-ingest,
not ``--no-resume``).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .core import Ledger, SourceItem, run_ingest, IngestStats
from .routing import collection_name
from .sources import cb_corpus as cb_corpus_source
from .sources import bottom_up_corpus as bottom_up_corpus_source

# Ledger lives at the repo root (one level above this package), or wherever
# ``RAGO_STATE_DIR`` points (useful once installed site-wide).
STATE_DIR = Path(
    os.environ.get("RAGO_STATE_DIR", Path(__file__).resolve().parents[1] / "state")
)
# Selectable sources.
SOURCES = ("cb_corpus", "bottom_up_corpus", "vault", "probe")

# For ``--count-only``: the two payload fields each disk source is summarised
# by (the vault source has its own count path — source_code/doc_type — in
# _count_vault_source).
COUNT_KEYS = {
    "cb_corpus": ("bank_code", "doc_type"),
    "bottom_up_corpus": ("cik", "doc_type"),
}

# Each disk source implies its own vault corpus, so --corpus can default
# per-source instead of one hard-coded value; an explicit --corpus always
# wins. Sources absent here (vault, probe) keep the "central-bank" default.
IMPLIED_CORPUS = {
    "cb_corpus": "central-bank",
    "bottom_up_corpus": "company",
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


def _print_summary(stats: IngestStats, elapsed: float | None = None) -> None:
    print("─" * 60)
    print("Done.")
    print(f"  docs seen      : {stats.docs_seen}")
    print(f"  docs ingested  : {stats.docs_ingested}")
    print(f"  skipped(resume): {stats.docs_skipped_resume}")
    print(f"  empty          : {stats.docs_empty}")
    print(f"  errors         : {stats.docs_error}")
    print(f"  chunks written : {stats.chunks_written}")
    if elapsed is not None:
        print(f"  time           : {elapsed:.1f}s")
    if stats.errors:
        print(f"  first errors   :")
        for path, msg in stats.errors[:10]:
            print(f"    - {path}: {msg}")


# ---------------------------------------------------------------------------
# Run reports — the shared contract shape written to the vault's ``runs``
# table (or, with --no-vault, appended to STATE_DIR/runs.jsonl).
#
# Doctrine (fixed): degraded/exit 3 iff docs_ingested == 0 and docs_error > 0
# (a run that touched documents and got none of them in); otherwise ok/0. A
# fatal exception is reported separately as failed/exit 1 (_fatal_report).
# ---------------------------------------------------------------------------
def _build_report(command: str, stats: IngestStats, started_at: str) -> dict:
    totals = {
        "docs_seen": stats.docs_seen,
        "docs_new": stats.docs_ingested,
        "docs_failed": stats.docs_error,
    }
    sources = [
        {"source_code": code, **counts}
        for code, counts in sorted(stats.by_source.items())
    ]
    degraded = stats.docs_ingested == 0 and stats.docs_error > 0
    return {
        "run_id": str(uuid.uuid4()),
        "tool": "rag-orchestrator",
        "command": command,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "outcome": "degraded" if degraded else "ok",
        "exit_code": 3 if degraded else 0,
        "totals": totals,
        "sources": sources,
    }


def _fatal_report(command: str, started_at: str, exc: Exception) -> dict:
    """Best-effort report for a run that never reached a normal IngestStats
    outcome (e.g. the vault connection or the engine itself failed)."""
    rep = _build_report(command, IngestStats(), started_at)
    rep["outcome"] = "failed"
    rep["exit_code"] = 1
    rep["error"] = f"{type(exc).__name__}: {exc}"
    return rep


def _append_report_jsonl(report: dict) -> None:
    """--no-vault fallback: a single atomic append to STATE_DIR/runs.jsonl."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with (STATE_DIR / "runs.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(report) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="RAGDataOrchestrator",
        description="Read data from a source and move it into the vector DB.",
    )
    parser.add_argument("source", choices=list(SOURCES), help="data source to ingest")
    parser.add_argument("--root", help="corpus root (folder containing raw/, or the source's data dir)")
    parser.add_argument("--banks", help="comma list of bank codes, e.g. ecb,fr")
    parser.add_argument("--doctypes", help="comma list of doc-type codes, e.g. C1,A3 (cb_corpus) or A1 (bottom_up_corpus)")
    parser.add_argument("--groups", help="comma list of doc groups, e.g. A,C")
    parser.add_argument("--ciks", help="comma list of SEC CIK numbers, e.g. 320193,789019 (bottom_up_corpus source)")
    parser.add_argument("--prefer", choices=["pdf", "text"], default="pdf",
                        help="(bottom_up_corpus source) artifact to ingest: rendered PDF "
                             "(default) or cleaned text")
    parser.add_argument("--source-codes", help="comma list of vault source codes, e.g. ecb,fr")
    parser.add_argument("--languages", help="comma list of vault language codes, e.g. en,fr")
    parser.add_argument("--year-min", type=int, help="inclusive lower year bound")
    parser.add_argument("--year-max", type=int, help="inclusive upper year bound")
    parser.add_argument("--include-html", action="store_true",
                        help="also ingest .html with no .pdf sibling")
    parser.add_argument("--collection", default=None,
                        help="Qdrant collection (default: {corpus}-{tag}-v1 "
                             "via routing.collection_name — same resolution "
                             "for every source, using --corpus or the "
                             "source's implied corpus)")
    parser.add_argument("--limit", type=int, help="stop after N newly ingested docs")
    parser.add_argument("--ocr", choices=["auto", "always", "never"], default="auto",
                        help="OCR fallback mode for scanned pages (default: auto)")
    parser.add_argument("--no-resume", action="store_true",
                        help="ignore the resume ledger (re-ingest everything)")
    parser.add_argument("--ledger", help="path to the resume ledger JSONL")
    parser.add_argument("--no-vault", action="store_true",
                        help="use the local file ledger instead of the vault "
                             "(no rag_ingestions state is read or written)")
    parser.add_argument("--corpus", default=None,
                        help="vault corpus this run belongs to (default: per-source — "
                             "central-bank for cb_corpus/vault/probe, company for "
                             "bottom_up_corpus)")
    parser.add_argument("--count-only", action="store_true",
                        help="just count matching documents, do not ingest")
    parser.add_argument("--progress-every", type=int, default=25,
                        help="print progress every N documents (default: 25)")
    args = parser.parse_args(argv)

    # Guard against invalid --progress-every early, before any engine/DB work.
    if getattr(args, "progress_every", 1) <= 0:
        print("error: --progress-every must be >= 1", file=sys.stderr)
        return 2

    # Each disk source implies its own vault corpus; an explicit --corpus
    # always wins. Never a source-name collection fallback (item 3/4 of the
    # bottom_up_corpus integration) — collection routing is always
    # routing.collection_name(args.corpus) below.
    if args.corpus is None:
        args.corpus = IMPLIED_CORPUS.get(args.source, "central-bank")

    if args.source == "cb_corpus":
        items = _build_cb_corpus_items(args)
    elif args.source == "bottom_up_corpus":
        items = _build_bottom_up_corpus_items(args)

    if args.source == "vault":
        if args.no_vault:
            print("error: the vault source requires the vault (drop --no-vault)",
                  file=sys.stderr)
            return 2
        if args.no_resume:
            print(
                "error: --no-resume is not supported for the vault source — "
                "its resume IS the documents/rag_ingestions anti-join (a doc "
                "already recorded for the target collection is never "
                "selected again, --no-resume or not). To re-ingest, use a "
                "fresh --collection name instead.",
                file=sys.stderr,
            )
            return 2
        # Validate corpus root BEFORE any engine/DB work.
        from .routing import corpus_root
        try:
            corpus_root(args.corpus)
        except (RuntimeError, KeyError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        collection = args.collection or collection_name(args.corpus)
        if args.count_only:
            return _count_vault_source(args, collection)
        return _run_vault_source(args, collection)

    if args.source == "probe":
        from . import vault as vault_mod
        from .probe import run_probe

        conn = vault_mod.connect()
        try:
            stats = run_probe(conn, args.corpus, limit=args.limit)
        finally:
            conn.close()
        print(f"Probe done: {stats}")
        return 0

    # Same routing.collection_name resolution as the vault source (default
    # args.corpus is "central-bank") — never the legacy hard-coded 384-d
    # "cb_corpus" collection name, which a 768-d upsert would be rejected
    # from on any machine that still has it.
    collection = args.collection or collection_name(args.corpus)

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
    vault_conn = None
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        if not args.no_resume:
            if args.no_vault:
                ledger_path = Path(args.ledger) if args.ledger else STATE_DIR / f"{collection}.jsonl"
                ledger = Ledger(ledger_path)
                print(f"Resume ledger (file): {ledger_path} ({len(ledger)} docs already done)")
            else:
                from . import vault as vault_mod
                from .routing import EMBEDDING_VERSION, embedding_model_name

                vault_conn = vault_mod.connect()
                ledger = vault_mod.VaultLedger(
                    vault_conn, collection=collection, corpus=args.corpus,
                    embedding_model=embedding_model_name(),
                    embedding_version=EMBEDDING_VERSION,
                )
                print(f"Resume ledger (vault): rag_ingestions/{collection} "
                      f"({len(ledger)} docs already done)")

        print(f"→ Ingesting source '{args.source}' into collection '{collection}' "
              f"(ocr={args.ocr}, limit={args.limit})")
        t0 = time.time()
        stats = run_ingest(
            items,
            collection=collection,
            ledger=ledger,
            ocr=args.ocr,
            limit=args.limit,
            on_progress=_make_progress(args.progress_every),
        )
        elapsed = time.time() - t0
        _print_summary(stats, elapsed)

        rep = _build_report(args.source, stats, started_at)
        if args.no_vault:
            _append_report_jsonl(rep)
        elif vault_conn is not None:
            try:
                vault_mod.insert_run_report(vault_conn, rep)
            except Exception as exc:  # noqa: BLE001 — report write never masks the run's own outcome
                print(f"warning: failed to write run report to vault: {exc}", file=sys.stderr)
        else:
            # Vault mode (not --no-vault) but no ledger connection was ever
            # opened (--no-resume skips the "if not args.no_resume" block
            # above) — open a short-lived one just for the report insert so
            # the run isn't silently unreported. Same warn-only contract: a
            # report failure must never mask the run's own outcome.
            from . import vault as vault_mod

            report_conn = None
            try:
                report_conn = vault_mod.connect()
                vault_mod.insert_run_report(report_conn, rep)
            except Exception as exc:  # noqa: BLE001 — report write never masks the run's own outcome
                print(f"warning: failed to write run report to vault: {exc}", file=sys.stderr)
            finally:
                if report_conn is not None:
                    report_conn.close()
        return rep["exit_code"]
    except Exception as exc:  # noqa: BLE001 — fatal: best-effort report, honest exit code
        rep = _fatal_report(args.source, started_at, exc)
        try:
            if args.no_vault:
                _append_report_jsonl(rep)
            elif vault_conn is not None:
                vault_mod.insert_run_report(vault_conn, rep)
        except Exception:
            pass
        print(f"error: fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        return rep["exit_code"]
    finally:
        if vault_conn is not None:
            vault_conn.close()


def _run_vault_source(args, collection: str) -> int:
    from . import vault as vault_mod
    from .routing import EMBEDDING_VERSION, embedding_model_name
    from .sources import vault as vault_source

    started_at = datetime.now(timezone.utc).isoformat()
    conn = None
    try:
        conn = vault_mod.connect()
        # --no-resume is rejected for this source before we ever get here
        # (main()), so the ledger — and therefore the anti-join's resume
        # semantics — is always coherent: always on.
        ledger = vault_mod.VaultLedger(
            conn, collection=collection, corpus=args.corpus,
            embedding_model=embedding_model_name(),
            embedding_version=EMBEDDING_VERSION,
        )
        print(f"Resume ledger (vault): rag_ingestions/{collection} "
              f"({len(ledger)} docs already done)")
        items = vault_source.iter_items(
            conn, args.corpus, collection,
            source_codes=_csv(args.source_codes),
            doctypes=_csv(args.doctypes),
            year_min=args.year_min, year_max=args.year_max,
            languages=_csv(args.languages),
        )
        print(f"→ Ingesting vault selection (corpus={args.corpus}) into "
              f"'{collection}' (ocr={args.ocr}, limit={args.limit})")
        stats = run_ingest(
            items, collection=collection, ledger=ledger, ocr=args.ocr,
            limit=args.limit, on_progress=_make_progress(args.progress_every),
        )
        _print_summary(stats)

        rep = _build_report("vault", stats, started_at)
        try:
            vault_mod.insert_run_report(conn, rep)
        except Exception as exc:  # noqa: BLE001 — report write never masks the run's own outcome
            print(f"warning: failed to write run report to vault: {exc}", file=sys.stderr)
        return rep["exit_code"]
    except Exception as exc:  # noqa: BLE001 — fatal: best-effort report, honest exit code
        rep = _fatal_report("vault", started_at, exc)
        if conn is not None:
            try:
                vault_mod.insert_run_report(conn, rep)
            except Exception:
                pass
        print(f"error: fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        return rep["exit_code"]
    finally:
        if conn is not None:
            conn.close()


def _count_vault_source(args, collection: str) -> int:
    """Count vault-selected documents without any engine work or ledger writes.

    Mirrors the cb_corpus ``--count-only`` output format (total + breakdowns),
    using ``source_code``/``doc_type`` — the vault source's payload columns —
    in place of cb_corpus's ``bank_code``/``doc_type``.
    """
    from . import vault as vault_mod
    from .sources import vault as vault_source

    conn = vault_mod.connect()
    try:
        items = vault_source.iter_items(
            conn, args.corpus, collection,
            source_codes=_csv(args.source_codes),
            doctypes=_csv(args.doctypes),
            year_min=args.year_min, year_max=args.year_max,
            languages=_csv(args.languages),
        )
        by_source: dict[str, int] = {}
        by_type: dict[str, int] = {}
        total = 0
        for it in items:
            total += 1
            source_code = it.payload.get("source_code", "<unknown>")
            doc_type = it.payload.get("doc_type", "<unknown>")
            by_source[source_code] = by_source.get(source_code, 0) + 1
            by_type[doc_type] = by_type.get(doc_type, 0) + 1
    finally:
        conn.close()
    print(f"Matching documents: {total}")
    print("  by source_code:", dict(sorted(by_source.items(), key=lambda kv: -kv[1])))
    print("  by doc_type:", dict(sorted(by_type.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
