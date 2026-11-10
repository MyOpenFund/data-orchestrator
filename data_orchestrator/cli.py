"""data_orchestrator CLI.

Installed as the ``data-orchestrator`` console script. Standalone repo that
drives the eigenmind engine (a separate editable install, see ``README.md``)::

    data-orchestrator vault --corpus central-bank
    data-orchestrator cb_corpus --banks ecb --doctypes C1 --limit 20
    data-orchestrator cb_corpus --year-min 2015 --collection central-bank-e5b-v1

    data-orchestrator company --ciks 320193 --collection company-e5b-v1
    data-orchestrator company --ciks 320193 --doctypes A1 --year-min 2024

(equivalently ``python -m data_orchestrator.cli <source> ...``)

The default collection is routed per corpus by
``routing.collection_name`` — ``{corpus}-{model_tag}-v1``, e.g.
``central-bank-e5b-v1`` for the ``cb_corpus``/``vault`` sources or
``company-e5b-v1`` for ``company`` — for every source, never a
hard-coded legacy name. Each disk source implies its own vault corpus
(``cb_corpus`` -> ``central-bank``, ``company`` -> ``company``), so
``--corpus`` defaults per-source; an explicit ``--corpus`` always wins.

The vault (``rag_ingestions``) is the *default* resume mechanism: a re-run
skips documents already recorded there. Pass ``--no-vault`` to fall back to a
local JSON-lines ledger under ``<repo>/state/`` instead (disk sources —
``cb_corpus``, ``company`` — only; the vault source's resume *is*
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

from . import config, policy
from .core import Ledger, SourceItem, run_ingest, IngestStats, source_label
from .routing import collection_name, supported_mime_types
from .sources import cb_corpus as cb_corpus_source
from .sources import company_corpus as company_corpus_source


def state_dir() -> Path:
    """Directory of the ``--no-vault`` file ledger and run reports.

    ``DATA_ORCHESTRATOR_STATE_DIR`` (useful once installed site-wide), else
    ``<repo>/state`` (one level above this package). Read at call time, never
    at import: the key may come from the ``.env`` that ``main()`` loads first.
    """
    value = os.environ.get("DATA_ORCHESTRATOR_STATE_DIR")
    if value:
        return Path(value).expanduser()  # .env values are literal: no shell ~
    return Path(__file__).resolve().parents[1] / "state"


# Selectable sources.
SOURCES = ("cb_corpus", "company", "vault", "probe")

# For ``--count-only``: the two payload fields each disk source is summarised
# by (the vault source has its own count path — source_code/doc_type — in
# _count_vault_source).
COUNT_KEYS = {
    "cb_corpus": ("bank_code", "doc_type"),
    "company": ("cik", "doc_type"),
}

# Each disk source implies its own vault corpus, so --corpus can default
# per-source instead of one hard-coded value; an explicit --corpus always
# wins. Sources absent here (vault, probe) keep the "central-bank" default.
IMPLIED_CORPUS = {
    "cb_corpus": "central-bank",
    "company": "company",
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
    )


def _build_company_items(args: argparse.Namespace):
    return company_corpus_source.iter_items(
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
    print(f"  path-metadata  : {stats.docs_path_metadata}")
    print(f"  chunks written : {stats.chunks_written}")
    if elapsed is not None:
        print(f"  time           : {elapsed:.1f}s")
    if stats.errors:
        print(f"  first errors   :")
        for path, msg in stats.errors[:10]:
            print(f"    - {path}: {msg}")
    if stats.docs_path_metadata:
        print(
            f"warning: {stats.docs_path_metadata} document(s) were not indexed in the "
            "cb_corpus manifest and carry path-derived metadata (year-only dates, no "
            "title/sha256). Run 'reindex-from-disk' in cb_corpus, then re-ingest them "
            "with --no-resume (disk source) or into a fresh --collection.",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------
# Run reports — the shared contract shape written to the vault's ``runs``
# table (or, with --no-vault, appended to state_dir()/runs.jsonl).
#
# The ok/degraded rule (incl. the failure-ratio threshold) lives in
# ``policy.py``; a fatal exception is reported separately as failed/exit 1
# (_fatal_report).
# ---------------------------------------------------------------------------

# Same cap as central-bank-corpus runreport.py (the other writer of runs.extra->>'fatal').
FATAL_MAX_CHARS = 500


def _build_report(command: str, stats: IngestStats, started_at: str, *,
                  corpus: str, ratio: float,
                  unsupported: dict[str, int] | None = None) -> dict:
    totals = {
        "docs_seen": stats.docs_seen,
        "docs_new": stats.docs_ingested,
        "docs_failed": stats.docs_error,
        "docs_path_metadata": stats.docs_path_metadata,
    }
    # sources[] lists only the source codes the run touched (stats.by_source):
    # a source with only unsupported documents is not a source the run
    # worked on, so it never appears there (it would read as a run of that
    # source in the vault's per-source views). Its count lives in the
    # top-level ``unsupported_by_source`` (no runs column: swept into
    # runs.extra), written whenever the count is known, even empty.
    by_source = {code: dict(counts) for code, counts in stats.by_source.items()}
    if unsupported is not None:
        totals["docs_unsupported"] = sum(unsupported.values())
        for code, counts in by_source.items():
            counts["docs_unsupported"] = unsupported.get(code, 0)
    sources = [{"source_code": code, **counts} for code, counts in sorted(by_source.items())]
    outcome, exit_code = policy.run_outcome(
        docs_ingested=stats.docs_ingested, docs_empty=stats.docs_empty,
        docs_failed=stats.docs_error, ratio=ratio,
    )
    report = {
        "run_id": str(uuid.uuid4()),
        "tool": "data-orchestrator",
        "command": command,
        "corpus": corpus,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "outcome": outcome,
        "exit_code": exit_code,
        "totals": totals,
        "sources": sources,
    }
    if unsupported is not None:
        report["unsupported_by_source"] = dict(sorted(unsupported.items()))
    return report


def _fatal_report(command: str, started_at: str, exc: Exception, *, corpus: str,
                  unsupported: dict[str, int] | None = None) -> dict:
    """Best-effort report for a run that never reached a normal outcome.

    The reason goes under ``fatal`` (runs.extra->>'fatal'), the key
    central-bank-corpus already writes; it was ``error`` before 2026-11.
    ``unsupported`` (the vault source's per-source count, when it was
    computed before the failure) is passed through so a known
    ``docs_unsupported`` and ``unsupported_by_source`` are not lost;
    ``sources[]`` stays empty (the run touched no source).
    """
    # The ratio is irrelevant to a fatal outcome (overwritten below); the
    # default is stated explicitly since _build_report has no default.
    rep = _build_report(command, IngestStats(), started_at, corpus=corpus,
                        ratio=policy.DEFAULT_DEGRADED_FAILURE_RATIO,
                        unsupported=unsupported)
    rep["outcome"] = "failed"
    rep["exit_code"] = 1
    rep["fatal"] = f"{type(exc).__name__}: {exc}"[:FATAL_MAX_CHARS]
    return rep


def _insert_fatal_report(vault_mod, conn, report: dict) -> None:
    """Write a fatal report on a connection the failure may have left inside
    an aborted transaction: roll back first (a failing rollback is not
    itself fatal; the insert then says whether the connection is usable),
    and warn on an insert failure — never silent, never masking the run's
    own fatal outcome."""
    try:
        conn.rollback()
    except Exception:  # noqa: BLE001 — the insert below reports a dead connection
        pass
    try:
        vault_mod.insert_run_report(conn, report)
    except Exception as exc:  # noqa: BLE001 — report write never masks the run's own outcome
        print(f"warning: failed to write run report to vault: {exc}", file=sys.stderr)


def _append_report_jsonl(report: dict) -> None:
    """--no-vault fallback: a single atomic append to state_dir()/runs.jsonl."""
    folder = state_dir()
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "runs.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(report) + "\n")


# eigenmind's own defaults (eigenmind/config.py qdrant_host/qdrant_port),
# used only to display a store that reports no host/port. 6333 is the LEGACY
# store.
QDRANT_DEFAULT_HOST = "localhost"
QDRANT_DEFAULT_PORT = 6333


def _print_qdrant_target(host, port) -> None:
    """``run_ingest``'s ``on_target``: print the Qdrant the constructed store
    writes to — resolved by eigenmind AFTER its own python-dotenv ran, so the
    line is right even when that loader set QDRANT_HOST/QDRANT_PORT — and warn
    on the legacy port. A store that reports no host or port is displayed
    with eigenmind's default (an invalid or empty ``QDRANT_PORT`` in the
    environment never gets here: ``_qdrant_port_error`` refuses it before
    any work); the port is compared numerically (``06333`` is 6333)."""
    host = str(host).strip() if host is not None else ""
    host = host or QDRANT_DEFAULT_HOST
    raw = str(port).strip() if port is not None else ""
    try:
        port = int(raw) if raw else QDRANT_DEFAULT_PORT
    except ValueError:
        port = raw  # displayed as is; cannot be the legacy port
    print(f"Qdrant: {host}:{port}", flush=True)
    if port == QDRANT_DEFAULT_PORT:
        print(f"warning: Qdrant port {port} is the legacy store (eigenmind's default); "
              "set QDRANT_PORT to the dedicated one unless this is intended",
              file=sys.stderr)


QDRANT_PORT_ENV = "QDRANT_PORT"


def _qdrant_port_error(environ=None) -> str | None:
    """Pre-flight for an ingest run: the error message when ``QDRANT_PORT`` is
    set but is not a plain integer in 1..65535 (surrounding whitespace
    ignored; empty is invalid, ``.env`` values are literal so an inline
    ``# comment`` is part of the value), else ``None``. Absent is fine
    (eigenmind's default, warned about by ``_print_qdrant_target``)."""
    environ = os.environ if environ is None else environ
    raw = environ.get(QDRANT_PORT_ENV)
    if raw is None:
        return None
    text = raw.strip()
    if text.isascii() and text.isdigit() and 1 <= int(text) <= 65535:
        return None
    return (f"{QDRANT_PORT_ENV}={raw!r} is not a port number (expected a plain "
            "integer 1..65535; unset it to use eigenmind's default)")


def _validate_corpus_root(corpus: str) -> int | None:
    """Eagerly resolve ``routing.corpus_root(corpus)`` before any ledger/engine
    work starts. Prints an ``error:`` line to stderr and returns the exit code
    the caller should return on failure; returns ``None`` on success.

    Kept separate from the ``--root`` override check (a disk source's explicit
    ``--root`` bypasses the env/routing lookup entirely — see its dispatch).
    """
    from .routing import corpus_root, ROUTING

    try:
        corpus_root(corpus)
        return None
    except KeyError:
        known = ", ".join(sorted(ROUTING))
        print(f"error: unknown corpus '{corpus}' (known: {known})", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="data-orchestrator",
        description="Read data from a source and move it into the vector DB.",
    )
    parser.add_argument("source", choices=list(SOURCES), help="data source to ingest")
    parser.add_argument("--root", help="corpus root (folder containing raw/, or the source's data dir)")
    parser.add_argument("--banks", help="comma list of bank codes, e.g. ecb,fr")
    parser.add_argument("--doctypes", help="comma list of doc-type codes, e.g. C1,A3 (cb_corpus) or A1 (company)")
    parser.add_argument("--groups", help="comma list of doc groups, e.g. A,C")
    parser.add_argument("--ciks", help="comma list of SEC CIK numbers, e.g. 320193,789019 (company source)")
    parser.add_argument("--prefer", choices=["pdf", "text"], default="pdf",
                        help="(company source) artifact to ingest: rendered PDF "
                             "(default) or cleaned text")
    parser.add_argument("--source-codes", help="comma list of vault source codes, e.g. ecb,fr")
    parser.add_argument("--languages", help="comma list of vault language codes, e.g. en,fr")
    parser.add_argument("--year-min", type=int, help="inclusive lower year bound")
    parser.add_argument("--year-max", type=int, help="inclusive upper year bound")
    parser.add_argument("--collection", default=None,
                        help="Qdrant collection (default: {corpus}-{tag}-v1 "
                             "via routing.collection_name — same resolution "
                             "for every source, using --corpus or the "
                             "source's implied corpus)")
    parser.add_argument("--limit", type=int, help="stop after N newly ingested docs")
    parser.add_argument("--ocr", choices=["auto", "always", "never"], default="auto",
                        help="OCR fallback mode for scanned pages (default: auto)")
    parser.add_argument("--no-resume", action="store_true",
                        help="ignore the resume ledger (disk sources only). Rejected "
                             "for the vault source: its resume is the anti-join; "
                             "re-ingest into a fresh --collection instead (a re-run "
                             "into the same collection would leave stale chunks "
                             "behind).")
    parser.add_argument("--ledger", help="path to the resume ledger JSONL")
    parser.add_argument("--no-vault", action="store_true",
                        help="use the local file ledger instead of the vault "
                             "(no rag_ingestions state is read or written)")
    parser.add_argument("--corpus", default=None,
                        help="vault corpus this run belongs to (default: per-source — "
                             "central-bank for cb_corpus/vault/probe, company for "
                             "company)")
    parser.add_argument("--count-only", action="store_true",
                        help="just count matching documents, do not ingest")
    parser.add_argument("--progress-every", type=int, default=25,
                        help="print progress every N documents (default: 25)")
    args = parser.parse_args(argv)
    # The one .env loader (issue #8), run before anything can import eigenmind:
    # eigenmind's own python-dotenv (override=False) runs at its first import
    # and can then only add keys this file did not set. No $VAR interpolation
    # here — values are taken literally, surrounding quotes stripped.
    # DATA_ORCHESTRATOR_DOTENV (real environment only) names the file
    # explicitly; a missing one is an operator error, before any work.
    try:
        config.load_dotenv()
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    loaded = config.loaded_dotenv()
    print(f"Config: .env = {loaded}" if loaded else "Config: no .env file found")

    # Guard against invalid --progress-every early, before any engine/DB work.
    if getattr(args, "progress_every", 1) <= 0:
        print("error: --progress-every must be >= 1", file=sys.stderr)
        return 2
    if args.limit is not None and args.limit < 1:
        print("error: --limit must be >= 1", file=sys.stderr)
        return 2
    # Run policy, read once from the (now loaded) environment: an invalid
    # value is an operator error (exit 2), caught before any DB/engine work.
    try:
        args.degraded_ratio = policy.degraded_failure_ratio()
        args.supported_mimes = supported_mime_types()
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.source == "probe":
        unsupported_flags = [flag for flag, given in (
            ("--count-only", args.count_only), ("--no-vault", args.no_vault),
            ("--no-resume", args.no_resume), ("--ledger", args.ledger is not None),
        ) if given]
        if unsupported_flags:
            print(f"error: {', '.join(unsupported_flags)} not supported by probe "
                  "(it always reads and writes the vault, and is resumable by "
                  "construction)", file=sys.stderr)
            return 2
    elif not args.count_only:
        # An ingest run: the engine will read QDRANT_PORT. Refuse a value it
        # cannot use before any DB/engine work, rather than fail mid-run.
        port_error = _qdrant_port_error()
        if port_error is not None:
            print(f"error: {port_error}", file=sys.stderr)
            return 2

    # Each disk source implies its own vault corpus; an explicit --corpus
    # always wins. Never a source-name collection fallback — collection
    # routing is always
    # routing.collection_name(args.corpus) below.
    if args.corpus is None:
        args.corpus = IMPLIED_CORPUS.get(args.source, "central-bank")

    if args.source == "cb_corpus":
        # Eager pre-flight: cb_corpus's iter_items() is a generator, so a
        # missing/misconfigured root only surfaces at its first next() —
        # after the ledger/vault connection and engine are already spun up
        # and an orphan collection may have been created (issue #7). Fail
        # here instead, before any of that. An explicit --root overrides the
        # env entirely (see _build_cb_corpus_items), so it gets its own
        # existence check rather than the routing/env lookup.
        if args.root:
            resolved_root = Path(args.root)
            if not resolved_root.is_dir():
                print(f"error: --root {args.root} does not exist or is not a directory",
                      file=sys.stderr)
                return 1
        else:
            err = _validate_corpus_root(args.corpus)
            if err is not None:
                return err
            from .routing import corpus_root

            resolved_root = corpus_root(args.corpus)
        # Same eager-pre-flight reasoning as the root check above:
        # _build_cb_corpus_items() never passes prefer_manifest, so
        # iter_items() always requires the per-bank manifest. Check for it
        # now too, before the ledger/vault connection and engine spin up.
        try:
            cb_corpus_source.require_manifest(resolved_root)
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        items = _build_cb_corpus_items(args)
    elif args.source == "company":
        items = _build_company_items(args)

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
                "fresh --collection name instead: a re-run into the same "
                "collection would leave stale chunks behind (a new chunking "
                "can yield fewer chunks than the points already stored).",
                file=sys.stderr,
            )
            return 2
        # Validate corpus root BEFORE any engine/DB work.
        err = _validate_corpus_root(args.corpus)
        if err is not None:
            return err
        collection = args.collection or collection_name(args.corpus)
        if args.count_only:
            return _count_vault_source(args, collection)
        return _run_vault_source(args, collection)

    if args.source == "probe":
        from . import vault as vault_mod
        from .probe import run_probe

        conn = None
        try:
            conn = vault_mod.connect()
            stats = run_probe(conn, args.corpus, limit=args.limit)
        except Exception as exc:  # noqa: BLE001 — a clean fatal line, never a traceback
            print(f"error: fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        finally:
            if conn is not None:
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
                ledger_path = Path(args.ledger) if args.ledger else state_dir() / f"{collection}.jsonl"
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
            on_target=_print_qdrant_target,
        )
        elapsed = time.time() - t0
        _print_summary(stats, elapsed)

        rep = _build_report(args.source, stats, started_at, corpus=args.corpus,
                            ratio=args.degraded_ratio)
        if args.no_vault:
            try:
                _append_report_jsonl(rep)
            except Exception as exc:  # noqa: BLE001 — report write never masks the run's own outcome
                print(f"warning: failed to append run report: {exc}", file=sys.stderr)
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
        rep = _fatal_report(args.source, started_at, exc, corpus=args.corpus)
        if args.no_vault:
            try:
                _append_report_jsonl(rep)
            except Exception as report_exc:  # noqa: BLE001 — never masks the run's own outcome
                print(f"warning: failed to append run report: {report_exc}", file=sys.stderr)
        elif vault_conn is not None:
            _insert_fatal_report(vault_mod, vault_conn, rep)
        else:
            # Same short-lived-connection pattern as the success path above:
            # vault mode, but no ledger connection was ever opened (e.g. a
            # crash before/under --no-resume). Best-effort connect just for
            # the report insert, warn-only — a report failure must never
            # mask the run's own (fatal) outcome or exit code.
            from . import vault as vault_mod

            report_conn = None
            try:
                report_conn = vault_mod.connect()
                vault_mod.insert_run_report(report_conn, rep)
            except Exception as report_exc:  # noqa: BLE001
                print(f"warning: failed to write run report to vault: {report_exc}", file=sys.stderr)
            finally:
                if report_conn is not None:
                    report_conn.close()
        print(f"error: fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        return rep["exit_code"]
    finally:
        if vault_conn is not None:
            vault_conn.close()


def _vault_filters(args) -> dict:
    """Selection filters shared by the vault selection and its unsupported count."""
    return dict(
        source_codes=_csv(args.source_codes), doctypes=_csv(args.doctypes),
        year_min=args.year_min, year_max=args.year_max,
        languages=_csv(args.languages), supported_mimes=args.supported_mimes,
    )


def _run_vault_source(args, collection: str) -> int:
    from . import vault as vault_mod
    from .routing import EMBEDDING_VERSION, embedding_model_name
    from .sources import vault as vault_source

    started_at = datetime.now(timezone.utc).isoformat()
    conn = None
    unsupported = None  # known once count_unsupported ran; kept by a fatal report
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
        # One filter set for both queries, so the unsupported count is
        # exactly the complement of the selection (same candidates).
        filters = _vault_filters(args)
        unsupported = vault_source.count_unsupported(conn, args.corpus, collection, **filters)
        n_unsupported = sum(unsupported.values())
        if n_unsupported:
            print(f"Skipping {n_unsupported} document(s) of a type the engine cannot read "
                  "(see DATA_ORCHESTRATOR_SUPPORTED_MIME_TYPES and the README's "
                  "'Supported document types')")
        items = vault_source.iter_items(conn, args.corpus, collection, **filters)
        print(f"→ Ingesting vault selection (corpus={args.corpus}) into "
              f"'{collection}' (ocr={args.ocr}, limit={args.limit})")
        stats = run_ingest(
            items, collection=collection, ledger=ledger, ocr=args.ocr,
            limit=args.limit, on_progress=_make_progress(args.progress_every),
            on_target=_print_qdrant_target,
        )
        _print_summary(stats)

        rep = _build_report("vault", stats, started_at, corpus=args.corpus,
                            ratio=args.degraded_ratio, unsupported=unsupported)
        try:
            vault_mod.insert_run_report(conn, rep)
        except Exception as exc:  # noqa: BLE001 — report write never masks the run's own outcome
            print(f"warning: failed to write run report to vault: {exc}", file=sys.stderr)
        return rep["exit_code"]
    except Exception as exc:  # noqa: BLE001 — fatal: best-effort report, honest exit code
        rep = _fatal_report("vault", started_at, exc, corpus=args.corpus,
                            unsupported=unsupported)
        if conn is not None:
            _insert_fatal_report(vault_mod, conn, rep)
        print(f"error: fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        return rep["exit_code"]
    finally:
        if conn is not None:
            conn.close()


def _count_vault_source(args, collection: str) -> int:
    """Count vault-selected documents without any engine work or ledger writes.

    Mirrors the cb_corpus ``--count-only`` output format (total + breakdowns),
    using ``source_code``/``doc_type`` — the vault source's payload columns —
    in place of cb_corpus's ``bank_code``/``doc_type``. Both source breakdowns
    use ``core.source_label`` (a NULL source_code reads as the corpus), the
    label the run report's ``sources[]`` uses; a NULL doc_type reads
    ``unknown``. No Qdrant line: a count never touches Qdrant.
    """
    from . import vault as vault_mod
    from .sources import vault as vault_source

    conn = None
    try:
        conn = vault_mod.connect()
        filters = _vault_filters(args)
        unsupported = vault_source.count_unsupported(conn, args.corpus, collection, **filters)
        items = vault_source.iter_items(conn, args.corpus, collection, **filters)
        by_source: dict[str, int] = {}
        by_type: dict[str, int] = {}
        total = 0
        for it in items:
            total += 1
            source_code = source_label(it.payload, args.corpus)
            doc_type = it.payload.get("doc_type") or "unknown"
            by_source[source_code] = by_source.get(source_code, 0) + 1
            by_type[doc_type] = by_type.get(doc_type, 0) + 1
    except Exception as exc:  # noqa: BLE001 — a clean fatal line, never a traceback
        print(f"error: fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        if conn is not None:
            conn.close()
    print(f"Matching documents: {total}")
    print("  by source_code:", dict(sorted(by_source.items(), key=lambda kv: -kv[1])))
    print("  by doc_type:", dict(sorted(by_type.items())))
    print(f"Unsupported (not offered to the engine): {sum(unsupported.values())}")
    print("  by source_code:", dict(sorted(unsupported.items(), key=lambda kv: -kv[1])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
