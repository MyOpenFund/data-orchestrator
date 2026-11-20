"""data_orchestrator CLI.

Installed as the ``data-orchestrator`` console script. Standalone repo that
drives the rag-engine library (a dependency, see ``README.md``)::

    data-orchestrator vault --corpus central-bank
    data-orchestrator vault --corpus company --doctypes A1 --year-min 2024
    data-orchestrator cb_corpus --banks ecb --doctypes C1 --limit 20
    data-orchestrator cb_corpus --year-min 2015 --collection central-bank-e5b-v1

    data-orchestrator sync-metadata --collection central-bank-e5b-v1           # dry run
    data-orchestrator sync-metadata --collection central-bank-e5b-v1 --write

(equivalently ``python -m data_orchestrator.cli <source> ...``)

The default collection is routed per corpus by
``routing.collection_name`` — ``{corpus}-{model_tag}-v1``, e.g.
``central-bank-e5b-v1`` for the ``cb_corpus`` source or ``vault --corpus
central-bank``, ``company-e5b-v1`` for ``vault --corpus company`` — for every
source, never a hard-coded legacy name. The ``cb_corpus`` disk source implies
the ``central-bank`` corpus, which is also the default of the other commands;
an explicit ``--corpus`` always wins. Company documents come from the vault
(``vault --corpus company``): the company corpus has no disk source.

The vault (``rag_ingestions``) is the *default* resume mechanism: a re-run
skips documents already recorded there. Pass ``--no-vault`` to fall back to a
local JSON-lines ledger under ``<repo>/state/`` instead (the disk source
``cb_corpus`` only; the vault source's resume *is*
the vault anti-join in ``sources/vault.py``, so ``--no-vault`` is rejected
there). ``--no-resume`` ignores the ledger and re-ingests everything; it is
likewise rejected for the vault source (its anti-join always filters on the
target collection, so a fresh ``--collection`` name is the way to re-ingest,
not ``--no-resume``).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import config, policy
from .core import (
    Ledger, SourceItem, run_ingest, IngestStats, engine_settings, interrupted_stats, source_label,
)
from .payload import dropped_extra_keys
from .routing import collection_name, supported_mime_types
from .sources import cb_corpus as cb_corpus_source


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


# Selectable sources (and the two vault maintenance commands, probe and
# sync-metadata, which take the same first positional argument).
SOURCES = ("cb_corpus", "vault", "probe", "sync-metadata")
SYNC_METADATA = "sync-metadata"
# The disk source of the company corpus, removed with the documents contract:
# named in a clear refusal rather than argparse's list of choices.
REMOVED_COMPANY_SOURCE = "company"


class _Sources(list):
    """The choices of the source argument: :data:`SOURCES`, plus the removed
    company source, accepted (``in``) but never listed (iteration), so that
    neither the help nor argparse's invalid-choice message offers it."""

    def __contains__(self, value) -> bool:
        return value == REMOVED_COMPANY_SOURCE or super().__contains__(value)

# For ``--count-only``: the two payload fields each disk source is summarised
# by (the vault source has its own count path — source_code/doc_type — in
# _count_vault_source; cb_corpus payloads use the vault's keys).
COUNT_KEYS = {
    "cb_corpus": ("source_code", "doc_type"),
}

# Each disk source implies its own vault corpus, so --corpus can default
# per-source instead of one hard-coded value; an explicit --corpus always
# wins. Sources absent here (vault, probe) keep the "central-bank" default.
# The company corpus has no disk source: its documents come from the vault
# (``vault --corpus company``).
IMPLIED_CORPUS = {
    "cb_corpus": "central-bank",
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
        corpus=args.corpus,
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


# First line of the summary: a run that reached its end, and a run a fatal
# error stopped (the counts that follow are then what was done before it).
SUMMARY_DONE = "Done."
SUMMARY_INTERRUPTED = "Interrupted by a fatal error. Before it:"


def _print_summary(stats: IngestStats, elapsed: float | None = None, *,
                   interrupted: bool = False) -> None:
    print("─" * 60)
    print(SUMMARY_INTERRUPTED if interrupted else SUMMARY_DONE)
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
    if stats.payload_keys_dropped:
        dropped = ", ".join(f"{key} x{n}" for key, n in sorted(stats.payload_keys_dropped.items()))
        print(
            f"warning: {sum(stats.payload_keys_dropped.values())} metadata key(s) reserved "
            f"for the engine were left out of the payloads ({dropped}): the engine "
            "writes these keys itself. Remove them from the source metadata "
            "(the vault's documents.extra).",
            file=sys.stderr,
        )
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
# (_fatal_report), with the counts of what was done before it.
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
    # Reserved payload keys the sources left out of free-form metadata
    # (core.split_reserved_keys). Written only when there are some, so the
    # report of a clean run keeps the shape it always had.
    if stats.payload_keys_dropped:
        totals["payload_keys_dropped"] = sum(stats.payload_keys_dropped.values())
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
                  unsupported: dict[str, int] | None = None,
                  stats: IngestStats | None = None) -> dict:
    """Best-effort report for a run that never reached a normal outcome.

    The reason goes under ``fatal`` (runs.extra->>'fatal'), the key
    central-bank-corpus already writes; it was ``error`` before 2026-11.
    ``stats`` are the statistics of a run interrupted in the middle
    (``core.interrupted_stats``): the totals and ``sources[]`` then say what
    was done before the failure, as ``rag_ingestions`` does. Without them
    (a failure before the run started) the totals are zeros and ``sources[]``
    is empty. ``unsupported`` (the vault source's per-source count, when it
    was computed before the failure) is passed through so a known
    ``docs_unsupported`` and ``unsupported_by_source`` are not lost.
    """
    # The ratio is irrelevant to a fatal outcome (overwritten below); the
    # default is stated explicitly since _build_report has no default.
    rep = _build_report(command, stats if stats is not None else IngestStats(), started_at,
                        corpus=corpus,
                        ratio=policy.DEFAULT_DEGRADED_FAILURE_RATIO,
                        unsupported=unsupported)
    rep["outcome"] = "failed"
    rep["exit_code"] = 1
    rep["fatal"] = f"{type(exc).__name__}: {exc}"[:FATAL_MAX_CHARS]
    return rep


def _print_interrupted_summary(stats: IngestStats | None, elapsed: float | None = None) -> None:
    """The counts of a run a fatal error stopped after its first document,
    under ``Interrupted…`` instead of ``Done.`` (the fatal line follows,
    printed by the caller). Nothing for a run that saw no document.

    Called once the run report is written, and never raises: the fatal error
    may be stdout itself (a pipe whose reader has gone), and failing to print
    the summary must not replace the error that ended the run nor change the
    exit code."""
    if stats is None or not stats.docs_seen:
        return
    try:
        _print_summary(stats, elapsed, interrupted=True)
    except Exception:  # noqa: BLE001 — the caller's fatal line says why the run ended
        pass


def _lock_guard(vault_mod, conn, collection: str, holder: str):
    """``run_ingest``'s guard: before each document, the run's vault session
    must be alive and still hold the shared collection lock."""
    def guard():
        vault_mod.check_collection_lock(conn, collection, exclusive=False, holder=holder)
    return guard


def _insert_fatal_report(vault_mod, conn, report: dict) -> None:
    """Write a fatal report on a connection the failure may have left inside
    an aborted transaction: roll back first (a failing rollback is not
    itself fatal; the insert then says whether the connection is usable),
    and warn on an insert failure — never silent, never masking the run's
    own fatal outcome."""
    fresh = None
    if getattr(conn, "closed", 0):
        # The session itself is gone (VaultSessionLost): a new one for the report.
        try:
            conn = fresh = vault_mod.connect()
        except Exception as exc:  # noqa: BLE001 — report write never masks the run's own outcome
            print(f"warning: failed to write run report to vault: {exc}", file=sys.stderr)
            return
    try:
        conn.rollback()
    except Exception:  # noqa: BLE001 — the insert below reports a dead connection
        pass
    try:
        vault_mod.insert_run_report(conn, report)
    except Exception as exc:  # noqa: BLE001 — report write never masks the run's own outcome
        print(f"warning: failed to write run report to vault: {exc}", file=sys.stderr)
    finally:
        if fresh is not None:
            fresh.close()


def _append_report_jsonl(report: dict) -> None:
    """--no-vault fallback: a single atomic append to state_dir()/runs.jsonl."""
    folder = state_dir()
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "runs.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(report) + "\n")


# The engine's own defaults (rag_engine.config.Settings qdrant_host/qdrant_port),
# used only to display a store that reports no host/port. 6333 is the LEGACY
# store.
QDRANT_DEFAULT_HOST = "localhost"
QDRANT_DEFAULT_PORT = 6333

QDRANT_API_KEY_ENV = "QDRANT_API_KEY"


def _api_key_state(environ=None) -> str:
    """``"set"`` when ``QDRANT_API_KEY`` is non-blank, else ``"unset"``; the
    value itself is never returned. The engine resolves its key from this same
    variable (``Settings.from_env``, same blank-is-unset rule) and this repo
    never passes an explicit key, so the state matches the store's."""
    environ = os.environ if environ is None else environ
    return "set" if (environ.get(QDRANT_API_KEY_ENV) or "").strip() else "unset"


def _print_qdrant_target(host, port) -> None:
    """``run_ingest``'s ``on_target``: print the Qdrant the constructed store
    writes to — what the store resolved, not a second reading of the
    environment — and
    whether an API key is set (never its value), and warn on the legacy port.
    A store that reports no host or port is displayed with the engine's default
    (an invalid or empty ``QDRANT_PORT`` in the environment never gets here:
    ``_qdrant_port_error`` refuses it before any work); the port is compared
    numerically (``06333`` is 6333)."""
    host = str(host).strip() if host is not None else ""
    host = host or QDRANT_DEFAULT_HOST
    raw = str(port).strip() if port is not None else ""
    try:
        port = int(raw) if raw else QDRANT_DEFAULT_PORT
    except ValueError:
        port = raw  # displayed as is; cannot be the legacy port
    print(f"Qdrant: {host}:{port} (api key: {_api_key_state()})", flush=True)
    if port == QDRANT_DEFAULT_PORT:
        print(f"warning: Qdrant port {port} is the legacy store (the engine's default); "
              "set QDRANT_PORT to the dedicated one unless this is intended",
              file=sys.stderr)


QDRANT_PORT_ENV = "QDRANT_PORT"


def _qdrant_port_error(environ=None) -> str | None:
    """Pre-flight for an ingest run: the error message when ``QDRANT_PORT`` is
    set but is not a plain integer in 1..65535 (surrounding whitespace
    ignored; empty is invalid, ``.env`` values are literal so an inline
    ``# comment`` is part of the value), else ``None``. Absent is fine
    (the engine's default, warned about by ``_print_qdrant_target``)."""
    environ = os.environ if environ is None else environ
    raw = environ.get(QDRANT_PORT_ENV)
    if raw is None:
        return None
    text = raw.strip()
    if text.isascii() and text.isdigit() and 1 <= int(text) <= 65535:
        return None
    return (f"{QDRANT_PORT_ENV}={raw!r} is not a port number (expected a plain "
            "integer 1..65535; unset it to use the engine's default)")


LOG_LEVEL_ENV = "DATA_ORCHESTRATOR_LOG_LEVEL"
DEFAULT_LOG_LEVEL = "INFO"
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
# Libraries that log every HTTP request or file lock at INFO.
_NOISY_LOGGERS = ("transformers", "sentence_transformers", "huggingface_hub", "urllib3",
                  "httpx", "httpcore", "filelock")


def _setup_logging(environ=None) -> None:
    """Send the log records of this package and of the engine to stderr.

    The engine, like any library, configures no logging: without this its
    INFO lines (model loaded, collection created) would be lost and its
    warnings printed bare. Level: ``DATA_ORCHESTRATOR_LOG_LEVEL`` (default
    ``INFO``); raises ``ValueError`` on a name that is not a logging level.

    Two things are done:

    - the root logger gets a stderr handler at that level, unless it already
      has handlers (a host process configured logging): they are then left as
      they are, the root level too: the level variable does not reach them;
    - in every case, the libraries that log each HTTP request or file lock at
      INFO (``_NOISY_LOGGERS``) are set to that level or WARNING, whichever is
      higher.
    """
    environ = os.environ if environ is None else environ
    raw = (environ.get(LOG_LEVEL_ENV) or "").strip()
    level = raw.upper() or DEFAULT_LOG_LEVEL
    if level not in LOG_LEVELS:
        raise ValueError(f"{LOG_LEVEL_ENV}={raw!r} is not a log level "
                         f"(one of {', '.join(LOG_LEVELS)})")
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    for noisy in _NOISY_LOGGERS:
        logging.getLogger(noisy).setLevel(max(logging.WARNING, getattr(logging, level)))


TESSDATA_PREFIX_ENV = "TESSDATA_PREFIX"


def _forced_ocr_error(ocr: str, environ=None) -> str | None:
    """Pre-flight for an ingest run: the error message when ``--ocr always``
    is asked on a host without ``TESSDATA_PREFIX`` (unset or blank), else
    ``None``. Forced OCR does not look at ``ENABLE_OCR`` but still needs the
    language data: asking for it without the variable is a configuration
    error, refused whatever the selection holds (the parser would refuse each
    PDF of it; a selection without PDFs would not notice)."""
    environ = os.environ if environ is None else environ
    if ocr != "always" or (environ.get(TESSDATA_PREFIX_ENV) or "").strip():
        return None
    return (f"forced OCR needs {TESSDATA_PREFIX_ENV}; set it or use --ocr auto/never "
            f"(--ocr always was asked and {TESSDATA_PREFIX_ENV}, the folder of "
            "Tesseract's *.traineddata files or its parent, is not set)")


def _print_ocr_state(settings, ocr: str) -> None:
    """Say, before a run, whether scanned pages will be read by OCR.

    ``--ocr auto`` depends on the host: the engine runs OCR only when
    ``ENABLE_OCR`` is ``true`` and ``TESSDATA_PREFIX`` is set, both read from
    the process environment each time a file is chunked — so they must be in
    this repo's ``.env`` (or the real environment). A silent "auto -> never"
    is what this line is for. The languages are ``RAG_ENGINE_OCR_LANGUAGES``.
    ``--ocr always`` gets here only with ``TESSDATA_PREFIX`` set
    (``_forced_ocr_error`` refuses the run otherwise).
    """
    from rag_engine.chunking import ocr_available

    languages = f"languages: {settings.ocr_languages}"
    if ocr == "never":
        print("OCR: off (--ocr never)", flush=True)
    elif ocr == "always":
        print(f"OCR: forced (--ocr always, {languages})", flush=True)
    elif ocr_available():
        print(f"OCR: available ({languages})", flush=True)
    else:
        print("OCR: off (ENABLE_OCR not true or TESSDATA_PREFIX unset)", flush=True)


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
    # The removed company source is accepted by the parser but never listed
    # (help, invalid-choice message), so that it is refused with its
    # replacement wherever it stands on the line.
    parser.add_argument("source", choices=_Sources(SOURCES), help="data source to ingest")
    parser.add_argument("--root", help="corpus root (folder containing raw/, or the source's data dir)")
    parser.add_argument("--banks", help="comma list of bank codes, e.g. ecb,fr")
    parser.add_argument("--doctypes", help="comma list of doc-type codes of the corpus, e.g. "
                                           "C1,A3 (central-bank) or A1 (company)")
    parser.add_argument("--groups", help="comma list of doc groups, e.g. A,C")
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
                        help="vault corpus this run belongs to (default: central-bank; "
                             "company documents: vault --corpus company)")
    parser.add_argument("--count-only", action="store_true",
                        help="just count matching documents, do not ingest")
    parser.add_argument("--progress-every", type=int, default=25,
                        help="print progress every N documents (default: 25)")
    sync = parser.add_argument_group(
        "sync-metadata", "correct the date metadata of ingested documents in Qdrant "
        "from the vault (dry run unless --write; README, 'Correcting dates in Qdrant')")
    sync.add_argument("--write", action="store_true",
                      help="apply the corrections (set_payload; creates the missing payload "
                           "indexes first). Without it nothing is written to Qdrant")
    sync.add_argument("--doc-ids", help="comma list of doc_ids: only these ledger documents")
    sync.add_argument("--csv", help="path of the CSV of differences (default: "
                                    "<state dir>/sync-metadata-<collection>-<UTC time>.csv)")
    sync.add_argument("--page-size", type=int, default=None,
                      help="points per Qdrant scroll request (default: 256)")
    sync.add_argument("--doc-batch", type=int, default=None,
                      help="documents per Qdrant read, one doc_id filter (default: 100)")
    sync.add_argument("--include-title", action="store_true",
                      help="also compare and write title (off by default)")
    sync.add_argument("--include-links", action="store_true",
                      help="also compare and write document_url, source_url, mime_type, and "
                           "remove pdf_url (the former name of document_url) and the extra "
                           "keys dropped by design (DATA_ORCHESTRATOR_PAYLOAD_DROP_EXTRA_KEYS) "
                           "from the points (off by default)")
    sync.add_argument("--create-indexes", action="store_true",
                      help="only create the missing payload indexes of rag-engine "
                           "(date_start, date_end) and exit; needs a key allowed to write")
    # Known arguments first: a line naming the removed source with its old
    # flags (--ciks, --prefer) gets the replacement, not "unrecognized arguments".
    args, _unknown = parser.parse_known_args(argv)
    if args.source == REMOVED_COMPANY_SOURCE:
        print("error: the 'company' disk source was removed: company documents come from the "
              "vault now. Run: data-orchestrator vault --corpus company (COMPANY_CORPUS_ROOT "
              "names the company corpus data root)", file=sys.stderr)
        return 2
    args = parser.parse_args(argv)  # anything unknown is an error now (exit 2)
    # The one .env loader (issue #8), run before anything reads the
    # environment: the engine loads no .env of its own, it reads the process
    # environment when asked (its settings below; ENABLE_OCR / TESSDATA_PREFIX
    # each time a file is chunked). No $VAR interpolation
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
        _setup_logging()
        args.degraded_ratio = policy.degraded_failure_ratio()
        from . import vault as _vault

        _vault.keepalive_options()  # DATA_ORCHESTRATOR_PG_KEEPALIVES_*: exit 2 here, not mid-run
        args.supported_mimes = supported_mime_types()
        # DATA_ORCHESTRATOR_PAYLOAD_DROP_EXTRA_KEYS: every payload is built
        # with it, so a bad value is refused here, not once per document.
        dropped_extra_keys()
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    sync_flags = [flag for flag, given in (
        ("--write", args.write), ("--doc-ids", args.doc_ids is not None),
        ("--csv", args.csv is not None), ("--page-size", args.page_size is not None),
        ("--doc-batch", args.doc_batch is not None), ("--include-title", args.include_title),
        ("--include-links", args.include_links), ("--create-indexes", args.create_indexes),
    ) if given]
    if args.source != SYNC_METADATA and sync_flags:
        print(f"error: {', '.join(sync_flags)} only apply to sync-metadata", file=sys.stderr)
        return 2

    if args.source == SYNC_METADATA:
        return _sync_metadata(args)

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
        # The engine's settings (model and dimension, batch size, OCR
        # languages, Qdrant timeout, ...), read once from the loaded
        # environment: an invalid value is one operator error here, not a
        # failure per document later.
        try:
            args.engine_settings = engine_settings()
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        ocr_error = _forced_ocr_error(args.ocr)
        if ocr_error is not None:
            print(f"error: {ocr_error}", file=sys.stderr)
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
    t0 = None  # set when the engine run starts
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        if not args.no_vault:
            from . import vault as vault_mod
            from .routing import EMBEDDING_VERSION, embedding_model_name

            # Vault mode, with or without the resume ledger: the run writes
            # the points of documents a sync-metadata run may be correcting
            # (--no-resume rewrites the payloads of ingested documents), so it
            # never runs alongside one (shared: ingestions do not exclude
            # each other). The connection is also the one of the run report.
            vault_conn = vault_mod.connect()
            vault_mod.lock_collection(vault_conn, collection, exclusive=False,
                                      holder=f"{args.source} ingestion")
            # Before any Qdrant write: a vault before the documents contract
            # has no key (corpus, doc_id, collection) for the ledger.
            vault_mod.require_contract_schema(vault_conn)
            if not args.no_resume:
                ledger = vault_mod.VaultLedger(
                    vault_conn, collection=collection, corpus=args.corpus,
                    embedding_model=embedding_model_name(),
                    embedding_version=EMBEDDING_VERSION,
                )
                print(f"Resume ledger (vault): rag_ingestions/{collection} "
                      f"({len(ledger)} docs already done)")
        elif not args.no_resume:
            ledger_path = Path(args.ledger) if args.ledger else state_dir() / f"{collection}.jsonl"
            ledger = Ledger(ledger_path)
            print(f"Resume ledger (file): {ledger_path} ({len(ledger)} docs already done)")

        print(f"→ Ingesting source '{args.source}' into collection '{collection}' "
              f"(ocr={args.ocr}, limit={args.limit})")
        _print_ocr_state(args.engine_settings, args.ocr)
        t0 = time.time()
        stats = run_ingest(
            items,
            collection=collection,
            ledger=ledger,
            ocr=args.ocr,
            limit=args.limit,
            on_progress=_make_progress(args.progress_every),
            on_target=_print_qdrant_target,
            settings=args.engine_settings,
            guard=(None if vault_conn is None else
                   _lock_guard(vault_mod, vault_conn, collection, f"{args.source} ingestion")),
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
            # Defensive: vault mode always opens vault_conn (and takes the
            # collection lock) before the run, with or without --no-resume.
            # Should it be missing, open a short-lived one just for the
            # report insert so the run isn't silently unreported. Same
            # warn-only contract: a report failure must never mask the run's
            # own outcome.
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
        # The report first, then the summary: printing can fail too.
        stats = interrupted_stats(exc)
        elapsed = None if t0 is None else time.time() - t0
        rep = _fatal_report(args.source, started_at, exc, corpus=args.corpus, stats=stats)
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
        _print_interrupted_summary(stats, elapsed)
        print(f"error: fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        return rep["exit_code"]
    finally:
        if vault_conn is not None:
            vault_conn.close()


def _sync_metadata(args) -> int:
    """``sync-metadata``: compare the date metadata of the ledger's documents
    in Qdrant with the vault and, with ``--write``, correct it
    (``data_orchestrator.sync_metadata``). No run report is written (like
    ``probe``): the CSV is the record of a run. Exit 0 when the run reached
    its end, 1 on a fatal error (the vault, Qdrant, a refused key, the
    collection lock held by an ingestion), 2 on a usage or configuration error.
    """
    from . import sync_metadata as sm
    from . import vault as vault_mod
    from .routing import ROUTING

    unsupported = [flag for flag, given in (
        ("--count-only", args.count_only), ("--no-vault", args.no_vault),
        ("--no-resume", args.no_resume), ("--ledger", args.ledger is not None),
        ("--source-codes", args.source_codes is not None),
        ("--doctypes", args.doctypes is not None), ("--languages", args.languages is not None),
        ("--year-min", args.year_min is not None), ("--year-max", args.year_max is not None),
        ("--banks", args.banks is not None), ("--groups", args.groups is not None),
        ("--root", args.root is not None),
    ) if given]
    if unsupported:
        print(f"error: {', '.join(unsupported)} not supported by sync-metadata (it reads "
              "the rag_ingestions ledger of the collection; narrow it with --doc-ids or "
              "--limit)", file=sys.stderr)
        return 2
    for flag, value in (("--page-size", args.page_size), ("--doc-batch", args.doc_batch)):
        if value is not None and value < 1:
            print(f"error: {flag} must be >= 1", file=sys.stderr)
            return 2
    if args.create_indexes and (args.write or args.doc_ids or args.limit or args.csv
                                or args.include_title or args.include_links):
        print("error: --create-indexes only creates the payload indexes; run it alone",
              file=sys.stderr)
        return 2
    doc_ids = _csv(args.doc_ids)
    if args.doc_ids is not None and not doc_ids:
        print("error: --doc-ids is empty", file=sys.stderr)
        return 2
    port_error = _qdrant_port_error()
    if port_error is not None:
        print(f"error: {port_error}", file=sys.stderr)
        return 2
    try:
        settings = engine_settings()
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.corpus is None:
        args.corpus = "central-bank"
    if args.corpus not in ROUTING:
        print(f"error: unknown corpus '{args.corpus}' (known: {', '.join(sorted(ROUTING))})",
              file=sys.stderr)
        return 1
    collection = args.collection or collection_name(args.corpus)
    from .core import build_store

    store = build_store(settings)
    _print_qdrant_target(getattr(store, "host", None), getattr(store, "port", None))

    if args.create_indexes:
        try:
            created = store.ensure_payload_indexes(collection)
        except Exception as exc:  # noqa: BLE001 — a clean fatal line, never a traceback
            print(f"error: fatal: {type(exc).__name__}: {exc}{_sync_advice(exc, collection)}",
                  file=sys.stderr)
            return 1
        print(f"Payload indexes created on '{collection}': "
              f"{', '.join(created) if created else 'none (all exist)'}")
        return 0

    csv_path = Path(args.csv).expanduser() if args.csv else sm.default_csv_path(state_dir(), collection)
    keys = sm.owned_keys(args.include_title, args.include_links)
    mode = "WRITE: corrections are applied" if args.write else "dry run: nothing is written to Qdrant"
    print(f"→ sync-metadata: collection '{collection}', corpus '{args.corpus}' ({mode})")
    print(f"  keys compared: {', '.join(keys)}")
    every = args.progress_every
    t0 = time.time()
    conn = None
    try:
        conn = vault_mod.connect()
        # --write: exclusive (no ingestion, no other sync-metadata). A dry run
        # only reads: shared, so an ingestion may start or go on beside it
        # (its report may then be stale for the documents that ingestion
        # writes), but it never runs beside a --write.
        vault_mod.lock_collection(conn, collection, exclusive=args.write, holder="sync-metadata")
        # An older vault keeps date_precision / date_kind in extra: comparing
        # against it would read them as removed.
        vault_mod.require_contract_schema(conn)
        schema = sm.payload_schema(store, collection)
        from rag_engine.schema import PAYLOAD_INDEXES

        missing = [f for f, _kind in PAYLOAD_INDEXES if f not in schema]
        if missing:
            print(f"  payload indexes missing: {', '.join(missing)} ("
                  + ("created before any correction" if args.write
                     else "--write creates them first, or --create-indexes") + ")")

        def progress(stats):
            if stats.documents % every == 0:
                print(f"  documents={stats.documents} "
                      + " ".join(f"{a}={n}" for a, n in stats.actions.items())
                      + f" {'written' if args.write else 'to_write'}="
                      + f"{stats.written if args.write else stats.to_write}", flush=True)

        def guard(conn=conn):
            vault_mod.check_collection_lock(conn, collection, exclusive=args.write,
                                            holder="sync-metadata")

        stats = sm.run_sync(
            conn, store, corpus=args.corpus, collection=collection, csv_path=csv_path,
            write=args.write, include_title=args.include_title,
            include_links=args.include_links, doc_ids=doc_ids,
            limit=args.limit,
            page_size=args.page_size or sm.DEFAULT_PAGE_SIZE,
            doc_batch=args.doc_batch or sm.DEFAULT_DOC_BATCH,
            on_progress=progress, guard=guard, schema=schema,
        )
    except Exception as exc:  # noqa: BLE001 — a clean fatal line, never a traceback
        print(f"error: fatal: {type(exc).__name__}: {exc}{_sync_advice(exc, collection)}",
              file=sys.stderr)
        if csv_path.exists():
            print(f"  CSV of the documents seen before the error: {csv_path}", file=sys.stderr)
        return 1
    finally:
        if conn is not None:
            conn.close()
    if stats.indexes_created:
        print(f"  payload indexes created: {', '.join(stats.indexes_created)}")
    if doc_ids:
        missing_ids = sorted(set(doc_ids) - stats.seen_ids)
        if missing_ids:
            print(f"warning: {len(missing_ids)} of the --doc-ids are not in rag_ingestions for "
                  f"'{collection}' and corpus '{args.corpus}': {', '.join(missing_ids[:10])}",
                  file=sys.stderr)
    sm.print_summary(stats, write=args.write, elapsed=time.time() - t0, csv_path=csv_path)
    return 0


def _sync_advice(exc: BaseException, collection: str) -> str:
    """What to do about the fatal error of a sync-metadata run."""
    from rag_engine.store import MissingIndexError, QdrantAuthError

    if isinstance(exc, MissingIndexError):
        return (f". Create the payload indexes first: data-orchestrator sync-metadata "
                f"--create-indexes --collection {collection} (with a key allowed to write)")
    if isinstance(exc, QdrantAuthError):
        return (". A dry run needs a key allowed to read; --write and --create-indexes "
                "need a key allowed to write (a read-only key is refused)")
    return ""


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
    bad_paths: list[str] = []  # rows skipped for their local_path (routing.BadLocalPath)
    try:
        conn = vault_mod.connect()
        # Never alongside a sync-metadata run of this collection (shared:
        # ingestions do not exclude each other).
        vault_mod.lock_collection(conn, collection, exclusive=False, holder="vault ingestion")
        vault_mod.require_contract_schema(conn)  # before any Qdrant write
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
        bad_paths: list[str] = []
        items = vault_source.iter_items(conn, args.corpus, collection, refused=bad_paths,
                                        **filters)
        print(f"→ Ingesting vault selection (corpus={args.corpus}) into "
              f"'{collection}' (ocr={args.ocr}, limit={args.limit})")
        _print_ocr_state(args.engine_settings, args.ocr)
        stats = run_ingest(
            items, collection=collection, ledger=ledger, ocr=args.ocr,
            limit=args.limit, on_progress=_make_progress(args.progress_every),
            on_target=_print_qdrant_target, settings=args.engine_settings,
            guard=_lock_guard(vault_mod, conn, collection, "vault ingestion"),
        )
        _print_summary(stats)

        rep = _build_report("vault", stats, started_at, corpus=args.corpus,
                            ratio=args.degraded_ratio, unsupported=unsupported)
        _report_bad_paths(rep, bad_paths)
        try:
            vault_mod.insert_run_report(conn, rep)
        except Exception as exc:  # noqa: BLE001 — report write never masks the run's own outcome
            print(f"warning: failed to write run report to vault: {exc}", file=sys.stderr)
        return rep["exit_code"]
    except Exception as exc:  # noqa: BLE001 — fatal: best-effort report, honest exit code
        # The report first, then the summary: printing can fail too.
        stats = interrupted_stats(exc)
        rep = _fatal_report("vault", started_at, exc, corpus=args.corpus,
                            unsupported=unsupported, stats=stats)
        _report_bad_paths(rep, bad_paths)
        if conn is not None:
            _insert_fatal_report(vault_mod, conn, rep)
        _print_interrupted_summary(stats)
        print(f"error: fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        return rep["exit_code"]
    finally:
        if conn is not None:
            conn.close()


#: How many doc_ids of rows skipped for their local_path the run report names.
BAD_PATH_EXAMPLES = 20


def _report_bad_paths(rep: dict, bad_paths: list) -> None:
    """Count the rows skipped for a local_path the documents contract refuses
    (``routing.BadLocalPath``) in the run report (keys outside the ``runs``
    columns: ``runs.extra``), and say so on stderr. Nothing when there is none."""
    if not bad_paths:
        return
    rep["docs_bad_local_path"] = len(bad_paths)
    rep["bad_local_path_doc_ids"] = bad_paths[:BAD_PATH_EXAMPLES]
    print(f"warning: {len(bad_paths)} document(s) skipped, never read: their local_path is "
          f"refused by the documents contract ({', '.join(bad_paths[:5])}"
          f"{', …' if len(bad_paths) > 5 else ''})", file=sys.stderr)


def _count_vault_source(args, collection: str) -> int:
    """Count vault-selected documents without any engine work or ledger writes.

    Mirrors the cb_corpus ``--count-only`` output format (total + breakdowns),
    by ``source_code``/``doc_type`` — the vault source's payload columns,
    which cb_corpus payloads use too. Both source breakdowns
    use ``core.source_label`` (a NULL source_code reads as the corpus), the
    label the run report's ``sources[]`` uses; a NULL doc_type reads
    ``unknown``. No Qdrant line: a count never touches Qdrant.
    """
    from . import vault as vault_mod
    from .sources import vault as vault_source

    conn = None
    try:
        conn = vault_mod.connect()
        vault_mod.require_contract_schema(conn)
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
