"""Run-outcome policy: when a finished run is ``ok`` and when it is ``degraded``.

Doctrine (shared with central-bank-corpus ``runreport.py``): a run that did no
useful work must never exit 0. On top of that shared rule, the orchestrator
degrades a run whose failure ratio is above a configurable threshold (issue
#17, decided 2026-11-10): before it, a run failing 97 % of what it touched
still reported ``ok``.

- ``degraded`` / exit 3 iff
  (a) ``docs_ingested == 0 and docs_failed > 0``, or
  (b) ``docs_failed / docs_attempted > ratio``, with
      ``docs_attempted = docs_ingested + docs_empty + docs_failed``
      (resume skips are not attempts);
- otherwise ``ok`` / exit 0. A fatal exception is ``failed`` / exit 1
  (``cli._fatal_report``). Documents of unsupported types are never selected,
  so they never count here.
"""
from __future__ import annotations

import math
import os

DEGRADED_RATIO_ENV = "DATA_ORCHESTRATOR_DEGRADED_FAILURE_RATIO"
# Off-spec choice made explicit (Marc, 2026-11-10): above 10 % failures the
# run is degraded. 0 = any failure degrades; 1 = rule (b) off.
DEFAULT_DEGRADED_FAILURE_RATIO = 0.10


def degraded_failure_ratio() -> float:
    """The configured threshold. Unset or blank -> default.

    Raises ``ValueError`` (caller turns it into exit 2) for a value that is not
    a number in ``[0, 1]``.
    """
    raw = os.environ.get(DEGRADED_RATIO_ENV)
    if raw is None or not raw.strip():
        return DEFAULT_DEGRADED_FAILURE_RATIO
    try:
        value = float(raw.strip())
    except ValueError:
        raise ValueError(
            f"{DEGRADED_RATIO_ENV}={raw!r} is not a number (expected 0..1)"
        ) from None
    if math.isnan(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{DEGRADED_RATIO_ENV}={raw!r} is outside 0..1")
    return value


def run_outcome(*, docs_ingested: int, docs_empty: int, docs_failed: int,
                ratio: float) -> tuple[str, int]:
    """(outcome, exit_code) of a run that finished without a fatal error."""
    if docs_ingested == 0 and docs_failed > 0:
        return "degraded", 3
    attempted = docs_ingested + docs_empty + docs_failed
    if attempted and docs_failed / attempted > ratio:
        return "degraded", 3
    return "ok", 0
