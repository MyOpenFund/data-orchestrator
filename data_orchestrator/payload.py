"""The metadata a document is ingested with, in the vault's vocabulary.

One rule for every source that carries the central-bank corpus, so a document
gets the same payload keys and values whichever way it reaches the engine:

- :func:`row_to_payload`: a ``documents`` row of the vault (the vault source,
  ``sync-metadata``);
- :func:`manifest_row_to_payload`: a row of a producer's manifest, split the
  way the vault splits it on ingestion (:func:`manifest_row_to_vault_row`),
  then the same :func:`row_to_payload` (the ``cb_corpus`` disk source).

The payload is the vault's ``extra`` first, without

- the keys dropped by design (:func:`dropped_extra_keys`,
  ``DATA_ORCHESTRATOR_PAYLOAD_DROP_EXTRA_KEYS``): keys of no use to a reader of
  a citation, such as a path on the producer's machine. They are left out
  silently, not counted;
- the keys the engine writes or derives itself (``core.split_reserved_keys``):
  left out and counted (``SourceItem.dropped_keys`` -> the run report);

then the vault's columns (:data:`PAYLOAD_COLUMNS`), which always win; a NULL
column is left out and removes the same key of ``extra``; a date is written as
its ISO string.
"""
from __future__ import annotations

import os
import re
from typing import Mapping

from rag_engine.schema import DERIVED_KEYS, RESERVED_KEYS

from .core import split_reserved_keys

#: The vault columns copied into the payload, in payload order. ``pdf_url``,
#: ``source_url`` and ``mime_type`` come last: added after the others, they
#: leave the position of every earlier key unchanged.
PAYLOAD_COLUMNS: tuple[str, ...] = (
    "doc_id", "corpus", "source_code", "doc_type", "title", "date",
    "year", "language", "sha256", "provenance",
    "pdf_url", "source_url", "mime_type",
)

#: The fields of a manifest row the vault stores as columns; every other field
#: goes to ``extra``. A copy of the vault's ``ingestion/ingest.py``
#: ``KNOWN_FIELDS`` (kept here so this package does not import the vault):
#: ``bank_code`` is a legacy name of ``source_code``; ``local_path`` is a
#: column the payload does not carry.
VAULT_KNOWN_FIELDS: frozenset[str] = frozenset({
    "doc_id", "corpus", "source_code", "bank_code", "doc_type", "title",
    "pdf_url", "source_url", "date", "year", "language", "provenance",
    "mime_type", "sha256", "local_path",
})

#: Setting: the comma list of ``extra`` keys left out of the payload by design.
DROP_EXTRA_KEYS_ENV = "DATA_ORCHESTRATOR_PAYLOAD_DROP_EXTRA_KEYS"
#: Its default: ``html_path`` (a path on the producer's machine),
#: ``alt_urls`` / ``alt_source_urls`` (lists of other addresses of the
#: document; ``pdf_url`` and ``source_url`` are the ones a citation links to),
#: ``local_path`` (a path in the corpus, never a reader's concern).
DEFAULT_DROP_EXTRA_KEYS: tuple[str, ...] = ("html_path", "alt_urls", "alt_source_urls",
                                            "local_path")
#: The value of the setting that drops no key.
DROP_NONE = "none"


def dropped_extra_keys() -> tuple[str, ...]:
    """The keys of ``extra`` left out of the payload by design.

    ``DATA_ORCHESTRATOR_PAYLOAD_DROP_EXTRA_KEYS``, read from the environment
    (the CLI loads ``.env`` first): unset or blank -> the default
    (:data:`DEFAULT_DROP_EXTRA_KEYS`); ``none`` -> no key; else a comma list,
    entries trimmed, de-duplicated (order kept). Raises ``ValueError`` (the
    CLI: exit 2) on a list with no entry, or naming a vault column (columns
    always win: dropping one from ``extra`` changes nothing) or a key the
    engine writes or derives itself (already left out, and counted).
    """
    raw = os.environ.get(DROP_EXTRA_KEYS_ENV)
    if raw is None or not raw.strip():
        return DEFAULT_DROP_EXTRA_KEYS
    if raw.strip().lower() == DROP_NONE:
        return ()
    keys: list[str] = []
    for part in raw.split(","):
        key = part.strip()
        if key and key not in keys:
            keys.append(key)
    if not keys:
        raise ValueError(f"{DROP_EXTRA_KEYS_ENV}={raw!r} lists no key "
                         f"(leave it unset for the default, '{DROP_NONE}' to drop none)")
    refused = [k for k in keys if k in PAYLOAD_COLUMNS or k in RESERVED_KEYS or k in DERIVED_KEYS]
    if refused:
        raise ValueError(
            f"{DROP_EXTRA_KEYS_ENV}: {', '.join(refused)} cannot be dropped (a vault column "
            "always wins over extra; a key the engine writes itself is already left out)")
    return tuple(keys)


def row_to_payload(row: Mapping, drop_keys: tuple[str, ...] | None = None
                   ) -> tuple[dict, tuple[str, ...]]:
    """The metadata a vault row is ingested with, and the reserved keys of ``extra`` left out.

    ``drop_keys`` None -> :func:`dropped_extra_keys`. The keys dropped by
    design are not in the returned tuple: it names the keys the engine would
    refuse, the ones a run counts. ``doc_id`` always comes from its column and
    is never reported.
    """
    if drop_keys is None:
        drop_keys = dropped_extra_keys()
    extra = {k: v for k, v in (row.get("extra") or {}).items() if k not in drop_keys}
    payload, dropped = split_reserved_keys(extra)
    dropped = tuple(key for key in dropped if key != "doc_id")
    for col in PAYLOAD_COLUMNS:
        value = row.get(col)
        if value is None:
            payload.pop(col, None)
            continue
        payload[col] = value.isoformat() if hasattr(value, "isoformat") else value
    return payload, dropped


# A string the vault's DATE column reads as that day: a full date, optionally
# followed by a time (Postgres keeps the day).
_DATE_PREFIX = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}(?=$|[Tt ])")


def _as_date_column(value):
    """``date`` as the vault's DATE column returns it, ISO-written: an empty
    value is NULL, a date with a time keeps its day; anything else is kept as
    it is (the vault would refuse it)."""
    if value is None or value == "":
        return None
    if isinstance(value, str):
        found = _DATE_PREFIX.match(value)
        if found:
            return found.group(0)
    return value


def _as_year_column(value):
    """``year`` as the vault's INTEGER column returns it: a string of digits is
    the integer (Postgres casts it); anything else is kept as it is."""
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return value


def manifest_row_to_vault_row(rec: Mapping, default_corpus: str) -> dict:
    """The ``documents`` row the vault stores for manifest row ``rec``.

    The vault's split (``ingestion/ingest.py`` ``parse_line``): the fields of
    :data:`VAULT_KNOWN_FIELDS` are columns (``source_code`` else the legacy
    ``bank_code``; an empty ``date`` is NULL; an absent ``corpus`` is
    ``default_corpus``), every other field goes to ``extra``. The vault
    refuses a row whose ``corpus`` is not its own; here the row's own
    ``corpus`` is kept.
    """
    corpus = rec.get("corpus")
    return {
        "doc_id": rec.get("doc_id"),
        "corpus": default_corpus if corpus is None else corpus,
        "source_code": rec.get("source_code") or rec.get("bank_code"),
        "doc_type": rec.get("doc_type"),
        "title": rec.get("title"),
        "pdf_url": rec.get("pdf_url"),
        "source_url": rec.get("source_url"),
        "date": _as_date_column(rec.get("date")),
        "year": _as_year_column(rec.get("year")),
        "language": rec.get("language"),
        "provenance": rec.get("provenance"),
        "mime_type": rec.get("mime_type"),
        "sha256": rec.get("sha256"),
        "local_path": rec.get("local_path"),
        "extra": {k: v for k, v in rec.items() if k not in VAULT_KNOWN_FIELDS} or None,
    }


def manifest_row_to_payload(rec: Mapping, default_corpus: str,
                            drop_keys: tuple[str, ...] | None = None
                            ) -> tuple[dict, tuple[str, ...]]:
    """The metadata manifest row ``rec`` is ingested with: the payload the
    vault source gives the same document (:func:`manifest_row_to_vault_row`,
    then :func:`row_to_payload`)."""
    return row_to_payload(manifest_row_to_vault_row(rec, default_corpus), drop_keys)
