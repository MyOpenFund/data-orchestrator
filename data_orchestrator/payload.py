"""The metadata a document is ingested with, in the vault's vocabulary.

One rule for every source of a corpus, so a document gets the same payload
keys and values whichever way it reaches the engine:

- :func:`row_to_payload`: a ``documents`` row of the vault (the vault source,
  ``sync-metadata``);
- :func:`manifest_row_to_payload`: a row of a producer's manifest, split the
  way the vault splits it on ingestion (:func:`manifest_row_to_vault_row`:
  ``vault_contract.split_row`` with ``transition=False``, the ``documents``
  contract's own split), then the same :func:`row_to_payload` (the
  ``cb_corpus`` disk source).

The payload is the vault's ``extra`` first, without

- the keys dropped by design (:func:`dropped_extra_keys`,
  ``DATA_ORCHESTRATOR_PAYLOAD_DROP_EXTRA_KEYS``): keys of no use to a reader of
  a citation, such as a path on the producer's machine. They are left out
  silently, not counted;
- the keys the engine writes or derives itself (``core.split_reserved_keys``):
  left out and counted (``SourceItem.dropped_keys`` -> the run report);

- the keys whose value is null (an absent or null field is left out);

then the vault's columns (:data:`PAYLOAD_COLUMNS`), which always win; a NULL
column is left out and removes the same key of ``extra``, and so does a blank
value of a column the contract reads blank as absent
(:data:`BLANK_IS_ABSENT_COLUMNS`: every column but the identity and the codes);
a date is written as its ISO string.

The contract's transition is over (vault 2859fa6, which ingests with
``transition=False``): the former column names ``pdf_url`` and ``bank_code``
are ordinary keys of ``extra``, and a NULL ``date_precision``, ``date_kind``,
``entity_scheme`` or ``entity_value`` removes the same key of ``extra`` like
any other column.
"""
from __future__ import annotations

import os
from typing import Mapping

from rag_engine.schema import DERIVED_KEYS, RESERVED_KEYS

from vault_contract import REFUSED_WHEN_ABSENT, split_row

from .core import split_reserved_keys

#: The vault columns copied into the payload, in payload order. A column
#: added later goes at the end, so the position of every earlier key stays the
#: same: ``document_url``, ``source_url`` and ``mime_type`` (``document_url``
#: was ``pdf_url`` until the 2026-11 rename), then the four columns of the
#: ``documents`` contract 1.0.0 (``date_precision``, ``date_kind``,
#: ``entity_scheme``, ``entity_value``).
PAYLOAD_COLUMNS: tuple[str, ...] = (
    "doc_id", "corpus", "source_code", "doc_type", "title", "date",
    "year", "language", "sha256", "provenance",
    "document_url", "source_url", "mime_type",
    "date_precision", "date_kind", "entity_scheme", "entity_value",
)

#: The columns where an empty or whitespace-only string is no value: read as
#: NULL, as the contract reads them (``split_row``: a blank value of a column
#: not refused when absent is absent). The vault stores them so from the
#: contract on; a vault row stored before it may still hold ``""`` (2,294
#: central-bank rows have ``source_url: ""``).
BLANK_IS_ABSENT_COLUMNS: frozenset[str] = (frozenset(PAYLOAD_COLUMNS)
                                           - frozenset(REFUSED_WHEN_ABSENT))

#: Setting: the comma list of ``extra`` keys left out of the payload by design.
DROP_EXTRA_KEYS_ENV = "DATA_ORCHESTRATOR_PAYLOAD_DROP_EXTRA_KEYS"
#: Its default: ``html_path`` (a path on the producer's machine),
#: ``alt_urls`` / ``alt_source_urls`` (lists of other addresses of the
#: document; ``document_url`` and ``source_url`` are the ones a citation links to),
#: ``local_path`` (a path in the corpus, never a reader's concern), and the
#: company corpus's paths of its other artefacts, ``pdf_path``, ``text_path``,
#: ``primary_path``, ``submission_path`` (extras of its rows; paths in the
#: corpus too).
DEFAULT_DROP_EXTRA_KEYS: tuple[str, ...] = ("html_path", "alt_urls", "alt_source_urls",
                                            "local_path", "pdf_path", "text_path",
                                            "primary_path", "submission_path")
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
    refused = [k for k in keys if k in PAYLOAD_COLUMNS
               or k in RESERVED_KEYS or k in DERIVED_KEYS]
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
    is never reported. A null key of ``extra`` is left out, not reported; a
    blank value of a column of :data:`BLANK_IS_ABSENT_COLUMNS` is a NULL one,
    and a NULL column removes the same key of ``extra``.
    """
    if drop_keys is None:
        drop_keys = dropped_extra_keys()
    extra = {k: v for k, v in (row.get("extra") or {}).items()
             if k not in drop_keys and v is not None}
    payload, dropped = split_reserved_keys(extra)
    dropped = tuple(key for key in dropped if key != "doc_id")
    for col in PAYLOAD_COLUMNS:
        value = row.get(col)
        if col in BLANK_IS_ABSENT_COLUMNS and isinstance(value, str) and not value.strip():
            value = None
        if value is None:
            payload.pop(col, None)
            continue
        payload[col] = value.isoformat() if hasattr(value, "isoformat") else value
    return payload, dropped


def manifest_row_to_vault_row(rec: Mapping, default_corpus: str) -> dict:
    """The ``documents`` row the vault stores for manifest row ``rec``.

    The vault's split, ``vault_contract.split_row`` with ``transition=False``,
    as the vault ingests: the contract's columns (blank optional values None;
    an undated row without ``date_precision`` and ``date_kind``), every other
    field in ``extra`` (None when empty). No old key is mapped: ``bank_code``
    and ``pdf_url`` are ordinary fields of ``extra``, and ``local_path`` is
    kept as given (a leading ``data/`` included). An absent or blank
    ``corpus`` is ``default_corpus`` (the disk source's ``--corpus``). The
    split never refuses: the vault refuses a row the contract refuses
    (``check_row``; a pre-contract row among them: ``missing_corpus``,
    ``missing_source_code``, ``bad_local_path``) and keeps no row for it; here
    the row is split all the same.
    """
    columns, extra = split_row(rec, transition=False)
    corpus = columns["corpus"]
    if corpus is None or (isinstance(corpus, str) and not corpus.strip()):
        columns["corpus"] = default_corpus
    return {**columns, "extra": extra or None}


def manifest_row_to_payload(rec: Mapping, default_corpus: str,
                            drop_keys: tuple[str, ...] | None = None
                            ) -> tuple[dict, tuple[str, ...]]:
    """The metadata manifest row ``rec`` is ingested with: the payload the
    vault source gives the same document (:func:`manifest_row_to_vault_row`,
    then :func:`row_to_payload`)."""
    return row_to_payload(manifest_row_to_vault_row(rec, default_corpus), drop_keys)
