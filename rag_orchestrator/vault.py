"""Vault (Postgres) plumbing: connection + the rag_ingestions-backed ledger.

The vault owns the schema (its ingestion service runs the DDL train); this
module only reads/writes through the documented contract. Write protocol:
``VaultLedger.mark`` is called by ``run_ingest`` strictly AFTER the document's
Qdrant points landed — a crash in between leaves the vault under-claiming,
which the next resume pass heals (deterministic point ids overwrite in place).
"""
from __future__ import annotations

import os

import psycopg2

from .config import load_dotenv

RESUME_SQL = "SELECT doc_id FROM rag_ingestions WHERE collection = %s"

UPSERT_INGESTION_SQL = """
INSERT INTO rag_ingestions (
    doc_id, collection, corpus, source_code,
    embedding_model, embedding_version, chunk_count
) VALUES (%s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (doc_id, collection) DO UPDATE SET
    corpus = EXCLUDED.corpus,
    source_code = EXCLUDED.source_code,
    embedding_model = EXCLUDED.embedding_model,
    embedding_version = EXCLUDED.embedding_version,
    chunk_count = EXCLUDED.chunk_count,
    ingested_at = now();
"""


def connect():
    """Connect to the vault Postgres (DATABASE_URL from .env / environment)."""
    load_dotenv()
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set (.env or environment). Set it, or run "
            "with --no-vault to use the local file ledger without vault state."
        )
    return psycopg2.connect(url)


class VaultLedger:
    """``Ledger``-compatible resume/record state backed by ``rag_ingestions``."""

    def __init__(self, conn, collection: str, corpus: str,
                 embedding_model: str, embedding_version: str):
        self._conn = conn
        self.collection = collection
        self.corpus = corpus
        self.embedding_model = embedding_model
        self.embedding_version = embedding_version
        with conn.cursor() as cur:
            cur.execute(RESUME_SQL, (collection,))
            self._done = {row[0] for row in cur.fetchall()}

    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self._done

    def __len__(self) -> int:
        return len(self._done)

    def mark(self, doc_id: str, chunks: int, payload: dict | None = None) -> None:
        payload = payload or {}
        source_code = payload.get("source_code") or payload.get("bank_code")
        try:
            with self._conn.cursor() as cur:
                cur.execute(UPSERT_INGESTION_SQL, (
                    doc_id, self.collection, self.corpus, source_code,
                    self.embedding_model, self.embedding_version, chunks,
                ))
            self._conn.commit()
        except Exception:
            # A raising execute (e.g. an FK violation for a doc_id unknown to
            # `documents`) leaves this shared connection in aborted-transaction
            # state: every later statement on it would raise
            # InFailedSqlTransaction until the abort is cleared, silently
            # blocking convergence for every document processed after this
            # one. Roll back so the connection is healthy again, then
            # propagate — run_ingest's per-document isolation still counts
            # this one as a docs_error and moves on cleanly.
            self._conn.rollback()
            raise
        self._done.add(doc_id)
