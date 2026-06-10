"""Streamlit dashboard for the RAG data platform.

Tracks, as-of any date, the documents that have been scraped (the cb_corpus
``manifest.jsonl``) versus what is actually loaded in the RAG (Qdrant), plus a
log of LLM queries and graph snapshots.

No separate catalog database: the manifest *is* the corpus source of truth and
Qdrant is queried live. The ``state/*.jsonl`` files (query/graph logs) follow
the same append-only convention as the ingestion ledger.

Run it with::

    python -m rag_orchestrator.dashboard      # wraps `streamlit run`
"""
