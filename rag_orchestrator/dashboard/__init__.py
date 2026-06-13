"""Streamlit dashboard for the RAG data platform.

Two jobs, no separate database:

* **Manage ingestion** into the RAG (Qdrant) and show, as-of any publication
  date, what is actually loaded versus what is on disk — so you can confirm
  everything got ingested and see what is still missing.
* **Inventory & QC** of the scraped corpus: per-bank / per-type coverage, the
  real per-year cadence, anomalies (missing / exceptional years), upcoming &
  overdue documents, and any fetch errors — i.e. what remains to scrape.

(Graph / LLM exploration lives in the RAG itself, not here.)

The cb_corpus ``manifest.jsonl`` + the ``raw/`` tree are the corpus source of
truth; Qdrant is queried live for the RAG state. The two join on ``doc_id``.

Run it with::

    python -m rag_orchestrator.dashboard      # wraps `streamlit run`
"""
