"""Streamlit entry point for the RAG data dashboard.

Launch with ``python -m rag_orchestrator.dashboard`` (preferred) or directly::

    streamlit run rag_orchestrator/dashboard/app.py
"""
from __future__ import annotations

import base64
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

# Allow ``streamlit run app.py`` (no package context) as well as ``-m``.
try:
    from rag_orchestrator.dashboard import catalog
except ImportError:  # pragma: no cover - direct-file launch
    import sys

    # parents[2] == the RAGDataOrchestrator repo root (which contains the
    # ``rag_orchestrator`` package). parents: [dashboard, rag_orchestrator, repo].
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from rag_orchestrator.dashboard import catalog


st.set_page_config(page_title="RAG Data Platform", page_icon="📚", layout="wide")

DEFAULT_COLLECTION = "cb_corpus"


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Scanning corpus (disk + manifest)…")
def _corpus(manifest_path: str | None) -> list[dict]:
    # Disk-truth: every PDF under raw/, enriched from the manifest by doc_id.
    return catalog.load_corpus(Path(manifest_path) if manifest_path else None)


@st.cache_data(show_spinner="Querying Qdrant…", ttl=30)
def _rag_state(collection: str, host: str, port: int):
    return catalog.get_rag_state(collection, host=host, port=port)


def _to_df(docs: list[dict]) -> pd.DataFrame:
    if not docs:
        return pd.DataFrame()
    df = pd.DataFrame(docs)
    keep = [
        "pub_date", "source", "bank_code", "doc_type", "group_label",
        "title", "in_rag", "n_chunks", "provenance", "pdf_url", "abs_path",
    ]
    df = df[[c for c in keep if c in df.columns]]
    return df


# ---------------------------------------------------------------------------
# Sidebar — shared controls
# ---------------------------------------------------------------------------
st.sidebar.title("📚 RAG Data Platform")
manifest_path = catalog.find_manifest()
st.sidebar.caption(
    f"Corpus: `{manifest_path.parent}/raw` + manifest" if manifest_path
    else "⚠️ cb_corpus not found"
)

collection = st.sidebar.text_input("Qdrant collection", DEFAULT_COLLECTION)
qhost = st.sidebar.text_input("Qdrant host", "localhost")
qport = st.sidebar.number_input("Qdrant port", value=6333, step=1)

corpus_rows = _corpus(str(manifest_path) if manifest_path else None)
rag = _rag_state(collection, qhost, int(qport))

if rag.reachable:
    st.sidebar.success(f"Qdrant up · {rag.total_points:,} chunks in '{collection}'")
else:
    st.sidebar.warning(f"Qdrant offline — corpus-only view\n\n{rag.error}")

# As-of date control (publication date)
pub_dates = [r["pub_date"] for r in corpus_rows if r.get("pub_date")]
if pub_dates:
    dmin, dmax = min(pub_dates), max(pub_dates)
else:
    dmin = dmax = date.today()
as_of = st.sidebar.date_input(
    "As-of date (publication)", value=dmax, min_value=dmin, max_value=dmax,
    help="Point-in-time: show the corpus as it stood on this date (no look-ahead).",
)
if isinstance(as_of, tuple):  # date_input can return a tuple
    as_of = as_of[0]

docs = catalog.build_documents(corpus_rows, rag, as_of=as_of)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_docs, tab_graphs, tab_queries = st.tabs(["📄 Documents", "🕸️ Graphs", "💬 LLM Queries"])

# === Documents =============================================================
with tab_docs:
    st.header(f"Documents — as of {as_of}")
    if not corpus_rows:
        st.error("No corpus found. Set CB_CORPUS_ROOT in .env or place cb_corpus alongside this repo.")
    else:
        n_scraped = len(docs)
        n_in_rag = sum(1 for d in docs if d["in_rag"])
        coverage = (n_in_rag / n_scraped * 100) if n_scraped else 0
        banks = {d.get("bank_code") for d in docs}

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("On disk (≤ as-of)", f"{n_scraped:,}")
        c2.metric("Loaded in RAG", f"{n_in_rag:,}")
        c3.metric("Coverage", f"{coverage:.1f}%")
        c4.metric("Banks", len(banks))

        df = _to_df(docs)

        # --- filters
        fc1, fc2, fc3 = st.columns(3)
        sources = sorted(df["source"].dropna().unique()) if "source" in df else []
        groups = sorted(df["group_label"].dropna().unique()) if "group_label" in df else []
        bank_opts = sorted(b for b in banks if b)
        f_sources = fc1.multiselect("Source", sources, default=sources)
        f_groups = fc2.multiselect("Category", groups, default=groups)
        f_banks = fc3.multiselect("Bank", bank_opts, default=[])

        view = df.copy()
        if f_sources:
            view = view[view["source"].isin(f_sources)]
        if f_groups:
            view = view[view["group_label"].isin(f_groups)]
        if f_banks:
            view = view[view["bank_code"].isin(f_banks)]

        only_rag = st.checkbox("Only documents loaded in the RAG", value=False)
        if only_rag:
            view = view[view["in_rag"]]

        # --- charts
        if not view.empty:
            ch1, ch2 = st.columns(2)
            with ch1:
                st.subheader("By year")
                years = view.assign(
                    year=pd.to_datetime(view["pub_date"], errors="coerce").dt.year
                )
                by_year = years.groupby("year").size()
                st.bar_chart(by_year)
            with ch2:
                st.subheader("By category")
                st.bar_chart(view.groupby("group_label").size())

            st.subheader("Cumulative corpus over time (publication date)")
            ts = (
                view.dropna(subset=["pub_date"])
                .assign(pub_date=pd.to_datetime(view["pub_date"], errors="coerce"))
                .sort_values("pub_date")
            )
            if not ts.empty:
                cum = ts.groupby("pub_date").size().cumsum()
                st.area_chart(cum)

        # --- table
        st.subheader(f"{len(view):,} documents")
        st.dataframe(
            view.drop(columns=["abs_path"], errors="ignore"),
            use_container_width=True, hide_index=True,
            column_config={
                "pdf_url": st.column_config.LinkColumn("Source URL", display_text="open"),
                "in_rag": st.column_config.CheckboxColumn("In RAG"),
            },
        )

        # --- document viewer
        st.subheader("Open a document")
        if not view.empty:
            view = view.reset_index(drop=True)
            labels = [
                f"{r.pub_date} · {r.bank_code}/{r.doc_type} · {str(r.title)[:70]}"
                for r in view.itertuples()
            ]
            idx = st.selectbox("Pick a document", range(len(labels)),
                               format_func=lambda i: labels[i])
            chosen = view.iloc[idx]
            meta_col, view_col = st.columns([1, 2])
            with meta_col:
                st.write({
                    "date": str(chosen.get("pub_date")),
                    "bank": chosen.get("bank_code"),
                    "type": f"{chosen.get('doc_type')} — {chosen.get('group_label')}",
                    "in_rag": bool(chosen.get("in_rag")),
                    "chunks": int(chosen.get("n_chunks", 0)),
                    "provenance": chosen.get("provenance"),
                })
                if chosen.get("pdf_url"):
                    st.markdown(f"[Official source]({chosen['pdf_url']})")
            with view_col:
                p = chosen.get("abs_path")
                p = Path(p) if p else None
                if p and p.is_file() and p.suffix.lower() == ".pdf":
                    data = p.read_bytes()
                    st.download_button("Download PDF", data, file_name=p.name)
                    b64 = base64.b64encode(data).decode()
                    st.markdown(
                        f'<iframe src="data:application/pdf;base64,{b64}" '
                        f'width="100%" height="600"></iframe>',
                        unsafe_allow_html=True,
                    )
                elif p and p.is_file():
                    st.info(f"Non-PDF artifact on disk: {p.name}")
                else:
                    st.warning(f"File not found on disk: {p}")

# === Graphs ================================================================
with tab_graphs:
    st.header(f"Graph snapshot — as of {as_of}")
    st.caption(
        "Time-series-of-graphs: a cognitive/similarity graph built from the "
        "documents available on the as-of date. v1 reuses the mvp-graph-rag "
        "spectral layer (singular / hinge / theta nodes)."
    )
    in_rag_docs = [d for d in docs if d["in_rag"]]
    st.metric("Documents available for the graph (in RAG, ≤ as-of)", f"{len(in_rag_docs):,}")
    if not in_rag_docs:
        st.info(
            "No ingested documents yet for this as-of date. Run the orchestrator "
            "(`rag-orchestrator cb_corpus …`) to populate the RAG, then build the "
            "graph here."
        )
    else:
        st.button("Build graph snapshot (coming next)", disabled=True)
        st.caption("Wiring to mvp-graph-rag build_graph/spectral is the next step.")
    # log of past snapshots
    snaps = catalog.load_jsonl("graphs")
    if snaps:
        st.subheader("Past graph snapshots")
        st.dataframe(pd.DataFrame(snaps), use_container_width=True, hide_index=True)

# === LLM Queries ===========================================================
with tab_queries:
    st.header("LLM queries")
    st.caption("Each query is logged with the as-of date it was run against.")
    queries = catalog.load_jsonl("queries")
    if queries:
        st.dataframe(pd.DataFrame(queries), use_container_width=True, hide_index=True)
    else:
        st.info("No queries logged yet.")

    with st.expander("Log a query manually"):
        q = st.text_input("Question")
        a = st.text_area("Answer")
        if st.button("Save to log") and q:
            catalog.log_jsonl("queries", {
                "question": q, "answer": a, "collection": collection,
                "as_of": str(as_of),
            })
            st.success("Logged. Refresh to see it above.")
