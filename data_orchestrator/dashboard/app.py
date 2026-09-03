"""Streamlit entry point for the RAG data dashboard.

Launch with ``python -m data_orchestrator.dashboard`` (preferred) or directly::

    streamlit run data_orchestrator/dashboard/app.py
"""
from __future__ import annotations

import base64
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

# Allow ``streamlit run app.py`` (no package context) as well as ``-m``.
try:
    from data_orchestrator.dashboard import catalog
except ImportError:  # pragma: no cover - direct-file launch
    import sys

    # parents[2] == the data-orchestrator repo root (which contains the
    # ``data_orchestrator`` package). parents: [dashboard, data_orchestrator, repo].
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from data_orchestrator.dashboard import catalog


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
        "pub_date", "source", "bank_code", "doc_type_label", "group_label",
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
tab_docs, tab_qc = st.tabs(["📄 Documents", "🔎 Inventory & QC"])

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
                "doc_type_label": st.column_config.TextColumn("type"),
                "group_label": st.column_config.TextColumn("category"),
                "pdf_url": st.column_config.LinkColumn("Source URL", display_text="open"),
                "in_rag": st.column_config.CheckboxColumn("In RAG"),
            },
        )

        # --- document viewer
        st.subheader("Open a document")
        if not view.empty:
            view = view.reset_index(drop=True)
            labels = [
                f"{r.pub_date} · {r.bank_code} · {r.doc_type_label} · {str(r.title)[:60]}"
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
                    "type": f"{chosen.get('doc_type_label')} ({chosen.get('group_label')})",
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

# === Inventory & QC ========================================================
with tab_qc:
    st.header("Inventory & quality control")
    if not corpus_rows:
        st.info("No corpus found.")
    else:
        current_year = date.today().year
        docs_all = catalog.build_documents(corpus_rows, rag, as_of=None)  # full, not as-of
        all_banks = sorted({r["bank_code"] for r in corpus_rows if r.get("bank_code")})
        default_bank = "ecb" if "ecb" in all_banks else all_banks[0]
        sel_bank = st.selectbox("Bank", all_banks, index=all_banks.index(default_bank))

        # --- Per-type summary: what we have & the typical cadence ---
        st.subheader(f"What we have — {sel_bank}, by document type")
        ts = catalog.type_summary(corpus_rows, sel_bank)
        if ts:
            tdf = pd.DataFrame(ts)[["label", "total", "years",
                                    "avg_per_year", "median_per_year",
                                    "calendar_per_year", "last_year"]]
            st.dataframe(
                tdf, use_container_width=True, hide_index=True,
                column_config={
                    "label": st.column_config.TextColumn("document type"),
                    "avg_per_year": st.column_config.NumberColumn("avg/yr"),
                    "median_per_year": st.column_config.NumberColumn("median/yr"),
                    "calendar_per_year": st.column_config.NumberColumn(
                        "expected/yr",
                        help="Expected per year — the bank's published meeting/release "
                             "count (reference only; the corpus often stores several "
                             "documents per meeting, so avg/median can be higher)"),
                })
            st.caption("**avg/yr · median/yr** = the corpus's real cadence (what you "
                       "actually have). **expected/yr** = the published meeting count, "
                       "shown as context — it is NOT used to flag anomalies.")

        # --- Coverage matrix (per bank) ---
        st.subheader(f"Coverage matrix — {sel_bank}: documents per year × type")
        cov = catalog.coverage_matrix(corpus_rows, sel_bank)
        if cov["types"]:
            years_sorted = sorted(cov["grid"])
            # rows = document type (full label), columns = year — readable with long labels
            mat = pd.DataFrame(
                {catalog.DOCTYPE_LABELS.get(dt, dt):
                    {y: cov["grid"].get(y, {}).get(dt, 0) for y in years_sorted}
                 for dt in cov["types"]}
            ).T
            mat.index.name = "document type"
            st.dataframe(mat, use_container_width=True)
            exp = {f"{t}": v for (b, t), v in catalog.EXPECTED_PER_YEAR.items() if b == sel_bank}
            if exp:
                st.caption("Expected/yr (published calendar): "
                           + ", ".join(f"{k}={v}" for k, v in sorted(exp.items())))
        else:
            st.info(f"No documents for {sel_bank}.")

        # --- Browse / view the documents behind a cell ---
        with st.expander(f"📂 Browse & view {sel_bank} documents"):
            bdocs = [d for d in docs_all if d.get("bank_code") == sel_bank]
            types_here = sorted({d["doc_type"] for d in bdocs})
            if types_here:
                cdt, cyr = st.columns(2)
                pick_t = cdt.selectbox("Type", types_here, key="qc_type",
                                       format_func=lambda c: catalog.DOCTYPE_LABELS.get(c, c))
                yrs = sorted({d["year"] for d in bdocs
                              if d["doc_type"] == pick_t and d.get("year")}, reverse=True)
                pick_y = cyr.selectbox("Year", yrs, key="qc_year") if yrs else None
                sub = [d for d in bdocs if d["doc_type"] == pick_t and d.get("year") == pick_y]
                st.caption(f"{len(sub)} document(s) · "
                           f"{sum(1 for d in sub if d['in_rag'])} in RAG")
                if sub:
                    labels = [f"{d.get('pub_date')} · "
                              f"{(str(d.get('title')) or d['doc_id'])[:70]}"
                              f"{'  ✓RAG' if d['in_rag'] else ''}" for d in sub]
                    di = st.selectbox("Document", range(len(labels)),
                                      format_func=lambda i: labels[i], key="qc_doc")
                    chosen = sub[di]
                    p = chosen.get("abs_path")
                    p = Path(p) if p else None
                    if p and p.is_file() and p.suffix.lower() == ".pdf":
                        data = p.read_bytes()
                        st.download_button("Download PDF", data, file_name=p.name, key="qc_dl")
                        b64 = base64.b64encode(data).decode()
                        st.markdown(
                            f'<iframe src="data:application/pdf;base64,{b64}" '
                            f'width="100%" height="500"></iframe>', unsafe_allow_html=True)
                    elif p:
                        st.warning(f"File not on disk: {p}")

        # --- Anomalies ---
        st.subheader("Anomalies — deviations from the expected count")
        anom = catalog.anomalies(corpus_rows, current_year=current_year)
        if anom:
            adf = pd.DataFrame(anom)
            adf["type"] = adf["doc_type"].map(lambda c: catalog.DOCTYPE_LABELS.get(c, c))
            cc = st.columns(4)
            cc[0].metric("Flagged", len(anom))
            cc[1].metric("Missing", int((adf["flag"] == "missing").sum()))
            cc[2].metric("Low", int((adf["flag"] == "low").sum()))
            cc[3].metric("Exceptional", int((adf["flag"] == "exceptional").sum()))
            if st.checkbox(f"Only {sel_bank}", value=False, key="anom_bank"):
                adf = adf[adf["bank_code"] == sel_bank]
            st.dataframe(adf[["bank_code", "type", "year", "count", "expected",
                              "basis", "flag"]],
                         use_container_width=True, hide_index=True)
            st.caption("`expected` = the type's own **recent median docs/yr** (data-driven, "
                       "within its active span — not the meeting calendar). "
                       "`missing` = 0 in an active year · `low` < 50% · "
                       "`exceptional` > 150% of that baseline (e.g. an emergency-meeting year).")
        else:
            st.success("No anomalies flagged against the expected baselines.")

        # --- Upcoming / overdue ---
        st.subheader("Upcoming & overdue (from publication cadence)")
        up = catalog.upcoming(corpus_rows, today=date.today())
        if up:
            udf = pd.DataFrame(up)
            udf["type"] = udf["doc_type"].map(lambda c: catalog.DOCTYPE_LABELS.get(c, c))
            c1u, c2u = st.columns(2)
            c1u.metric("Overdue (>7d past)", int((udf["status"] == "overdue").sum()))
            c2u.metric("Due soon (≤60d)", int((udf["status"] == "soon").sum()))
            st.dataframe(udf[["bank_code", "type", "last", "interval_days",
                              "next_expected", "days_until", "status"]],
                         use_container_width=True, hide_index=True)
            st.caption("Next release ≈ last date + median recent interval. "
                       "`overdue` often signals a fetch gap rather than a late release.")
        else:
            st.info("Not enough dated history to estimate cadence.")

        # --- Fetching problems ---
        st.subheader("Fetching problems")
        errs = catalog.load_discovery_errors()
        if errs:
            st.error(f"{len(errs)} discovery error(s) recorded — see below.")
            st.dataframe(pd.DataFrame(errs), use_container_width=True, hide_index=True)
        else:
            st.success("No fetch errors recorded "
                       "(cb_corpus writes data/discovery_errors.jsonl on failures).")
