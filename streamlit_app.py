"""
streamlit_app.py  —  Patent Intelligence UI

Run with:
    cd /home/dennis/venv/projects/sarosh/patent-search
    streamlit run streamlit_app.py
"""

import sys
import json
import time
from pathlib import Path

import streamlit as st
import pandas as pd
import logging
import textwrap

# Configure logging to stdout so server logs appear in the terminal
logger = logging.getLogger("patent_app")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

# Make sure the project root is on the path regardless of cwd
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Patent Intelligence",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── session-state defaults ──────────────────────────────────────────────────────
_DEFAULTS = {
    "expanded_queries": [],
    "expansion_metadata": {},
    "expansion_evaluation": {},
    "raw_patents": [],
    "unique_patents": [],
    "ranked": [],
    "comparison": "",
    "report": "",
    "stage": 0,          # 0=idle 1=expanded 2=searched 3=ranked 4=analysed
    "error": "",
    "elapsed": {},
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def _reset():
    for k, v in _DEFAULTS.items():
        st.session_state[k] = v


# ── sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configuration")

    st.divider()
    st.subheader("🤖 Active Models")
    st.caption(f"**Query Expansion:** {settings.mistral_model}")
    st.caption(f"**Embeddings:** {settings.embedding_model}")

    st.divider()
    top_k = st.number_input("Top-K results", min_value=1, max_value=100, value=20)

    st.divider()
    st.subheader("Ollama / Mistral")
    use_mistral = st.toggle("Use Mistral for query expansion", value=True)

    st.divider()
    st.subheader("Pipeline stages")
    st.caption(
        "Run each stage in order, or jump straight to **Run to Top 20** "
        "which chains stages 1–3 automatically."
    )

    if st.button("🔄 Reset / New query", use_container_width=True):
        logger.info("Reset button clicked — resetting session state")
        _reset()
        st.rerun()

    st.divider()
    if st.session_state.elapsed:
        st.subheader("⏱ Stage timings")
        for stage, secs in st.session_state.elapsed.items():
            st.caption(f"{stage}: {secs:.1f}s")


# ── main ────────────────────────────────────────────────────────────────────────
st.title("🔬 Patent Intelligence")
st.caption("Test the pipeline stage by stage, or run straight through to Top 20.")

query = st.text_area(
    "Technology query",
    value="A system for autonomous vehicle lane detection using infrared sensors",
    height=80,
    key="query_input",
)

col1, col2, col3 = st.columns([2, 2, 1])
run_to_top20 = col1.button("▶ Run to Top 20", type="primary", use_container_width=True)
run_claude   = col2.button("🤖 Send Top 20 to Claude", use_container_width=True,
                            disabled=(st.session_state.stage < 3))
col3.button("🔄 Reset", on_click=_reset, use_container_width=True)

if st.session_state.error:
    st.error(st.session_state.error)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _run_expand(q: str) -> tuple[list[str], dict]:
    """Expand query and return (expanded_queries, structured_metadata)."""
    if use_mistral:
        from models.mistral_client import MistralClient
        from services.query_expansion import expand_query_with_metadata
        client = MistralClient()
        return expand_query_with_metadata(q, client)
    return [q], {}


def _run_search(queries: list[str]) -> tuple[list, list]:
    from services.search_service import fetch_all_patents
    from services.dedup_service import deduplicate
    raw = fetch_all_patents(queries)
    unique = deduplicate(raw)
    return raw, unique


def _run_rank(q: str, patents: list, k: int) -> list:
    from models.embedding_model import EmbeddingModel
    from services.embedding_service import embed_patents, embed_query
    from services.ranking_service import rank
    em = EmbeddingModel()
    doc_vecs = embed_patents(patents, em)
    query_vec = embed_query(q, em)
    return rank(query=q, patents=patents, doc_vecs=doc_vecs,
                query_vec=query_vec, top_k=k)


def _run_claude(q: str, ranked: list) -> tuple[str, str]:
    from models.claude_client import ClaudeClient
    from services.comparison_service import compare_patents
    from services.report_service import generate
    claude = ClaudeClient()
    comparison = compare_patents(q, ranked, claude)
    report     = generate(q, ranked, comparison, claude)
    return comparison, report


def _timed(label: str, fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    st.session_state.elapsed[label] = time.perf_counter() - t0
    return result


# ══════════════════════════════════════════════════════════════════════════════
# STAGE EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

if run_to_top20 and query.strip():
    logger.info("Run to Top 20 clicked — starting pipeline for query: %s", query)
    _reset()
    try:
        # Stage 1 — expand
        with st.spinner("Stage 1/3 — Expanding query with Mistral…"):
            expanded, metadata = _timed("1. Query expansion", _run_expand, query)
            logger.debug("Expansion result: %s", expanded)
            
            # SCORING TEMPORARILY DISABLED — evaluate top-10 relevance manually.
            # from services.query_evaluator import evaluate_expansion
            # evaluation = evaluate_expansion(
            #     original_query=query,
            #     expanded_terms=expanded[1:],
            #     structured_data=metadata
            # )
            evaluation = {}
            logger.debug("Expansion evaluation: disabled")
            
        st.session_state.expanded_queries = expanded
        st.session_state.expansion_metadata = metadata
        st.session_state.expansion_evaluation = evaluation
        st.session_state.stage = 1

        # Stage 2 — search + dedup
        with st.spinner(f"Stage 2/3 — Searching PatentsView for {len(expanded)} queries…"):
            raw, unique = _timed("2. Search + dedup", _run_search, expanded)
        st.session_state.raw_patents   = raw
        st.session_state.unique_patents = unique
        st.session_state.stage = 2

        if not unique:
            st.session_state.error = "No patents returned. Try a broader query."
            logger.warning("No patents returned for expanded queries: %s", expanded)
        else:
            # Stage 3 — embed + rank
            with st.spinner(f"Stage 3/3 — Embedding & ranking {len(unique)} patents…"):
                ranked = _timed("3. Embed + rank", _run_rank, query, unique, top_k)
                logger.debug("Ranking result count: %d", len(ranked))
            st.session_state.ranked = ranked
            st.session_state.stage = 3

    except Exception as exc:
        logger.exception("Unhandled exception during pipeline run")
        st.session_state.error = f"{type(exc).__name__}: {exc}"

if run_claude and st.session_state.stage >= 3:
    try:
        with st.spinner("Calling Claude for comparison & report…"):
            comparison, report = _timed(
                "4. Claude analysis", _run_claude,
                query, st.session_state.ranked
            )
            logger.debug("Claude comparison length=%d report length=%d", len(comparison), len(report))
        st.session_state.comparison = comparison
        st.session_state.report     = report
        st.session_state.stage      = 4
    except Exception as exc:
        logger.exception("Unhandled exception during Claude call")
        st.session_state.error = f"{type(exc).__name__}: {exc}"


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS DISPLAY
# ══════════════════════════════════════════════════════════════════════════════

tab_expand, tab_search, tab_rank, tab_claude = st.tabs([
    "1 · Query Expansion",
    "2 · Search Results",
    "3 · Top 20 Ranked",
    "4 · Claude Analysis",
])

# ── Tab 1: Query Expansion ──────────────────────────────────────────────────
with tab_expand:
    if st.session_state.expanded_queries:
        st.success(f"{len(st.session_state.expanded_queries)} queries generated"
                   + (" (incl. original)" if use_mistral else " (Mistral disabled)"))
        
        # Display expanded queries
        for i, q in enumerate(st.session_state.expanded_queries):
            label = "🔵 Original" if i == 0 else f"🟢 Expanded {i}"
            st.markdown(f"**{label}:** {q}")
        
        # Quality scoring temporarily disabled — focus on top-10 relevance.
        # To re-enable: restore the `if st.session_state.expansion_evaluation:` block.
        if False:  # DISABLED
            st.divider()
            eval_data = st.session_state.expansion_evaluation
            
            # Score header
            score = eval_data["score"]
            normalized = eval_data["normalized_score"]
            
            # Color coding based on score
            if score >= 70:
                score_color = "green"
                quality = "High"
            elif score >= 50:
                score_color = "orange"
                quality = "Medium"
            else:
                score_color = "red"
                quality = "Low"
            
            st.subheader("🎯 Query Expansion Quality")
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                st.metric("Total Score", f"{score}/100")
            with col2:
                st.metric("Normalized", f"{normalized}/10")
            with col3:
                st.caption(f":{score_color}[Overall Quality: {quality}]")
            
            # Category scores breakdown
            st.markdown("**Category Scores:**")
            category_scores = eval_data.get("category_scores", {})
            
            category_df = pd.DataFrame([
                {"Category": "Structural grouping", "Score": f"{category_scores.get('structural_grouping', 0)}/20"},
                {"Category": "Domain preservation", "Score": f"{category_scores.get('domain_preservation', 0)}/20"},
                {"Category": "Function preservation", "Score": f"{category_scores.get('function_preservation', 0)}/20"},
                {"Category": "Technology preservation", "Score": f"{category_scores.get('technology_preservation', 0)}/20"},
                {"Category": "Generic leakage control", "Score": f"{category_scores.get('generic_leakage_control', 0)}/20"},
            ])
            st.dataframe(category_df, use_container_width=True, hide_index=True)
            
            # Additional metrics
            st.markdown("**Additional Metrics:**")
            metrics = eval_data["metrics"]
            metrics_df = pd.DataFrame([
                {"Metric": "Domain preserved?", "Result": metrics["domain_preserved"]},
                {"Metric": "Function preserved?", "Result": metrics["function_preserved"]},
                {"Metric": "Technology preserved?", "Result": metrics["technology_preserved"]},
                {"Metric": "Specificity", "Result": metrics["specificity"]},
                {"Metric": "Dangerous generic terms?", "Result": "Yes" if metrics["dangerous_generic_terms"] else "No"},
                {"Metric": "Boolean fitness", "Result": metrics["boolean_fitness"]},
                {"Metric": "Token preservation", "Result": f"{int(metrics['token_preservation_ratio']*100)}%"},
            ])
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
            
            # Detailed breakdown
            with st.expander("📊 Detailed Breakdown", expanded=False):
                for line in eval_data["breakdown"]:
                    st.markdown(f"• {line}")
            
            # Structured data if available
            if st.session_state.expansion_metadata:
                with st.expander("📋 Structured Extraction", expanded=False):
                    meta = st.session_state.expansion_metadata
                    if meta.get("domain"):
                        st.markdown(f"**Domain:** {', '.join(meta['domain'])}")
                    if meta.get("function"):
                        st.markdown(f"**Function:** {', '.join(meta['function'])}")
                    if meta.get("technology"):
                        st.markdown(f"**Technology:** {', '.join(meta['technology'])}")
    else:
        st.info("Run the pipeline to see expanded queries here.")

# ── Tab 2: Search Results ───────────────────────────────────────────────────
with tab_search:
    if st.session_state.unique_patents:
        raw_count    = len(st.session_state.raw_patents)
        unique_count = len(st.session_state.unique_patents)
        dupes        = raw_count - unique_count

        m1, m2, m3 = st.columns(3)
        m1.metric("Raw fetched",  raw_count)
        m2.metric("After dedup",  unique_count)
        m3.metric("Duplicates removed", dupes)

        st.divider()
        rows = [
            {
                "patent_id":    p.patent_id,
                "title":        "\n".join(textwrap.wrap(p.patent_title or "", width=80)),
                "type":         p.patent_type or "—",
                "date":         p.patent_date  or "—",
            }
            for p in st.session_state.unique_patents
        ]
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True,
                     column_config={"title": st.column_config.TextColumn(width="large")})
        st.divider()
        st.subheader("Readable titles")
        for r in rows:
            st.write(f"• {r['patent_id']} — {r['title']}")
    else:
        st.info("Run the pipeline to see raw search results here.")

# ── Tab 3: Top 20 Ranked ─────────────────────────────────────────────────────
with tab_rank:
    if st.session_state.ranked:
        total_unique = len(st.session_state.unique_patents)
        shown        = len(st.session_state.ranked)
        filtered_out = total_unique - shown
        from app.config import settings as _s
        st.success(
            f"Top {shown} of {total_unique} unique patent(s) — "
            f"cosine ≥ {_s.cosine_threshold} threshold applied"
            + (f" · {filtered_out} discarded" if filtered_out > 0 else "")
        )

        rows = []
        for i, rp in enumerate(st.session_state.ranked, 1):
            rows.append({
                "#":            i,
                "patent_id":    rp.patent.patent_id,
                "title":        "\n".join(textwrap.wrap(rp.patent.patent_title or "", width=80)),
                "type":         rp.patent.patent_type  or "—",
                "date":         rp.patent.patent_date  or "—",
                "hybrid":       round(rp.hybrid_score,  4),
                "cosine":       round(rp.cosine_score,  4),
                "bm25":         round(rp.bm25_score,    4),
            })

        df = pd.DataFrame(rows)
        st.dataframe(
            df, use_container_width=True, hide_index=True,
            column_config={
                "title":  st.column_config.TextColumn(width="large"),
                "hybrid": st.column_config.ProgressColumn(
                    "hybrid", format="%.4f", min_value=0, max_value=1
                ),
            },
        )
        st.divider()
        st.subheader("Readable titles")
        for r in rows:
            st.write(f"{r['#']}. {r['patent_id']} — {r['title']}")

        st.divider()
        st.subheader("Score distribution")
        chart_df = pd.DataFrame({
            "hybrid": [r["hybrid"] for r in rows],
            "cosine": [r["cosine"] for r in rows],
            "bm25":   [r["bm25"]   for r in rows],
        })
        st.bar_chart(chart_df)

        st.divider()
        st.subheader("Patent detail")
        patent_labels = [f"{r['#']}. {r['patent_id']} — {r['title'][:60]}" for r in rows]
        selected_label = st.selectbox("Select a patent", patent_labels)
        selected_idx = int(selected_label.split(".")[0]) - 1
        rp = st.session_state.ranked[selected_idx]
        p  = rp.patent
        st.markdown(f"**ID:** `{p.patent_id}`")
        st.markdown(f"**Title:** {p.patent_title}")
        st.markdown(f"**Type:** {p.patent_type or '—'}   **Date:** {p.patent_date or '—'}")
        st.markdown(f"**Hybrid:** `{rp.hybrid_score:.4f}`  |  "
                    f"**Cosine:** `{rp.cosine_score:.4f}`  |  "
                    f"**BM25:** `{rp.bm25_score:.4f}`")
        if p.patent_abstract:
            with st.expander("Abstract"):
                st.write(p.patent_abstract)
    else:
        st.info("Run the pipeline to see ranked patents here.")

# ── Tab 4: Claude Analysis ───────────────────────────────────────────────────
with tab_claude:
    if st.session_state.stage < 3:
        st.info("Complete stages 1–3 first, then click **Send Top 20 to Claude**.")
    elif st.session_state.stage == 3:
        st.warning("Top 20 are ready. Click **Send Top 20 to Claude** to continue.")
    else:
        sub1, sub2 = st.tabs(["Comparison JSON", "Report"])

        with sub1:
            st.subheader("Structured claim comparison")
            try:
                parsed = json.loads(st.session_state.comparison)
                comp_df = pd.DataFrame(parsed)
                st.dataframe(comp_df, use_container_width=True, hide_index=True)
            except (json.JSONDecodeError, Exception):
                st.text_area("Raw comparison output", st.session_state.comparison,
                             height=400)

        with sub2:
            st.subheader("Patent intelligence report")
            st.markdown(st.session_state.report)

            st.divider()
            report_bytes = st.session_state.report.encode()
            st.download_button(
                "⬇ Download report (.md)",
                data=report_bytes,
                file_name="patent_report.md",
                mime="text/markdown",
            )
