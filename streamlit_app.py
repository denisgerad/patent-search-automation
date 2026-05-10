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
    "claude_raw_response": "",
    "expansion_evaluation": {},
    "raw_patents": [],
    "unique_patents": [],
    "ranked": [],
    "comparison": "",
    "report": "",
    "stage": 0,          # 0=idle 1=expanded 2=searched 3=ranked 4=analysed
    "error": "",
    "elapsed": {},
    "json_query": "",         # optional structured JSON query (raw text)
    "json_boolean": "",        # generated Boolean string
    "json_uspto_url": "",      # USPTO Full Text link
    "json_result_count": None, # PatentsView total_patent_count
    "json_groups": [],         # list[list[str]] — for badge matching
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
    st.subheader("🔍 Search Backend")
    _backend_name = settings.search_backend.lower()
    if _backend_name == "lens" and settings.lens_api_token:
        st.success("Lens.org API ✅")
    elif _backend_name == "lens" and not settings.lens_api_token:
        st.warning("Lens.org — token missing\nAdd `lens_api_token=` to .env")
    else:
        st.info("Offline fixture data\n_(add lens_api_token to .env for live search)_")

    st.divider()
    st.subheader("🤖 Active Models")
    st.caption(f"**Query Expansion:** claude-sonnet-4 (API)")
    st.caption(f"**Embeddings:** {settings.embedding_model}")

    st.divider()
    top_k = st.number_input("Top-K results", min_value=1, max_value=100, value=20)

    st.divider()
    st.subheader("Claude API")
    use_mistral = st.toggle("Use Claude for query expansion", value=True)

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

# ── JSON format input ──────────────────────────────────────────────────────────
_JSON_PLACEHOLDER = (
    '{\n'
    '  "groups": [\n'
    '    {\n'
    '      "terms": [\n'
    '        "autonomous vehicle",\n'
    '        "self-driving vehicle",\n'
    '        "driverless vehicle",\n'
    '        "autonomous driving system"\n'
    '      ]\n'
    '    },\n'
    '    {\n'
    '      "terms": [\n'
    '        "lane detection",\n'
    '        "lane recognition",\n'
    '        "lane boundary detection",\n'
    '        "road lane identification"\n'
    '      ]\n'
    '    },\n'
    '    {\n'
    '      "terms": [\n'
    '        "infrared sensor",\n'
    '        "IR detector",\n'
    '        "infrared camera",\n'
    '        "infrared camera based lane detection"\n'
    '      ]\n'
    '    }\n'
    '  ],\n'
    '  "combine_groups_with": "AND"\n'
    '}'
)

with st.expander("📋 JSON Format  *(optional — overrides Mistral expansion)*", expanded=bool(st.session_state.json_query)):
    st.caption(
        "Paste a structured JSON query to bypass Mistral and use your own term groups. "
        "The **Technology query** above is still used for embedding-based ranking."
    )
    st.info(
        "💡 **Tips for better precision:**\n"
        "- Aim for **3–4 terms per group** — more synonyms = better recall without losing precision\n"
        "- Add at least one **specific technical phrase** per group "
        "(e.g. `\"infrared camera based lane detection\"` instead of just `\"infrared sensor\"`)\n"
        "- Add a **constraint group** for key technology constraints "
        "(e.g. `{\"terms\": [\"real-time processing\", \"embedded system\"]}` combined with AND)\n"
        "- Multi-word terms are automatically quoted for phrase matching"
    )
    _json_raw = st.text_area(
        "json_query_area",
        value=st.session_state.json_query,
        height=190,
        placeholder=_JSON_PLACEHOLDER,
        label_visibility="collapsed",
        key="json_query_input",
    )
    st.session_state.json_query = _json_raw

    # Live validation
    if _json_raw.strip():
        try:
            from services.json_query_service import (
                validate_json_query as _vjq_live,
                build_boolean_string as _bbs_live,
            )
            _schema_live = _vjq_live(_json_raw)
            _combine_live = _schema_live.combine_with
            _n_groups_live = len(_schema_live.groups)
            _n_terms_live  = len(_schema_live.all_terms)
            st.success(
                f"✅ Valid — {_n_groups_live} groups, {_n_terms_live} terms "
                f"(combined with **{_combine_live}**). Mistral expansion will be skipped."
            )
            st.caption(f"**Boolean Query:** `{_bbs_live(_schema_live)}`")
            # Nudge toward richer groups
            _thin = [
                i + 1 for i, g in enumerate(_schema_live.groups)
                if len(g.terms) < 3
            ]
            if _thin:
                _glist = ", ".join(f"group {n}" for n in _thin)
                st.warning(
                    f"⚠️ {_glist} {'has' if len(_thin) == 1 else 'have'} fewer than 3 terms. "
                    "Consider adding more synonyms or a specific technical phrase to improve precision."
                )
        except Exception as _json_err:
            st.error(f"❌ Invalid JSON — {_json_err}")
    else:
        st.caption("_No JSON pasted — Mistral expansion will run normally._")

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

def _queries_from_json(raw: str) -> list[str]:
    """Return a flat list of unique search terms from the structured JSON query."""
    from services.json_query_service import validate_json_query as _vjq
    schema = _vjq(raw)
    return schema.all_terms


def _run_expand(q: str) -> tuple[list[str], dict]:
    """Expand query and return (expanded_queries, structured_metadata)."""
    if use_mistral:
        from models.claude_client import ClaudeClient
        from services.query_expansion import expand_query_with_metadata
        client = ClaudeClient()
        expanded, metadata = expand_query_with_metadata(q, client)
        st.session_state.claude_raw_response = metadata.get("claude_raw_response", "")
        return expanded, metadata
    return [q], {}


def _run_search(queries: list[str]) -> tuple[list, list]:
    from services.dedup_service import deduplicate
    backend = settings.search_backend.lower()

    if backend == "lens" and settings.lens_api_token:
        from services.lens_search_service import fetch_all_patents
    elif backend == "patentsview":
        from services.search_service import fetch_all_patents
    else:
        # fallback to fixture if no API token is configured
        from services.fixture_search_service import fetch_all_patents

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


def _run_search_json(schema) -> tuple[list, list]:
    """
    Search by sending each group as a separate OR query to PatentsView,
    then merge + dedup. High recall at search time; ranking handles precision.
    (Mirrors the pipeline's existing query_builder.py philosophy.)
    """
    from services.json_query_service import build_patentsview_query, JsonQuerySchema, TermGroup
    from services.search_service import fetch_patents_for_query_dict
    from services.dedup_service import deduplicate

    all_raw = []
    for group in schema.groups:
        # Send each group as a standalone OR query
        single_group_schema = JsonQuerySchema(groups=[group], combine_with="OR")
        query_dict = build_patentsview_query(single_group_schema)
        results = fetch_patents_for_query_dict(query_dict)
        all_raw.extend(results)

    unique = deduplicate(all_raw)
    return all_raw, unique


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


# ── Group-match badge helpers ───────────────────────────────────────────────
_GROUP_COLOURS = ["#3B82F6", "#10B981", "#F59E0B", "#8B5CF6", "#EC4899", "#EF4444"]

def _group_badge_html(group_idx: int, term: str) -> str:
    colour = _GROUP_COLOURS[group_idx % len(_GROUP_COLOURS)]
    return (
        f'<span style="background:{colour};color:#fff;'
        f'padding:1px 8px;border-radius:10px;font-size:0.78em;'
        f'margin-left:6px;white-space:nowrap">'
        f'G{group_idx + 1} {term}</span>'
    )

def _badges_for(title: str, abstract: str, groups: list) -> str:
    """Return HTML: escaped title + coloured group-match badge spans."""
    from services.json_query_service import match_groups_in_text
    matches = match_groups_in_text(f"{title} {abstract or ''}", groups)
    badges  = "".join(_group_badge_html(idx, term) for idx, term in matches)
    safe    = title.replace("<", "&lt;").replace(">", "&gt;")
    return f"{safe}{badges}"


# ══════════════════════════════════════════════════════════════════════════════
# STAGE EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

if run_to_top20 and query.strip():
    logger.info("Run to Top 20 clicked — starting pipeline for query: %s", query)
    # Preserve JSON query text across the reset
    _saved_json = st.session_state.json_query
    _reset()
    st.session_state.json_query = _saved_json

    # Determine whether to use the pasted JSON or Mistral expansion
    _active_json = st.session_state.json_query.strip()
    _use_json_expansion = False
    _json_terms: list[str] = []
    if _active_json:
        try:
            _json_terms = _queries_from_json(_active_json)
            _use_json_expansion = bool(_json_terms)
        except Exception as _je:
            logger.warning("JSON query parse failed at run time: %s", _je)

    try:
        # Stage 1 — expand
        if _use_json_expansion:
            with st.spinner(f"Stage 1/3 — Building Boolean from JSON ({len(_json_terms)} terms)…"):
                from services.json_query_service import (
                    validate_json_query as _vjq,
                    build_boolean_string as _bbs,
                    build_patentsview_query as _bpq,
                    build_uspto_url as _buu,
                )
                _schema    = _vjq(_active_json)
                _bool_str  = _bbs(_schema)
                _uspto_url = _buu(_bool_str)
                expanded   = _schema.all_terms
                metadata   = {}
                evaluation = {}
                logger.info("JSON Boolean query: %s", _bool_str)
                logger.info("JSON expansion terms (%d): %s", len(expanded), expanded)

            # Fetch result count from whichever backend is active
            with st.spinner("Fetching result count…"):
                _backend = settings.search_backend.lower()
                if _backend == "lens" and settings.lens_api_token:
                    from services.lens_search_service import fetch_result_count as _lens_count
                    _count = _lens_count(
                        [g.terms for g in _schema.groups],
                        _schema.combine_with,
                    )
                else:
                    _count = None  # fixture / patentsview offline

            st.session_state.json_boolean      = _bool_str
            st.session_state.json_uspto_url    = _uspto_url
            st.session_state.json_result_count = _count
            st.session_state.json_groups       = [g.terms for g in _schema.groups]
        else:
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
        if _use_json_expansion:
            with st.spinner(f"Stage 2/3 — Searching PatentsView with structured JSON query…"):
                raw, unique = _timed("2. Search + dedup", _run_search_json, _schema)
        else:
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

        # ── JSON mode panel ────────────────────────────────────────────────
        if st.session_state.json_boolean:
            st.subheader("📋 JSON Query Results")

            # Boolean query
            st.markdown("**Generated Boolean Query**")
            st.code(st.session_state.json_boolean, language="")

            # Metrics row
            _jcount = st.session_state.json_result_count
            _count_label = f"{_jcount:,}" if _jcount is not None else "—"
            _term_count  = len(st.session_state.expanded_queries)
            _n_groups    = len(st.session_state.json_groups)

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Groups", _n_groups)
            mc2.metric("Unique terms", _term_count)
            mc3.metric("PatentsView hits", _count_label)

            # USPTO link
            if st.session_state.json_uspto_url:
                st.markdown(
                    f"🔗 **[Open in USPTO Patent Full-Text Search]({st.session_state.json_uspto_url})**",
                    unsafe_allow_html=False,
                )
                with st.expander("Show raw USPTO URL", expanded=False):
                    st.text(st.session_state.json_uspto_url)

            st.divider()
            st.markdown("**Terms sent to PatentsView search:**")
            for t in st.session_state.expanded_queries:
                st.markdown(f"- `{t}`")

        # ── Claude expansion mode panel ────────────────────────────────────
        else:
            st.success(f"{len(st.session_state.expanded_queries)} queries generated"
                       + (" (incl. original)" if use_mistral else " (Claude disabled)"))
            for i, q in enumerate(st.session_state.expanded_queries):
                label = "🔵 Original" if i == 0 else f"🟢 Expanded {i}"
                st.markdown(f"**{label}:** {q}")

            if st.session_state.claude_raw_response:
                with st.expander("🔍 Raw Claude response (for validation)", expanded=False):
                    st.code(st.session_state.claude_raw_response, language="json")

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

        _pv_total = st.session_state.json_result_count
        if _pv_total is not None:
            m1, m2, m3, m4 = st.columns(4)
            m4.metric("PatentsView total", f"{_pv_total:,}")
        else:
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
        _jg = st.session_state.json_groups
        for p in st.session_state.unique_patents:
            if _jg:
                st.markdown(
                    "• " + _badges_for(p.patent_title or "", p.patent_abstract or "", _jg),
                    unsafe_allow_html=True,
                )
            else:
                st.write(f"• {p.patent_id} — {p.patent_title}")
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
        _jg = st.session_state.json_groups
        for i, rp in enumerate(st.session_state.ranked, 1):
            p = rp.patent
            if _jg:
                st.markdown(
                    f"{i}. <b>{p.patent_id}</b> — "
                    + _badges_for(p.patent_title or "", p.patent_abstract or "", _jg),
                    unsafe_allow_html=True,
                )
            else:
                st.write(f"{i}. {p.patent_id} — {p.patent_title}")

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
        _jg = st.session_state.json_groups
        if _jg:
            from services.json_query_service import match_groups_in_text
            _matches = match_groups_in_text(
                f"{p.patent_title} {p.patent_abstract or ''}", _jg
            )
            if _matches:
                _badge_html = " ".join(_group_badge_html(idx, term) for idx, term in _matches)
                st.markdown(f"**Matched groups:** {_badge_html}", unsafe_allow_html=True)
            else:
                st.caption("🔴 No group terms matched in this patent's title or abstract.")
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
