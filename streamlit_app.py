"""
streamlit_app.py  —  Patent Intelligence UI

Run with:
    cd /home/dennis/venv/projects/sarosh/patent-search
    streamlit run streamlit_app.py
"""

import sys
import json
import time
import hashlib
import pickle
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

ranking_logger = logging.getLogger("services.ranking_service")
ranking_logger.setLevel(logging.DEBUG)

if not ranking_logger.handlers:
    ranking_logger.addHandler(logging.StreamHandler(sys.stdout))

ranking_logger.propagate = False

# Make sure the project root is on the path regardless of cwd
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings
from models.claude_client import ClaudeClient
from models.schemas import SearchStrategy, SearchConcept, ProximityRule
from services.ai_classification_service import (
    generate_classification_suggestions,
)
from services.classification_catalogue_service import (
    get_cpc_catalogue_entry,
)
from services.ai_invention_structure_service import (
    generate_invention_structure,
    build_search_paths_from_structure,
)
from services.classification_definition_service import (
    get_cpc_definition,
    get_uspc_definition,
)
from services.classification_service import aggregate_classifications
from services.classification_refinement_service import (
    filter_patents_by_classification,
)
from services.search_strategy_service import (
    build_boolean_search,
    build_core_pair_queries,
    build_search_queries,
    build_uspto_search_strings,
    build_uspto_proximity_strings,
)
from services.search_terminology import add_terminology_variants
from services.retrieval_diagnostics_service import (
    calculate_retrieval_concept_coverage,
    calculate_pairwise_query_overlap,
)

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

    # Phase 1 — manual search strategy
    "search_strategy": None,
    "search_strategy_approved": False,

    # Classification refinement
    "classification_refined_patents": [],
    "classification_refinement_applied": False,
    "ai_invention_structure": None,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def _reset():
    _preserved_keys = {"strategy_retrieval_mode"}

    for k, v in _DEFAULTS.items():
        if k not in _preserved_keys:
            st.session_state[k] = v


def _get_cached_classification_data(patents):
    """Reuse classification aggregates for an unchanged patent result set."""
    cache = st.session_state.setdefault("_classification_cache", {})

    fingerprint = hashlib.sha256(
        pickle.dumps(patents, protocol=pickle.HIGHEST_PROTOCOL)
    ).hexdigest()

    if fingerprint not in cache:
        cache[fingerprint] = aggregate_classifications(patents)

    while len(cache) > 3:
        cache.pop(next(iter(cache)))

    return cache[fingerprint]


# ── sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configuration")

    st.divider()
    st.subheader("🔍 Search Backend")
    _backend_name = (
        settings.patent_search_backend or settings.search_backend or "auto"
    ).lower()

    if _backend_name == "uspto" and settings.patentsview_api_key:
        st.success("🇺🇸 USPTO Patent Search (ODP) ✅")
        st.caption("Live USPTO patent data")

    elif _backend_name == "uspto" and not settings.patentsview_api_key:
        st.warning(
            "USPTO ODP — API key missing\\n"
            "Add `patentsview_api_key=` to .env"
        )

    elif _backend_name == "epo" and settings.epo_consumer_key:
        st.success("🇪🇺 EPO Open Patent Services (OPS) ✅")
        st.caption("Live EPO patent data")

    elif _backend_name == "epo" and not settings.epo_consumer_key:
        st.warning(
            "EPO OPS — credentials missing\\n"
            "Add `epo_consumer_key` and `epo_consumer_secret` to .env"
        )

    elif _backend_name == "patentsview" and settings.patentsview_api_key:
        st.success("USPTO ODP compatibility backend ✅")
        st.caption("Live USPTO patent data")

    elif _backend_name == "auto":
        st.info("Automatic backend selection")

    else:
        st.info("No live patent-search backend configured")

    st.divider()
    st.subheader("🤖 Active Models")
    st.caption(f"**Query Expansion:** {settings.anthropic_model} (API)")
    st.caption(f"**Embeddings:** {settings.embedding_model}")

    st.divider()
    top_k = st.number_input("Top-K results", min_value=1, max_value=100, value=20, key="top_k_results",)

    st.divider()
    st.subheader("Claude API")
    use_mistral = st.toggle("Use Claude for query expansion", value=True)

    st.divider()
    st.subheader("Pipeline stages")
    st.caption(
        "Run each stage in order, or jump straight to **Run to Top 20** "
        "which chains stages 1–3 automatically."
    )

    if st.button("🔄 Reset / New query", width="stretch"):
        logger.info("Reset button clicked — resetting session state")
        _reset()
        st.rerun()

    st.divider()
    if st.session_state.elapsed:
        st.subheader("⏱ Stage timings")
        for stage, secs in st.session_state.elapsed.items():
            st.caption(f"{stage}: {secs:.1f}s")

    st.divider()
    st.subheader("📊 Results Workspace")
    result_view = st.radio(
        "Open results screen",
        options=[
            "Concept Editor",
            "Search Results",
            "Technical Relevance",
            "Claude Technical Analysis",
        ],
        index=0,
        key="results_workspace_view_v2",
        label_visibility="collapsed",
    )
    st.caption("Choose a view to open it in the main workspace.")


# ── main ────────────────────────────────────────────────────────────────────────
# Persist the Technology query separately from its widget key. Streamlit may clean
# up widget-owned state when the Concept Editor screen is not rendered; this
# independent key preserves the user's text while navigating between screens.
_DEFAULT_TECHNOLOGY_QUERY = (
    "A system for autonomous vehicle lane detection using infrared sensors"
)


def _persist_technology_query():
    value = st.session_state.get("query_input", _DEFAULT_TECHNOLOGY_QUERY)
    st.session_state["technology_query_persisted"] = value
    # Keep the AI concept-generation description aligned with the main query
    # unless the user has explicitly edited that separate description.
    if not st.session_state.get("ai_invention_description_user_edited", False):
        st.session_state["ai_invention_description_persisted"] = value


def _persist_ai_invention_description():
    st.session_state["ai_invention_description_persisted"] = st.session_state.get(
        "ai_invention_description", st.session_state.get("technology_query_persisted", _DEFAULT_TECHNOLOGY_QUERY)
    )
    st.session_state["ai_invention_description_user_edited"] = True


if "technology_query_persisted" not in st.session_state:
    st.session_state["technology_query_persisted"] = st.session_state.get(
        "query_input", _DEFAULT_TECHNOLOGY_QUERY
    )

if "ai_invention_description_persisted" not in st.session_state:
    st.session_state["ai_invention_description_persisted"] = st.session_state.get(
        "ai_invention_description",
        st.session_state["technology_query_persisted"],
    )
if "ai_invention_description_user_edited" not in st.session_state:
    st.session_state["ai_invention_description_user_edited"] = False

# Restore widget values only when entering the screen and their widget keys have
# been cleaned up by Streamlit. Do not overwrite an actively edited widget value.
if result_view == "Concept Editor" and "query_input" not in st.session_state:
    st.session_state["query_input"] = st.session_state["technology_query_persisted"]
if result_view == "Concept Editor" and "ai_invention_description" not in st.session_state:
    st.session_state["ai_invention_description"] = st.session_state[
        "ai_invention_description_persisted"
    ]

query = (
    st.session_state.get("ai_invention_description", "").strip()
    or st.session_state.get("technology_query_persisted", "").strip()
    or _DEFAULT_TECHNOLOGY_QUERY
)
run_to_top20 = False
run_claude = False

if result_view == "Concept Editor":
    st.title("🔬 Patent Intelligence")

    # ── Phase 1 — Concept editor ─────────────────────────────────────────────────
    st.divider()
    st.subheader("🧭 1. Concept Editor")
    st.caption(
        "Generate and refine the technical concepts and patent terminology here. "
        "Approval creates the active search strategy; it does not run a patent search."
    )

    st.markdown("### 🤖 AI-assisted search concepts")
    st.caption(
        "Claude proposes 10 technical concepts, each with a primary term and "
        "two alternatives. Concepts are collapsed by default to keep this editor compact."
    )

    if "ai_search_concepts" not in st.session_state:
        st.session_state.ai_search_concepts = []

    if "ai_search_concepts_approved" not in st.session_state:
        st.session_state.ai_search_concepts_approved = False

    def _invalidate_ai_concept_approval():
        """Any concept edit invalidates the previously approved strategy."""
        st.session_state.ai_search_concepts_approved = False
        st.session_state.search_strategy_approved = False
        # Do not leave an old strategy visible or accidentally re-approvable.
        st.session_state.search_strategy = None


    _ai_invention = st.text_area(
        "Invention description for Claude",
        height=100,
        key="ai_invention_description",
        help="Describe the invention or technology you want to search for.",
        on_change=lambda: (_persist_ai_invention_description(), _invalidate_ai_concept_approval()),
    )
    st.session_state["ai_invention_description_persisted"] = _ai_invention

    if st.button("🤖 Generate 10 Search Concepts", key="generate_ai_search_concepts"):
        if not _ai_invention.strip():
            st.warning("Please enter an invention description first.")
        else:
            try:
                from models.claude_client import ClaudeClient
                from services.ai_search_strategy_service import generate_search_concepts

                with st.spinner("Claude is generating search concepts…"):
                    generated = generate_search_concepts(
                        _ai_invention,
                        ClaudeClient(),
                    )

                st.session_state.ai_search_concepts = generated
                st.session_state.ai_search_concepts_approved = False
                st.session_state.search_strategy_approved = False
                st.session_state.search_strategy = None
                st.success(
                    f"Generated {len(generated)} concepts. "
                    "Review and edit them below."
                )

            except Exception as exc:
                st.error(f"Unable to generate search concepts: {exc}")

    _ai_concepts = st.session_state.get("ai_search_concepts", [])

    if _ai_concepts:
        st.markdown("#### Review and edit concepts")
        if "ai_concepts_expand_all" not in st.session_state:
            st.session_state.ai_concepts_expand_all = False

        expand_col, summary_col = st.columns([1, 3])
        with expand_col:
            if st.button(
                "⬆ Collapse all" if st.session_state.ai_concepts_expand_all else "⬇ Expand all",
                key="ai_concepts_toggle_all",
            ):
                st.session_state.ai_concepts_expand_all = not st.session_state.ai_concepts_expand_all
                st.rerun()
        with summary_col:
            st.caption(
                f"{len(_ai_concepts)} concepts · Expand a row to edit terms, importance, and rationale."
            )

        edited_concepts = []

        for index, concept in enumerate(_ai_concepts, start=1):
            primary_term = concept.get("primary_term", "")
            alternative_terms = concept.get("alternative_terms", [])

            confidence = str(
                concept.get("confidence", "low")
            ).lower()

            importance_options = [
                "critical",
                "important",
                "supporting",
            ]

            recommended_importance = str(
                concept.get("importance", "important")
            ).lower()

            if recommended_importance not in importance_options:
                recommended_importance = "important"

            # Compact summary shown when the concept is collapsed.
            concept_label = (
                f"Concept {index} · "
                f"{concept.get('name', 'Untitled concept')} · "
                f"{recommended_importance.title()} · "
                f"{confidence.title()} confidence"
            )

            with st.expander(
                concept_label,
                expanded=st.session_state.ai_concepts_expand_all,
            ):
                concept_name = st.text_input(
                    "Concept name",
                    value=concept.get("name", ""),
                    key=f"ai_concept_name_{index}",
                    on_change=_invalidate_ai_concept_approval,
                )

                edited_primary_term = st.text_input(
                    "Primary search term",
                    value=primary_term,
                    key=f"ai_concept_{index}_primary_term",
                    on_change=_invalidate_ai_concept_approval,
                )

                edited_alternative_terms = []

                for alternative_index in range(2):
                    default_alternative = (
                        alternative_terms[alternative_index]
                        if alternative_index < len(alternative_terms)
                        else ""
                    )

                    edited_alternative = st.text_input(
                        f"Alternative search term {alternative_index + 1}",
                        value=default_alternative,
                        key=(
                            f"ai_concept_{index}_alternative_"
                            f"{alternative_index + 1}"
                        ),
                        on_change=_invalidate_ai_concept_approval,
                    )

                    edited_alternative_terms.append(
                        edited_alternative
                    )

                st.caption(
                    "Claude's recommendation: "
                    f"{concept.get('classification_reason', 'No reason provided.') }"
                )

                st.caption(
                    f"Classification confidence: {confidence.title()}"
                )

                importance = st.selectbox(
                    "Importance — review or override Claude",
                    importance_options,
                    index=importance_options.index(
                        recommended_importance
                    ),
                    key=f"ai_concept_{index}_importance",
                    on_change=_invalidate_ai_concept_approval,
                )

            edited_concepts.append(
                {
                    "name": concept_name,
                    "primary_term": edited_primary_term,
                    "alternative_terms": edited_alternative_terms,
                    "importance": importance,
                    "classification_reason": concept.get(
                        "classification_reason",
                        "No reason provided.",
                    ),
                    "confidence": concept.get("confidence", "low"),
                }
            )

        st.session_state.ai_search_concepts = edited_concepts

        if st.button(
            "✅ Approve Concepts & Search Strategy",
            type="primary",
            key="approve_ai_search_concepts",
        ):
            _approved_strategy_concepts = []
            _validation_errors = []

            for _index, _concept in enumerate(edited_concepts):
                _name = str(_concept.get("name", "")).strip()
                _primary_term = str(_concept.get("primary_term", "")).strip()
                _alternative_terms = [
                    str(term).strip()
                    for term in _concept.get("alternative_terms", [])
                    if str(term).strip()
                ]
                _terms = ([ _primary_term ] if _primary_term else []) + _alternative_terms

                if not _name:
                    _validation_errors.append(f"Concept {_index + 1}: add a concept name.")
                if not _terms:
                    _validation_errors.append(
                        f"Concept {_index + 1}: add a primary term or at least one alternative."
                    )

                if _name and _terms:
                    _approved_strategy_concepts.append(
                        SearchConcept(
                            name=_name,
                            importance=str(_concept.get("importance", "important")).lower(),
                            terms=_terms,
                            operator="OR",
                        )
                    )

            if _validation_errors:
                st.error("Please fix the following before approval:\n- " + "\n- ".join(_validation_errors))
            elif not _approved_strategy_concepts:
                st.error("There are no valid concepts to approve.")
            else:
                st.session_state.search_strategy = SearchStrategy(
                    original_input=_ai_invention.strip(),
                    key_inventive_points=[],
                    concepts=_approved_strategy_concepts,
                    concept_operator="AND",
                    search_fields=["title", "abstract", "claims"],
                )
                st.session_state.ai_search_concepts_approved = True
                st.session_state.search_strategy_approved = True
                st.session_state.active_search_strategy_source = "concepts"

                # Keep the optional manual editor populated for users who choose to
                # refine the generated strategy manually later.
                st.session_state.strategy_concept_count = len(edited_concepts)
                for _index, _concept in enumerate(edited_concepts):
                    st.session_state[f"strategy_concept_name_{_index}"] = str(
                        _concept.get("name", "")
                    ).strip()
                    _terms = [
                        str(term).strip()
                        for term in (
                            [_concept.get("primary_term", "")]
                            + _concept.get("alternative_terms", [])
                        )
                        if str(term).strip()
                    ]
                    st.session_state[f"strategy_concept_terms_{_index}"] = "\n".join(_terms)
                    st.session_state[f"strategy_concept_importance_{_index}"] = _concept.get(
                        "importance", "important"
                    )
                    st.session_state[f"strategy_concept_operator_{_index}"] = "OR"

                st.success(
                    "Concepts approved. The active search strategy is ready; "
                    "configure retrieval separately below. No search has run yet."
                )

        if st.session_state.get("ai_search_concepts_approved", False):
            st.success(
                "Approved strategy is ready for retrieval configuration. "
                "Editing or regenerating a concept will require approval again."
            )

            if st.button(
                "🔎 Run AI Patent Search & Review",
                key="run_ai_patent_feedback_workflow",
                type="primary",
                disabled=True,
            ):
                invention = str(
                    st.session_state.get("ai_invention_description", "")
                ).strip()
                approved_concepts = st.session_state.get(
                    "ai_search_concepts", []
                )

                st.session_state.pop("ai_feedback_workflow_result", None)
                st.session_state.pop("ai_feedback_workflow_error", None)

                if not invention:
                    st.session_state.ai_feedback_workflow_error = (
                        "Please enter an invention description."
                    )
                elif not approved_concepts:
                    st.session_state.ai_feedback_workflow_error = (
                        "No approved search concepts were found."
                    )
                else:
                    try:
                        from models.claude_client import ClaudeClient
                        from services.search_feedback_workflow import (
                            run_patent_search_feedback_workflow,
                        )

                        with st.spinner(
                            "Generating search queries, retrieving patents, "
                            "ranking evidence, and reviewing results..."
                        ):
                            result = run_patent_search_feedback_workflow(
                                invention=invention,
                                approved_concepts=approved_concepts,
                                client=ClaudeClient(),
                                max_runs=3,
                            )

                        st.session_state.ai_feedback_workflow_result = result

                    except Exception as exc:
                        st.session_state.ai_feedback_workflow_error = str(exc)

        if st.session_state.get("ai_feedback_workflow_error"):
            st.error(
                "The AI patent search workflow failed: "
                + st.session_state.ai_feedback_workflow_error
            )

        if st.session_state.get("ai_feedback_workflow_result") is not None:
            result = st.session_state.ai_feedback_workflow_result

            usage = result.get("token_usage", {})

            if usage:
                st.markdown("#### Claude token usage")

                metric_cols = st.columns(3)
                metric_cols[0].metric(
                    "Input tokens", f"{usage.get('input_tokens', 0):,}"
                )
                metric_cols[1].metric(
                    "Output tokens", f"{usage.get('output_tokens', 0):,}"
                )
                metric_cols[2].metric(
                    "Total tokens", f"{usage.get('total_tokens', 0):,}"
                )

                st.caption(
                    f"Claude API calls: {usage.get('calls', 0)} · "
                    "Includes successful calls and validation retries."
                )

                with st.expander("Token usage by stage"):
                    stage_rows = [
                        {"Stage": stage, **values}
                        for stage, values in usage.get("by_stage", {}).items()
                    ]
                    if stage_rows:
                        st.dataframe(stage_rows, hide_index=True, use_container_width=True)

                with st.expander("Individual Claude API calls"):
                    call_rows = [
                        {
                            "Stage": event["stage"],
                            "Model": event["model"],
                            "Input": event["input_tokens"],
                            "Output": event["output_tokens"],
                            "Total": event["total_tokens"],
                        }
                        for event in usage.get("events", [])
                    ]
                    if call_rows:
                        st.dataframe(
                            call_rows, hide_index=True, use_container_width=True
                        )

            st.markdown("### 📊 AI Patent Search & Review Results")

            if "ai_feedback_expand_all" not in st.session_state:
                st.session_state.ai_feedback_expand_all = False

            control_col, status_col = st.columns([1, 3])

            with control_col:
                if st.button(
                    (
                        "⬆️ Collapse all"
                        if st.session_state.ai_feedback_expand_all
                        else "⬇️ Expand all"
                    ),
                    key="ai_feedback_toggle_all",
                ):
                    st.session_state.ai_feedback_expand_all = (
                        not st.session_state.ai_feedback_expand_all
                    )
                    st.rerun()

            with status_col:
                st.caption(
                    f"Runs: {result.get('run_count', 0)} / "
                    f"{result.get('max_runs', 3)} · "
                    f"Decision: {result.get('decision', 'unknown')} · "
                    f"Stop reason: {result.get('stop_reason', 'unknown')}"
                )

            expanded = st.session_state.ai_feedback_expand_all
            history = result.get("history", [])

            for run in history:
                run_number = run.get("run_number", "?")
                plan = run.get("plan", {})
                queries = plan.get("queries", [])
                patents = run.get("results", [])
                review = run.get("review", {})

                with st.expander(
                    f"Run {run_number} — "
                    f"{len(queries)} queries · "
                    f"{len(patents)} retrieved records · "
                    f"{review.get('decision', 'not reviewed')}",
                    expanded=expanded,
                ):
                    st.markdown("#### Search plan")

                    for query in queries:
                        with st.expander(
                            query.get("name", "Unnamed query"),
                            expanded=expanded,
                        ):
                            st.write(query.get("purpose", ""))
                            st.json(query.get("groups", []), expanded=expanded)

                    st.markdown("#### 🏆 Top 8 ranked candidates")

                    ranked_candidates = review.get("ranked_candidates", [])

                    if not ranked_candidates:
                        st.info("No ranked candidates were retained for this run.")
                    else:
                        st.caption(
                            f"{len(ranked_candidates)} candidates selected by the ranker. "
                            "Scores indicate ranking relevance, not verified patent overlap."
                        )

                        for rank_position, patent in enumerate(ranked_candidates, start=1):
                            patent_id = patent.get("patent_id", "Unknown ID")
                            title = patent.get("title", "Untitled patent")

                            with st.expander(
                                f"#{rank_position} — {patent_id} — {title}",
                                expanded=expanded,
                            ):
                                score_cols = st.columns(4)
                                score_cols[0].metric(
                                    "Hybrid",
                                    f"{patent.get('hybrid_score', 0):.4f}",
                                )
                                score_cols[1].metric(
                                    "Cosine",
                                    f"{patent.get('cosine_score', 0):.4f}",
                                )
                                score_cols[2].metric(
                                    "BM25",
                                    f"{patent.get('bm25_score', 0):.4f}",
                                )
                                score_cols[3].metric(
                                    "Coverage",
                                    f"{patent.get('coverage', 0):.2f}",
                                )

                                st.write("**Abstract**")
                                st.write(patent.get("abstract") or "Not available")

                                st.write("**Claims**")
                                st.write(patent.get("claims") or "Not available")

                                if patent.get("search_queries"):
                                    st.caption(
                                        "Found by: " + ", ".join(patent["search_queries"])
                                    )

                    st.markdown("#### All retrieved patent records")
                    if not patents:
                        st.info(
                            "No patent records were retrieved in this run. "
                            "This does not establish that relevant prior art is absent."
                        )
                    else:
                        for patent in patents:
                            patent_id = patent.get("patent_id", "Unknown ID")
                            title = patent.get("title", "Untitled patent")

                            with st.expander(
                                f"{patent_id} — {title}",
                                expanded=expanded,
                            ):
                                st.write("**Abstract**")
                                st.write(patent.get("abstract") or "Not available")

                                st.write("**Claims**")
                                st.write(patent.get("claims") or "Not available")

                                if patent.get("search_queries"):
                                    st.caption(
                                        "Found by: "
                                        + ", ".join(patent["search_queries"])
                                    )

                                for score in (
                                    "cosine_score",
                                    "bm25_score",
                                    "hybrid_score",
                                ):
                                    if patent.get(score) is not None:
                                        st.caption(
                                            f"{score}: {patent[score]}"
                                        )

                    st.markdown("#### Review and coverage")

                    st.write(
                        review.get(
                            "rationale",
                            "No review rationale was recorded.",
                        )
                    )

                    gaps = review.get("coverage_gaps", [])
                    if gaps:
                        st.write("**Coverage gaps**")
                        for gap in gaps:
                            st.markdown(f"- {gap}")

                    revisions = review.get("queries_to_revise", [])
                    if revisions:
                        st.write("**Recommended query revisions**")
                        st.json(revisions, expanded=expanded)

                    patent_reviews = review.get("patent_reviews", [])
                    if patent_reviews:
                        st.markdown("**Evidence review by patent**")

                        for patent_review in patent_reviews:
                            with st.expander(
                                (
                                    f"{patent_review.get('patent_id', 'Unknown ID')} "
                                    f"— {patent_review.get('label', 'unlabelled')}"
                                ),
                                expanded=expanded,
                            ):
                                st.write(patent_review.get("evidence", ""))

                                evidence_excerpts = patent_review.get(
                                    "evidence_excerpts", []
                                )
                                if evidence_excerpts:
                                    st.markdown("**Source evidence excerpts**")
                                    for excerpt in evidence_excerpts:
                                        st.caption(
                                            f"Source: {excerpt.get('source', 'unknown')}"
                                        )
                                        st.markdown(
                                            f"> {excerpt.get('excerpt', '')}"
                                        )
                                        st.caption(
                                            "Mechanism supported: "
                                            f"{excerpt.get('supports', '')}"
                                        )
                                else:
                                    st.warning(
                                        "No verified source excerpts are available "
                                        "for this patent review."
                                    )

                                st.write("**Matched mechanisms**")
                                st.write(
                                    patent_review.get("matched_mechanisms", [])
                                )

                                st.write("**Missing mechanisms**")
                                st.write(
                                    patent_review.get("missing_mechanisms", [])
                                )

            with st.expander("Raw workflow JSON", expanded=False):
                st.json(result, expanded=False)

    st.markdown("### 🧩 AI Invention Structure")

    st.caption(
        "Claude can decompose the invention into application, core system, "
        "sensors, functions, and objects, then generate structured patent-search "
        "paths. Review and edit the structure before generating search paths."
    )

    _invention_for_structure = st.session_state.get(
        "ai_invention_description",
        "",
    ).strip()

    if _invention_for_structure:
        if st.button(
            "🧩 Build Invention Search Structure",
            key="build_ai_invention_structure",
        ):
            try:
                with st.spinner(
                    "Claude is building the invention search structure..."
                ):
                    _structure = generate_invention_structure(
                        _invention_for_structure,
                        ClaudeClient(),
                    )

                _debug_search_paths = _structure.get("search_paths", [])
                if _debug_search_paths:
                    st.write(
                        "DEBUG SEARCH PATHS:",
                        [path.get("name") for path in _debug_search_paths],
                    )
                    st.json(_debug_search_paths)

                st.session_state.ai_invention_structure = _structure
                logger.info(
                    "AI INVENTION STRUCTURE:\n%s",
                    json.dumps(_structure, indent=2, ensure_ascii=False),
                )
                st.session_state.ai_search_path_generation = (
                    st.session_state.get("ai_search_path_generation", 0) + 1
                )
                st.session_state.ai_search_paths_approved = False

            except Exception as exc:
                st.error(
                    f"Unable to build invention search structure: {exc}"
                )

    _structure = st.session_state.get(
        "ai_invention_structure"
    )

    if _structure:
        st.markdown("#### Review Invention Structure")

        st.caption(
            "Edit the suggested terminology as needed. "
            "Regenerate the search paths after making changes."
        )

        _roles = [
            ("Application", "application"),
            ("Core System", "core_system"),
            ("Sensors", "sensors"),
            ("Functions", "functions"),
            ("Objects", "objects"),
        ]

        _edited_structure = {}

        _role_columns = st.columns(5)

        for _column, (_label, _key) in zip(
            _role_columns,
            _roles,
        ):
            with _column:
                st.markdown(f"**{_label}**")

                _current_terms = _structure.get(
                    _key,
                    [],
                )

                _edited_text = st.text_area(
                    _label,
                    value="\n".join(_current_terms),
                    key=f"ai_structure_edit_{_key}",
                    height=220,
                    label_visibility="collapsed",
                )

                _edited_structure[_key] = [
                    _term.strip()
                    for _term in _edited_text.splitlines()
                    if _term.strip()
                ]

        st.session_state.ai_edited_invention_structure = (
            _edited_structure
        )

        if st.button(
            "🔄 Regenerate Search Paths",
            key="regenerate_ai_search_paths",
        ):
            _reviewed_structure = st.session_state.get(
                "ai_edited_invention_structure",
                {},
            )

            _updated_structure = {
                "application": list(
                    _reviewed_structure.get("application", [])
                ),
                "core_system": list(
                    _reviewed_structure.get("core_system", [])
                ),
                "sensors": list(
                    _reviewed_structure.get("sensors", [])
                ),
                "functions": list(
                    _reviewed_structure.get("functions", [])
                ),
                "objects": list(
                    _reviewed_structure.get("objects", [])
                ),
            }

            _updated_structure["search_paths"] = (
                build_search_paths_from_structure(
                    _updated_structure
                )
            )

            logger.info(
                "GENERATED SEARCH PATHS:\n%s",
                json.dumps(
                    _updated_structure["search_paths"],
                    indent=2,
                    ensure_ascii=False,
                ),
            )

            st.session_state.ai_invention_structure = (
                _updated_structure
            )
            st.session_state.ai_search_path_generation = (
                st.session_state.get("ai_search_path_generation", 0) + 1
            )

            st.session_state.ai_edited_invention_structure = (
                _updated_structure
            )

            st.session_state.ai_search_paths_approved = False
            st.session_state.ai_approved_search_paths = []

            for _index in range(6):
                st.session_state.pop(
                    f"ai_search_path_expression_{_index}",
                    None,
                )
                st.session_state.pop(
                    f"ai_search_path_selected_{_index}",
                    None,
                )

            st.rerun()

        with st.expander("Debug: Generated Search Paths"):
            st.json(_structure.get("search_paths", []))

        st.markdown("#### Generated Search Paths")

        st.caption(
            "Select the search paths you want to review. "
            "You can also edit the Boolean expression before approval."
        )

        _paths = _structure.get(
            "search_paths",
            [],
        )

        _selected_search_paths = []

        for _index, _path in enumerate(_paths):
            _path_name = str(
                _path.get("name", "")
            ).strip()

            if not _path_name:
                continue

            _expression = str(
                _path.get("expression", "")
            ).strip()

            _purpose = str(
                _path.get("purpose", "")
            ).strip()

            _selected = st.checkbox(
                _path_name,
                key=f"ai_search_path_selected_{_index}",
            )

            if _selected:
                _selected_search_paths.append(
                    _index
                )

            if _purpose:
                st.caption(_purpose)

            st.text_area(
                f"Search expression — {_path_name}",
                value=_expression,
                key=(
                    f"ai_search_path_expression_"
                    f"{st.session_state.get('ai_search_path_generation', 0)}_"
                    f"{_index}"
                ),
                height=90,
            )

            st.divider()

        if _selected_search_paths:
            st.markdown(
                f"**{len(_selected_search_paths)} search path(s) selected**"
            )
        else:
            st.info(
                "Select at least one search path to approve it."
            )

        if st.button(
            "✅ Approve Selected Search Paths",
            key="approve_ai_search_paths",
            disabled=not _selected_search_paths,
        ):
            _approved_paths = []

            for _index in _selected_search_paths:
                _path = _paths[_index]

                _approved_paths.append(
                    {
                        "name": str(
                            _path.get("name", "")
                        ).strip(),
                        "roles": list(
                            _path.get("roles", [])
                        ),
                        "expression": st.session_state.get(
                            (
                                f"ai_search_path_expression_"
                                f"{st.session_state.get('ai_search_path_generation', 0)}_"
                                f"{_index}"
                            ),
                            "",
                        ).strip(),
                        "purpose": str(
                            _path.get("purpose", "")
                        ).strip(),
                        # Search-quality metadata
                        "quality_level": str(
                            _path.get(
                                "quality_level",
                                "balanced",
                            )
                        ).strip(),
                        "recall_priority": int(
                            _path.get(
                                "recall_priority",
                                0,
                            )
                        ),
                        "precision_priority": int(
                            _path.get(
                                "precision_priority",
                                0,
                            )
                        ),
                    }
                )

            st.session_state.ai_approved_search_paths = (
                _approved_paths
            )

            st.session_state.ai_search_paths_approved = True

            st.success(
                f"{len(_approved_paths)} search path(s) approved "
                "for the next search stage."
            )

        if st.session_state.get(
            "ai_search_paths_approved",
            False,
        ):
            st.markdown("#### Approved Search Paths")

            for _path in st.session_state.get(
                "ai_approved_search_paths",
                [],
            ):
                st.markdown(
                    f"**{_path['name']}**"
                )

                st.code(
                    _path["expression"],
                    language="text",
                )

        if _structure:
            with st.expander("Debug: AI Invention Structure"):
                st.json(_structure)

    with st.expander("Retrieval Configuration", expanded=True):
        st.caption(
            "Choose the retrieval breadth separately from concept editing and approval. "
            "This setting affects query generation, not the approved concept definitions."
        )
        _retrieval_mode = st.radio(
            "How should critical concepts be searched?",
            options=["Broad", "Balanced", "Strict"],
            index=0,
            format_func=lambda mode: {
                "Broad": "Broad — search each critical concept independently",
                "Balanced": "Balanced — individual searches + concept pairs",
                "Strict": "Strict — require all critical concepts",
            }[mode],
            help=(
                "Broad maximizes discovery. Balanced also searches pairs of "
                "critical concepts. Strict requires all critical concepts "
                "in one Boolean query."
            ),
            key="strategy_retrieval_mode",
        )

    # ── Retrieval execution and results ───────────────────────────────────────────
    st.divider()
    st.subheader("3. Retrieval Execution & Results")
    st.caption(
        "Run the approved strategy or the standard pipeline here. Retrieved records, "
        "deduplication, ranking, and analysis appear in the results below."
    )

    col1, col2 = st.columns([2, 1])

    run_to_top20 = col1.button(
        "▶ Run to Top 20",
        type="primary",
        width="stretch",
    )

    col2.button(
        "🔄 Reset",
        on_click=_reset,
        width="stretch",
    )

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

    # Approved AI Invention Structure search paths
    if (
        st.session_state.get("ai_search_paths_approved", False)
        and st.session_state.get("ai_approved_search_paths")
        and backend == "uspto"
        and settings.patentsview_api_key
    ):
        from services.uspto_search_service import (
            search_patents_by_strategy,
        )

        approved_paths = st.session_state.get(
            "ai_approved_search_paths",
            [],
        )

        raw = []
        path_results = []
        path_provenance = []

        for path in approved_paths:
            expression = str(path.get("expression", "")).strip()

            if not expression:
                continue

            path_name = path.get("name", "Unnamed path")

            logger.info(
                "USPTO STRATEGY SEARCH | path=%s | query=%s",
                path_name,
                expression,
            )

            limit = path.get("limit", 100)
            results = search_patents_by_strategy(
                query=expression,
                limit=limit,
                offset=0,
            )

            logger.info(
                "SEARCH RESULT | path=%s | count=%d",
                path_name,
                len(results),
            )

            if path_name in {
                "Core + Function",
                "Core + Sensor",
                "Core + Function + Sensor",
            } and not results:
                logger.warning(
                    "ZERO CORE RESULTS | path=%s | query=%s",
                    path_name,
                    expression,
                )

            path_results.append(
                {
                    "name": path_name,
                    "expression": expression,
                    "original_expression": expression,
                    "reruns": [],
                    "status": (
                        "success" if results else "zero_results"
                    ),
                    "error": None,
                    "quality_level": path.get(
                        "quality_level",
                        "balanced",
                    ),
                    "recall_priority": path.get(
                        "recall_priority",
                        0,
                    ),
                    "precision_priority": path.get(
                        "precision_priority",
                        0,
                    ),
                    "count": len(results),
                    "limit": limit,
                }
            )

            for patent in results:
                if patent is None:
                    continue

                existing_paths = list(getattr(patent, "search_paths", []) or [])
                if path_name not in existing_paths:
                    patent.search_paths = existing_paths + [path_name]

                path_provenance.append(
                    {
                        "patent": patent,
                        "path": path_name,
                    }
                )

                raw.append(patent)

            logger.info(
                "AI Invention Structure path '%s' query: %s",
                path_name,
                expression,
            )

            logger.info(
                "AI Invention Structure path '%s' returned %d results",
                path_name,
                len(results),
            )

        st.session_state.ai_search_path_results = path_results
        st.session_state.ai_search_path_provenance = path_provenance

        unique = deduplicate(raw)
        from collections import Counter
        path_counter = Counter()
        for patent in unique:
            for path in getattr(patent, "search_paths", []):
                path_counter[path] += 1
        logger.info(
            "UNIQUE CONTRIBUTION BY SEARCH PATH: %s",
            dict(path_counter),
        )
        return raw, unique

    # Approved manual Search Strategy
    if (
        st.session_state.get("search_strategy_approved", False)
        and st.session_state.get("search_strategy") is not None
        and backend == "uspto"
        and settings.patentsview_api_key
    ):
        from services.search_strategy_service import build_queries_for_mode
        from services.uspto_search_service import search_patents_by_strategy

        strategy = st.session_state.search_strategy
        retrieval_mode = st.session_state.get(
            "strategy_retrieval_mode",
            "Broad",
        )
        strategy_queries = build_queries_for_mode(
            strategy,
            retrieval_mode,
        )

        logger.info(
            "MANUAL STRATEGY RETRIEVAL | mode=%s | query_count=%d",
            retrieval_mode,
            len(strategy_queries),
        )

        raw = []
        query_diagnostics = []
        query_family_sets = []

        seen_publication_ids = set()
        seen_family_keys = set()

        for query_index, strategy_query in enumerate(
            strategy_queries,
            start=1,
        ):
            if not strategy_query:
                continue

            logger.info(
                "USPTO STRATEGY SEARCH | mode=%s | query=%d/%d | query=%s",
                retrieval_mode,
                query_index,
                len(strategy_queries),
                strategy_query,
            )

            results = search_patents_by_strategy(
                query=strategy_query,
                limit=100,
                offset=0,
            )

            valid_results = [
                patent for patent in results
                if patent is not None
            ]

            # Match the normalization and family fallback rules
            # used by services.dedup_service.
            from services.dedup_service import (
                _publication_key,
                _family_key,
            )

            new_publication_ids = set()
            new_family_keys = set()
            query_family_keys = set()

            missing_publication_id_count = 0
            missing_family_id_count = 0

            for patent in valid_results:
                publication_id = _publication_key(patent)

                if not publication_id:
                    missing_publication_id_count += 1
                elif publication_id not in seen_publication_ids:
                    new_publication_ids.add(publication_id)

                family_id = str(
                    getattr(patent, "continuity_family_id", "") or ""
                ).strip()

                if not family_id:
                    missing_family_id_count += 1

                family_key = _family_key(patent)

                if family_key:
                    query_family_keys.add(family_key)

                    if family_key not in seen_family_keys:
                        new_family_keys.add(family_key)

            seen_publication_ids.update(new_publication_ids)
            seen_family_keys.update(new_family_keys)
            query_family_sets.append(
                (query_index, strategy_query, query_family_keys)
            )

            query_diagnostics.append(
                {
                    "query_index": query_index,
                    "query": strategy_query,
                    "raw_count": len(results),
                    "valid_count": len(valid_results),
                    "new_publications": len(new_publication_ids),
                    "new_families": len(new_family_keys),
                    "missing_publication_id_count": (
                        missing_publication_id_count
                    ),
                    "missing_family_id_count": (
                        missing_family_id_count
                    ),
                }
            )

            logger.info(
                "RETRIEVAL DIAGNOSTIC | mode=%s | query=%d | "
                "valid=%d | new_publications=%d | new_families=%d",
                retrieval_mode,
                query_index,
                len(valid_results),
                len(new_publication_ids),
                len(new_family_keys),
            )

            for patent in valid_results:
                path_name = "Manual Search Strategy"
                existing_paths = list(
                    getattr(patent, "search_paths", []) or []
                )

                if path_name not in existing_paths:
                    patent.search_paths = existing_paths + [path_name]

            raw.extend(valid_results)

        query_overlap_diagnostics = calculate_pairwise_query_overlap(
            query_family_sets
        )

        unique = deduplicate(raw)

        concept_coverage = calculate_retrieval_concept_coverage(
            unique,
            strategy,
        )

        st.session_state.manual_retrieval_diagnostics = {
            "mode": retrieval_mode,
            "query_count": len(query_diagnostics),
            "queries": query_diagnostics,
            "query_overlap": query_overlap_diagnostics,
            "raw_count": len(raw),
            "unique_count": len(unique),
            "concept_coverage": concept_coverage,
        }

        from collections import Counter

        path_counter = Counter()
        for patent in unique:
            for path in getattr(patent, "search_paths", []):
                path_counter[path] += 1

        logger.info(
            "UNIQUE CONTRIBUTION BY SEARCH PATH: %s",
            dict(path_counter),
        )

        logger.info(
            "USPTO strategy search — mode=%s, queries=%d, "
            "raw=%d, unique=%d",
            retrieval_mode,
            len(query_diagnostics),
            len(raw),
            len(unique),
        )

        return raw, unique

    if backend == "epo" and settings.epo_consumer_key:
        from services.epo_search_service import epo_fetch_by_keywords
        meta = st.session_state.get("expansion_metadata", {})

        # Use the full tokens object stored by expand_query_with_metadata
        tokens = meta.get("tokens")
        if tokens is not None:
            logger.info(
                "EPO search — primary_anchor='%s' concept='%s'",
                tokens.primary_anchor, tokens.primary_concept,
            )
            raw = epo_fetch_by_keywords(tokens)
        else:
            # Fallback: build a minimal ExtractedTokens from the metadata dict
            from services.token_extractor import ExtractedTokens
            keyword_terms = meta.get("epo_search_order", [])[:4]
            if not keyword_terms:
                keyword_terms = meta.get("critical_tokens", [])[:4]
            if not keyword_terms and queries:
                keyword_terms = [w for w in queries[0].lower().split() if len(w) > 5][:4]
            logger.info("EPO CQL order (streamlit fallback): %s", keyword_terms)
            stub = ExtractedTokens(
                primary_anchor  = keyword_terms[0] if keyword_terms else "",
                primary_concept = "unknown",
                critical_tokens = keyword_terms,
                epo_search_order= keyword_terms,
            )
            raw = epo_fetch_by_keywords(stub)
    elif backend == "lens" and settings.lens_api_token:
        from services.lens_search_service import fetch_all_patents
        raw = fetch_all_patents(queries)
    elif backend == "uspto":
        from services.search_service import fetch_patents_with_fallback

        meta = st.session_state.get("expansion_metadata", {})
        tokens = meta.get("tokens")

        if tokens is not None:
            raw = fetch_patents_with_fallback(
                tokens,
                queries,
                min_results=15,
            )
        else:
            logger.warning(
                "USPTO search: no extracted tokens available; "
                "falling back to expanded-query search."
            )
            from services.search_service import fetch_all_patents
            raw = fetch_all_patents(queries)

    elif backend == "patentsview":
        from services.search_service import fetch_all_patents
        raw = fetch_all_patents(queries)
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
    # Pass expanded queries so embed_query applies BGE prefix + expansion enrichment
    expanded = st.session_state.get("expanded_queries", None)
    query_vec = embed_query(q, em, expanded=expanded)
    # Pass critical_tokens + synonyms for anchor penalty
    meta = st.session_state.get("expansion_metadata", {})
    tokens = meta.get("tokens")
    critical = list(dict.fromkeys(
        meta.get("critical_tokens", []) + meta.get("patent_synonyms", [])
    ))
    search_strategy = st.session_state.get("search_strategy")
    logger.warning(
        "RANK STRATEGY DEBUG | present=%s | type=%s | approved=%s",
        search_strategy is not None,
        type(search_strategy).__name__ if search_strategy is not None else None,
        st.session_state.get("search_strategy_approved", False),
    )

    logger.warning(
        "RANK STRATEGY CONCEPTS | %s",
        [
            {
                "name": getattr(c, "name", None),
                "importance": getattr(c, "importance", None),
                "terms": getattr(c, "terms", None),
                "operator": getattr(c, "operator", None),
            }
            for c in getattr(search_strategy, "concepts", [])
        ],
    )

    logger.warning(
        "RANK TOKEN DEBUG | tokens_present=%s | token_type=%s | "
        "taxonomy_miss=%s | concept_groups=%s | critical_count=%d",
        tokens is not None,
        type(tokens).__name__ if tokens is not None else None,
        getattr(tokens, "taxonomy_miss", None),
        list(getattr(tokens, "concept_groups", {}).keys())
        if tokens is not None else [],
        len(critical),
    )

    return rank(
        query=q,
        patents=patents,
        doc_vecs=doc_vecs,
        query_vec=query_vec,
        top_k=k,
        critical_tokens=critical or None,
        tokens=tokens,
        search_strategy=search_strategy,
    )


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


def _run_claude(q: str, ranked: list) -> str:
    from models.claude_client import ClaudeClient
    from services.comparison_service import compare_patents

    claude = ClaudeClient()
    return compare_patents(q, ranked, claude)


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
    # Preserve JSON query and approved search state across the reset
    _saved_json = st.session_state.json_query

    _saved_ai_paths_approved = st.session_state.get(
        "ai_search_paths_approved",
        False,
    )
    _saved_ai_approved_paths = st.session_state.get(
        "ai_approved_search_paths",
        [],
    )

    _saved_strategy = st.session_state.get(
        "search_strategy",
    )
    _saved_strategy_approved = st.session_state.get(
        "search_strategy_approved",
        False,
    )

    _reset()

    st.session_state.json_query = _saved_json

    st.session_state.ai_search_paths_approved = (
        _saved_ai_paths_approved
    )
    st.session_state.ai_approved_search_paths = (
        _saved_ai_approved_paths
    )

    st.session_state.search_strategy = _saved_strategy
    st.session_state.search_strategy_approved = (
        _saved_strategy_approved
    )

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
            # Stage 3 — technical relevance ranking
            # Use classification-refined results when the user has
            # explicitly applied classification refinement.
            ranking_patents = unique

            if st.session_state.get(
                "classification_refinement_applied",
                False,
            ):
                ranking_patents = st.session_state.get(
                    "classification_refined_patents",
                    unique,
                )

                logger.info(
                    "Classification refinement active: "
                    "%d → %d patents before ranking",
                    len(unique),
                    len(ranking_patents),
                )

            with st.spinner(
                f"Stage 3 — Technical relevance ranking "
                f"{len(ranking_patents)} patents…"
            ):
                ranked = _timed(
                    "3. Embed + rank",
                    _run_rank,
                    query,
                    ranking_patents,
                    top_k,
                )

                logger.debug(
                    "Ranking result count: %d",
                    len(ranked),
                )

            st.session_state.ranked = ranked
            st.session_state.stage = 3
            st.rerun()

    except Exception as exc:
        logger.exception("Unhandled exception during pipeline run")
        st.session_state.error = f"{type(exc).__name__}: {exc}"

if run_claude and st.session_state.stage >= 3:
    try:
        # Invalidate the previous analysis before generating a new one.
        st.session_state.comparison = ""
        st.session_state.report = ""

        with st.spinner("Calling Claude for technical comparison…"):
            comparison = _timed(
                "4. Claude comparison",
                _run_claude,
                query,
                st.session_state.ranked,
            )

        st.session_state.comparison = comparison
        st.session_state.stage = 4

    except Exception as exc:
        logger.exception("Unhandled exception during Claude comparison")
        st.session_state.error = f"{type(exc).__name__}: {exc}"


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS DISPLAY
# ══════════════════════════════════════════════════════════════════════════════

# Results are rendered as separate screens selected from the sidebar.
# Switching views only changes the displayed section; pipeline results remain in
# st.session_state and the retrieval pipeline is not rerun.

# ── Screen 2: Search Results ────────────────────────────────────────────────
elif result_view == "Search Results":
    _path_results = st.session_state.get(
        "ai_search_path_results", []
    )

    _zero_paths = [
        (index, path)
        for index, path in enumerate(_path_results)
        if path.get("count", 0) == 0
        or path.get("status") == "error"
    ]

    if _zero_paths:
        st.markdown("### Search paths requiring attention")
        st.caption(
            "Edit and rerun only a search path that returned no records "
            "or encountered an error. The original query is retained."
        )

        for _path_index, _path in _zero_paths:
            with st.expander(
                f"🔎 {_path['name']} — "
                f"{_path.get('status', 'zero_results')}",
                expanded=True,
            ):
                st.write("**Original query**")
                st.code(
                    _path.get("original_expression", ""),
                    language="text",
                )

                if _path.get("error"):
                    st.error(
                        "The last attempt failed: "
                        + _path["error"]
                    )
                elif _path.get("count", 0) == 0:
                    st.warning(
                        "This search path returned zero records."
                    )

                _edited_expression = st.text_area(
                    "Edit Boolean search expression",
                    value=_path.get("expression", ""),
                    key=f"manual_retry_expression_{_path_index}",
                    height=100,
                )

                if st.button(
                    "Rerun this search path",
                    key=f"manual_retry_button_{_path_index}",
                    type="primary",
                ):
                    _edited_expression = (
                        _edited_expression.strip()
                    )

                    if not _edited_expression:
                        st.warning(
                            "Enter a search expression before rerunning."
                        )
                    else:
                        try:
                            from services.uspto_search_service import (
                                search_patents_by_strategy,
                            )
                            from services.dedup_service import deduplicate

                            _previous_expression = _path.get(
                                "expression", ""
                            )
                            _search_limit = _path.get("limit", 100)

                            with st.spinner(
                                f"Rerunning {_path['name']}..."
                            ):
                                _new_results = (
                                    search_patents_by_strategy(
                                        query=_edited_expression,
                                        limit=_search_limit,
                                        offset=0,
                                    )
                                )

                            _new_results = [
                                patent
                                for patent in (_new_results or [])
                                if patent is not None
                            ]

                            _path.setdefault("reruns", []).append(
                                {
                                    "query": _edited_expression,
                                    "previous_query": (
                                        _previous_expression
                                    ),
                                    "count": len(_new_results),
                                    "timestamp": time.strftime(
                                        "%Y-%m-%d %H:%M:%S"
                                    ),
                                    "status": (
                                        "success"
                                        if _new_results
                                        else "zero_results"
                                    ),
                                }
                            )

                            _path["expression"] = _edited_expression
                            _path["count"] = len(_new_results)
                            _path["status"] = (
                                "success"
                                if _new_results
                                else "zero_results"
                            )
                            _path["error"] = None

                            if _new_results:
                                for _patent in _new_results:
                                    _existing_paths = list(
                                        getattr(
                                            _patent,
                                            "search_paths",
                                            [],
                                        )
                                        or []
                                    )
                                    if _path["name"] not in _existing_paths:
                                        _patent.search_paths = (
                                            _existing_paths
                                            + [_path["name"]]
                                        )

                                _combined_raw = (
                                    list(
                                        st.session_state.raw_patents
                                    )
                                    + _new_results
                                )
                                _combined_unique = deduplicate(
                                    _combined_raw
                                )

                                st.session_state.raw_patents = (
                                    _combined_raw
                                )
                                st.session_state.unique_patents = (
                                    _combined_unique
                                )

                                # Rank the refreshed, deduplicated candidate pool.
                                # The previous classification-refined list may be
                                # stale because it was created before this manual retry.
                                _ranking_patents = _combined_unique

                                with st.spinner(
                                    "Updating technical relevance ranking..."
                                ):
                                    st.session_state.ranked = _run_rank(
                                        query,
                                        _ranking_patents,
                                        top_k,
                                    )

                                st.session_state.stage = 3

                            st.session_state.ai_search_path_results = (
                                _path_results
                            )

                            if _new_results:
                                st.success(
                                    f"Retrieved {len(_new_results)} records. "
                                    "The candidate pool and ranking have "
                                    "been refreshed."
                                )
                            else:
                                st.warning(
                                    "The refined query also returned "
                                    "zero records. You can edit it and "
                                    "try again."
                                )

                            st.rerun()

                        except Exception as exc:
                            logger.exception(
                                "Manual search-path rerun failed"
                            )
                            _path["status"] = "error"
                            _path["error"] = (
                                f"{type(exc).__name__}: {exc}"
                            )
                            st.session_state.ai_search_path_results = (
                                _path_results
                            )
                            st.rerun()

                _history = _path.get("reruns", [])
                if _history:
                    st.markdown("**Previous rerun attempts**")
                    for _attempt in _history:
                        st.caption(
                            f"{_attempt['timestamp']} · "
                            f"{_attempt['status']} · "
                            f"{_attempt['count']} records"
                        )
                        st.code(
                            _attempt["query"],
                            language="text",
                        )

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

        _manual_diag = st.session_state.get(
            "manual_retrieval_diagnostics"
        )

        if _manual_diag and _manual_diag.get("queries"):
            st.markdown("### Conventional Retrieval Diagnostics")

            st.caption(
                f"Mode: {_manual_diag['mode']} · "
                f"Queries executed: {_manual_diag['query_count']}"
            )

            st.info(
                "These measurements describe retrieval volume and "
                "query contribution. They do not, by themselves, "
                "establish patent relevance or novelty."
            )

            _diag_rows = [
                {
                    "Query #": item["query_index"],
                    "Query": item["query"],
                    "Raw results": item["raw_count"],
                    "Valid records": item["valid_count"],
                    "New identifiable publications": (
                        item["new_publications"]
                    ),
                    "New families": item["new_families"],
                    "Missing publication IDs": (
                        item["missing_publication_id_count"]
                    ),
                    "Missing family IDs": item["missing_family_id_count"],
                }
                for item in _manual_diag["queries"]
            ]

            st.dataframe(
                pd.DataFrame(_diag_rows),
                width="stretch",
                hide_index=True,
            )

            _diag_col1, _diag_col2 = st.columns(2)

            _diag_col1.metric(
                "Raw records",
                _manual_diag["raw_count"],
            )
            _diag_col2.metric(
                "Deduplicated records",
                _manual_diag["unique_count"],
            )

            _coverage = _manual_diag.get("concept_coverage")

            if _coverage:
                st.markdown("### Critical Concept Coverage")

                st.caption(
                    "Coverage is calculated across deduplicated patent "
                    "records using literal matches in titles, abstracts, "
                    "or claims."
                )

                _coverage_col1, _coverage_col2 = st.columns(2)

                _coverage_col1.metric(
                    "Critical concepts",
                    _coverage["critical_concept_count"],
                )

                _coverage_col2.metric(
                    "Records matching any critical concept",
                    (
                        f'{_coverage["records_with_any_critical_match"]} '
                        f'/ {_coverage["total_records"]}'
                    ),
                )

                if _coverage["per_concept"]:
                    _coverage_rows = [
                        {
                            "Critical concept": item["concept"],
                            "Matched records": item["matched_records"],
                            "Total records": item["total_records"],
                            "Coverage (%)": item["coverage_pct"],
                        }
                        for item in _coverage["per_concept"]
                    ]

                    st.dataframe(
                        pd.DataFrame(_coverage_rows),
                        width="stretch",
                        hide_index=True,
                    )

                _unmatched = _coverage["unmatched_concepts"]

                if _coverage["total_records"] == 0:
                    st.info(
                        "No deduplicated records were retrieved. "
                        "Coverage cannot yet be assessed."
                    )
                elif _unmatched:
                    st.warning(
                        "No literal matches were found for these critical "
                        "concepts: " + ", ".join(_unmatched)
                    )
                else:
                    st.success(
                        "Every critical concept has at least one literal "
                        "match in the retrieved records."
                    )

                st.caption(_coverage["warning"])

            _overlap_rows = _manual_diag.get("query_overlap", [])

            if _overlap_rows:
                st.markdown("### Query Overlap Analysis")
                st.caption(
                    "Pairwise comparison of identifiable patent families. "
                    "Jaccard overlap = shared families / distinct families "
                    "across both queries."
                )

                st.dataframe(
                    pd.DataFrame(_overlap_rows),
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.caption(
                    "Query overlap requires at least two executed queries."
                )

        _path_results = st.session_state.get(
            "ai_search_path_results",
            [],
        )

        if _path_results:
            st.markdown("### Search Structure Performance")

            _path_df = pd.DataFrame(
                _path_results
            )

            _path_df = _path_df[
                [
                    "name",
                    "quality_level",
                    "count",
                    "recall_priority",
                    "precision_priority",
                ]
            ]

            _path_df.columns = [
                "Search Path",
                "Mode",
                "Raw Results",
                "Recall Priority",
                "Precision Priority",
            ]

            st.dataframe(
                _path_df,
                width="stretch",
                hide_index=True,
            )

            core_paths = {
                "Core + Function",
                "Core + Sensor",
                "Core + Function + Sensor",
            }
            core_results = sum(
                result["count"]
                for result in _path_results
                if result.get("name") in core_paths
            )

            if core_results == 0:
                st.warning(
                    "No patents were retrieved from Core-based search paths. "
                    "The current candidate set is therefore dominated by "
                    "sensor/function terminology. Review the Core System "
                    "decomposition before relying on ranking results."
                )

        st.divider()
        st.subheader("Patent Results & Preview")
        st.caption(
            "Select a row to inspect the patent. The retrieval, deduplication, "
            "and ranking results are unchanged."
        )

        _patents_for_preview = list(st.session_state.unique_patents)
        _result_rows = [
            {
                "Patent ID": str(getattr(p, "patent_id", "") or "—"),
                "Family ID": str(getattr(p, "continuity_family_id", "") or "—"),
                "Title": str(getattr(p, "patent_title", "") or "Untitled patent"),
                "Type": str(getattr(p, "patent_type", "") or "—"),
                "Date": str(getattr(p, "patent_date", "") or "—"),
            }
            for p in _patents_for_preview
        ]
        _results_df = pd.DataFrame(_result_rows)

        _table_col, _preview_col = st.columns([3, 2], gap="medium")
        with _table_col:
            st.markdown("#### Retrieved patents")
            _table_event = st.dataframe(
                _results_df,
                width="stretch",
                hide_index=True,
                height=520,
                on_select="rerun",
                selection_mode="single-row",
                key="patent_results_select_table",
                column_config={
                    "Patent ID": st.column_config.TextColumn(width="small"),
                    "Family ID": st.column_config.TextColumn(width="small"),
                    "Title": st.column_config.TextColumn(width="large"),
                    "Type": st.column_config.TextColumn(width="small"),
                    "Date": st.column_config.TextColumn(width="small"),
                },
            )

        _selected_rows = []
        try:
            _selected_rows = list(_table_event.selection.rows)
        except (AttributeError, TypeError):
            _selected_rows = []

        _selected_key = "patent_preview_selected_index"
        if _selected_rows:
            _selected_index = int(_selected_rows[0])
            if 0 <= _selected_index < len(_patents_for_preview):
                st.session_state[_selected_key] = _selected_index
        elif _selected_key not in st.session_state:
            st.session_state[_selected_key] = 0

        _preview_index = st.session_state.get(_selected_key, 0)
        if not isinstance(_preview_index, int) or not (0 <= _preview_index < len(_patents_for_preview)):
            _preview_index = 0
            st.session_state[_selected_key] = 0
        _selected_patent = _patents_for_preview[_preview_index]

        with _preview_col:
            st.markdown("#### Patent Preview")
            _preview_id = str(getattr(_selected_patent, "patent_id", "") or "—")
            _preview_title = str(getattr(_selected_patent, "patent_title", "") or "Untitled patent")
            st.markdown(f"**{_preview_title}**")
            st.caption(f"Publication / Patent ID: {_preview_id}")

            _preview_meta = [
                ("Family ID", getattr(_selected_patent, "continuity_family_id", None)),
                ("Type", getattr(_selected_patent, "patent_type", None)),
                ("Publication date", getattr(_selected_patent, "patent_date", None)),
            ]
            for _label, _value in _preview_meta:
                if _value:
                    st.markdown(f"**{_label}:** { _value }")

            _preview_abstract = str(getattr(_selected_patent, "patent_abstract", "") or "").strip()
            if _preview_abstract:
                st.markdown("##### Abstract")
                st.write(_preview_abstract)
            else:
                st.info("No abstract is available for this record.")

            _preview_claims = getattr(_selected_patent, "patent_claims", None)
            if _preview_claims:
                with st.expander("Claims", expanded=False):
                    if isinstance(_preview_claims, (list, tuple)):
                        st.write("\n\n".join(str(claim) for claim in _preview_claims))
                    else:
                        st.write(str(_preview_claims))

            _preview_url = (
                getattr(_selected_patent, "patent_url", None)
                or getattr(_selected_patent, "url", None)
                or getattr(_selected_patent, "google_patents_url", None)
            )
            if _preview_url and str(_preview_url).startswith(("http://", "https://")):
                st.link_button("Open original patent", str(_preview_url), use_container_width=True)
    else:
        st.info("Run the pipeline to see raw search results here.")

    st.divider()
    st.subheader("Classification Analysis")
    st.caption(
        "Classification frequency across the unique USPTO search results. "
        "Patent IDs are retained for later refinement."
    )

    with st.spinner("⏳ Loading classification analysis, please wait…"):
        classification_data = _get_cached_classification_data(
            st.session_state.unique_patents
        )

    # ── CPC ────────────────────────────────────────────────
    cpc_data = classification_data.get("cpc", {})
    st.markdown("### CPC")

    if cpc_data:
        cpc_rows = []
        for classification, data in sorted(
            cpc_data.items(),
            key=lambda x: x[1]["count"],
            reverse=True,
        ):
            catalogue_entry = get_cpc_catalogue_entry(classification)
            cpc_title = (
                catalogue_entry["title"]
                if catalogue_entry
                else "—"
            )

            cpc_rows.append(
                {
                    "CPC": classification,
                    "CPC title": cpc_title,
                    "Patents": data["count"],
                    "% of results": round(
                        data["count"]
                        / len(st.session_state.unique_patents)
                        * 100,
                        1,
                    ),
                }
            )

        st.dataframe(
            pd.DataFrame(cpc_rows),
            width="stretch",
            hide_index=True,
        )

        st.markdown("#### Review CPC Classifications")

        cpc_options = sorted(
            cpc_data.keys(),
            key=lambda classification: cpc_data[classification]["count"],
            reverse=True,
        )

        selected_cpc = st.multiselect(
            "Select CPC classifications for review",
            options=cpc_options,
            format_func=lambda classification: (
                f"{classification} "
                f"({cpc_data[classification]['count']} patents)"
            ),
            key="selected_cpc_classifications",
        )

        # Display official CPC catalogue information without changing
        # the existing search or patent-selection behavior.
        if selected_cpc:
            st.markdown("#### Official CPC Catalogue Information")

            for classification in selected_cpc:
                entry = get_cpc_catalogue_entry(classification)

                with st.expander(
                    f"{classification} — "
                    f"{entry['title'] if entry else 'Catalogue entry unavailable'}"
                ):
                    if not entry:
                        st.info(
                            "No matching entry was found in the imported "
                            "CPC catalogue. The classification remains "
                            "available for review."
                        )
                        continue

                    st.markdown(f"**Official title:** {entry['title']}")

                    if entry.get("official_definition"):
                        st.markdown("**Official definition**")
                        st.write(entry["official_definition"])
                    elif entry.get("definition_available") is False:
                        st.info(
                            "This CPC classification exists in the official scheme, "
                            "but no separate definition is available in the imported "
                            "definition catalogue."
                        )
                    else:
                        st.info(
                            "No official definition text is available in the "
                            "imported catalogue."
                        )

                    if entry.get("scope_notes"):
                        st.markdown("**Scope notes**")
                        st.write(entry["scope_notes"])

                    if entry.get("special_rules"):
                        st.markdown("**Special rules**")
                        st.write(entry["special_rules"])

                    st.caption(
                        f"Catalogue source version: "
                        f"{entry.get('source_version') or 'Unknown'}"
                    )

        if selected_cpc:
            patent_ids = []

            for classification in selected_cpc:
                for patent_id in cpc_data[classification]["patent_ids"]:
                    if patent_id not in patent_ids:
                        patent_ids.append(patent_id)

            patent_lookup = {
                patent.patent_id: patent
                for patent in st.session_state.unique_patents
            }

            st.markdown(
                f"**{len(patent_ids)} unique patent(s) "
                f"associated with the selected CPC classification(s)**"
            )

            selected_patent_rows = []

            for patent_id in patent_ids:
                patent = patent_lookup.get(patent_id)

                if patent:
                    selected_patent_rows.append(
                        {
                            "Patent ID": patent.patent_id,
                            "Title": patent.patent_title or "—",
                            "Date": patent.patent_date or "—",
                        }
                    )

            if selected_patent_rows:
                st.dataframe(
                    pd.DataFrame(selected_patent_rows),
                    width="stretch",
                    hide_index=True,
                )
    else:
        st.info(
            "No CPC classifications were found in the current unique patent results. "
            "Check whether the search backend returned CPC data for these records."
        )

    # ── USPC ───────────────────────────────────────────────
    uspc_data = classification_data.get("uspc", {})

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### USPC Class")

        uspc_class_data = uspc_data.get("class", {})

        uspc_class_rows = []
        for classification, data in sorted(
            uspc_class_data.items(),
            key=lambda x: x[1]["count"],
            reverse=True,
        ):
            uspc_class_rows.append(
                {
                    "USPC Class": classification,
                    "Patents": data["count"],
                    "% of results": round(
                        data["count"]
                        / len(st.session_state.unique_patents)
                        * 100,
                        1,
                    ),
                }
            )

        st.dataframe(
            pd.DataFrame(uspc_class_rows),
            width="stretch",
            hide_index=True,
        )

    with col2:
        st.markdown("### USPC Subclass")

        uspc_subclass_data = uspc_data.get("subclass", {})

        uspc_subclass_rows = []
        for classification, data in sorted(
            uspc_subclass_data.items(),
            key=lambda x: x[1]["count"],
            reverse=True,
        ):
            uspc_subclass_rows.append(
                {
                    "USPC Subclass": classification,
                    "Patents": data["count"],
                    "% of results": round(
                        data["count"]
                        / len(st.session_state.unique_patents)
                        * 100,
                        1,
                    ),
                }
            )

        st.dataframe(
            pd.DataFrame(uspc_subclass_rows),
            width="stretch",
            hide_index=True,
        )

    # ── USPC Review ───────────────────────────────────────

    st.markdown("#### Review USPC Classifications")

    # -------------------------------------------------------
    # USPC Class
    # -------------------------------------------------------

    uspc_class_options = sorted(
        uspc_class_data.keys(),
        key=lambda classification: uspc_class_data[classification]["count"],
        reverse=True,
    )

    selected_uspc_classes = st.multiselect(
        "Select USPC classes for review",
        options=uspc_class_options,
        format_func=lambda classification: (
            f"{classification} "
            f"({uspc_class_data[classification]['count']} patents)"
        ),
        key="selected_uspc_classes",
    )

    if selected_uspc_classes:
        class_patent_ids = []

        for classification in selected_uspc_classes:
            for patent_id in uspc_class_data[classification]["patent_ids"]:
                if patent_id not in class_patent_ids:
                    class_patent_ids.append(patent_id)

        patent_lookup = {
            patent.patent_id: patent
            for patent in st.session_state.unique_patents
        }

        st.markdown(
            f"**{len(class_patent_ids)} unique patent(s) "
            f"associated with the selected USPC class(es)**"
        )

        class_rows = []

        for patent_id in class_patent_ids:
            patent = patent_lookup.get(patent_id)

            if patent:
                class_rows.append(
                    {
                        "Patent ID": patent.patent_id,
                        "Title": patent.patent_title or "—",
                        "Date": patent.patent_date or "—",
                    }
                )

        if class_rows:
            st.dataframe(
                pd.DataFrame(class_rows),
                width="stretch",
                hide_index=True,
            )

    # -------------------------------------------------------
    # USPC Subclass
    # -------------------------------------------------------

    uspc_subclass_options = sorted(
        uspc_subclass_data.keys(),
        key=lambda classification: (
            uspc_subclass_data[classification]["count"]
        ),
        reverse=True,
    )

    selected_uspc_subclasses = st.multiselect(
        "Select USPC subclasses for review",
        options=uspc_subclass_options,
        format_func=lambda classification: (
            f"{classification} "
            f"({uspc_subclass_data[classification]['count']} patents)"
        ),
        key="selected_uspc_subclasses",
    )

    if selected_uspc_subclasses:
        subclass_patent_ids = []

        for classification in selected_uspc_subclasses:
            for patent_id in uspc_subclass_data[classification]["patent_ids"]:
                if patent_id not in subclass_patent_ids:
                    subclass_patent_ids.append(patent_id)

        patent_lookup = {
            patent.patent_id: patent
            for patent in st.session_state.unique_patents
        }

        st.markdown(
            f"**{len(subclass_patent_ids)} unique patent(s) "
            f"associated with the selected USPC subclass(es)**"
        )

        subclass_rows = []

        for patent_id in subclass_patent_ids:
            patent = patent_lookup.get(patent_id)

            if patent:
                subclass_rows.append(
                    {
                        "Patent ID": patent.patent_id,
                        "Title": patent.patent_title or "—",
                        "Date": patent.patent_date or "—",
                    }
                )

        if subclass_rows:
            st.dataframe(
                pd.DataFrame(subclass_rows),
                width="stretch",
                hide_index=True,
            )

    # ── AI-Assisted Classification Suggestions ─────────────

    st.divider()
    st.subheader("🤖 AI-Assisted Classification Suggestions")
    st.caption(
        "Claude can suggest potentially relevant CPC and USPC classifications "
        "from the approved search concepts. Review the suggestions and the "
        "authoritative USPTO definitions before selecting any classification. "
        "Suggestions are not automatically applied."
    )

    _ai_concepts = []

    for _index in range(
        st.session_state.get("strategy_concept_count", 0)
    ):
        _name = st.session_state.get(
            f"strategy_concept_name_{_index}",
            "",
        ).strip()

        _terms_text = st.session_state.get(
            f"strategy_concept_terms_{_index}",
            "",
        )

        _terms = [
            term.strip()
            for term in _terms_text.splitlines()
            if term.strip()
        ]

        if _name or _terms:
            _ai_concepts.append(
                {
                    "name": _name,
                    "terms": _terms,
                }
            )

    if not _ai_concepts:
        st.info(
            "Approve the search concepts first to generate "
            "AI classification suggestions."
        )
    else:
        if st.button(
            "🤖 Generate Classification Suggestions",
            key="generate_ai_classifications",
        ):
            try:
                from models.claude_client import ClaudeClient

                with st.spinner(
                    "Claude is analyzing the approved concepts..."
                ):
                    _suggestions = generate_classification_suggestions(
                        _ai_concepts,
                        ClaudeClient(),
                    )

                st.session_state.ai_classification_suggestions = _suggestions

            except Exception as exc:
                st.error(
                    f"Unable to generate classification suggestions: {exc}"
                )

    _ai_classifications = st.session_state.get(
        "ai_classification_suggestions"
    )

    if _ai_classifications:
        st.markdown("### Review AI Suggestions")

        st.caption(
            "The definitions below are retrieved from USPTO classification "
            "sources. They are provided for review; selecting a classification "
            "is still a user decision."
        )

        _ai_cpc = _ai_classifications.get("cpc", [])
        _ai_uspc = _ai_classifications.get("uspc", [])

        if _ai_cpc:
            st.markdown("#### Suggested CPC Classifications")

            for _index, _item in enumerate(_ai_cpc):
                _classification = (
                    _item.get("classification", "").strip()
                )

                _concept = _item.get("concept", "").strip()
                _reason = _item.get("reason", "").strip()

                if not _classification:
                    continue

                _definition = get_cpc_definition(_classification)

                _normalized_cpc = (
                    _classification.replace(" ", "").upper()
                )

                _result_count = sum(
                    1
                    for _patent in st.session_state.unique_patents
                    if any(
                        str(_cpc).replace(" ", "").upper()
                        == _normalized_cpc
                        for _cpc in _patent.cpc_classifications
                    )
                )

                _checkbox_key = (
                    f"ai_cpc_selected_{_index}_{_classification}"
                )

                st.markdown(
                    f"**{_classification}**"
                    f" — {_concept or 'Related technical concept'}"
                )

                col1, col2 = st.columns([3, 1])

                with col1:
                    if _reason:
                        st.write(
                            f"**AI rationale:** {_reason}"
                        )

                    if _definition:
                        with st.expander(
                            "View USPTO definition",
                            expanded=False,
                        ):
                            st.write(_definition)
                    else:
                        st.warning(
                            "USPTO definition could not be retrieved."
                        )

                with col2:
                    st.metric(
                        "Current results",
                        _result_count,
                    )

                    st.checkbox(
                        "Select",
                        key=_checkbox_key,
                    )

                st.divider()

        if _ai_uspc:
            st.markdown("#### Suggested USPC Classifications")

            for _index, _item in enumerate(_ai_uspc):
                _uspc_class = (
                    _item.get("class", "").strip()
                )

                _uspc_subclass = (
                    _item.get("subclass", "").strip()
                )

                _concept = _item.get("concept", "").strip()
                _reason = _item.get("reason", "").strip()

                if not _uspc_class:
                    continue

                _uspc_label = _uspc_class

                if _uspc_subclass:
                    _uspc_label = (
                        f"{_uspc_class}/{_uspc_subclass}"
                    )

                _definition = get_uspc_definition(
                    _uspc_class,
                    _uspc_subclass,
                )

                _normalized_uspc_class = str(
                    _uspc_class
                ).strip().upper()

                _normalized_uspc_subclass = str(
                    _uspc_subclass
                ).strip().upper()

                if _uspc_subclass:
                    _result_count = sum(
                        1
                        for _patent in st.session_state.unique_patents
                        if (
                            str(_patent.uspc_class).strip().upper()
                            == _normalized_uspc_class
                            and
                            str(_patent.uspc_subclass).strip().upper()
                            == _normalized_uspc_subclass
                        )
                    )
                else:
                    _result_count = sum(
                        1
                        for _patent in st.session_state.unique_patents
                        if (
                            str(_patent.uspc_class).strip().upper()
                            == _normalized_uspc_class
                        )
                    )

                _checkbox_key = (
                    f"ai_uspc_selected_{_index}_{_uspc_label}"
                )

                st.markdown(
                    f"**USPC {_uspc_label}**"
                    f" — {_concept or 'Related technical concept'}"
                )

                col1, col2 = st.columns([3, 1])

                with col1:
                    if _reason:
                        st.write(
                            f"**AI rationale:** {_reason}"
                        )

                    if _definition:
                        with st.expander(
                            "View USPTO definition",
                            expanded=False,
                        ):
                            st.write(_definition)
                    else:
                        st.warning(
                            "USPTO definition could not be retrieved."
                        )

                with col2:
                    st.metric(
                        "Current results",
                        _result_count,
                    )

                    st.checkbox(
                        "Select",
                        key=_checkbox_key,
                    )

                st.divider()

        if st.button(
            "✅ Apply Reviewed AI Classifications",
            key="apply_reviewed_ai_classifications",
        ):
            _selected_ai_cpc = []

            for _index, _item in enumerate(_ai_cpc):
                _classification = (
                    _item.get("classification", "").strip()
                )

                if not _classification:
                    continue

                _checkbox_key = (
                    f"ai_cpc_selected_{_index}_{_classification}"
                )

                if st.session_state.get(_checkbox_key, False):
                    _selected_ai_cpc.append(_classification)

            _selected_ai_uspc_classes = []
            _selected_ai_uspc_subclasses = []

            for _index, _item in enumerate(_ai_uspc):
                _uspc_class = (
                    _item.get("class", "").strip()
                )

                _uspc_subclass = (
                    _item.get("subclass", "").strip()
                )

                if not _uspc_class:
                    continue

                _uspc_label = _uspc_class

                if _uspc_subclass:
                    _uspc_label = (
                        f"{_uspc_class}/{_uspc_subclass}"
                    )

                _checkbox_key = (
                    f"ai_uspc_selected_{_index}_{_uspc_label}"
                )

                if st.session_state.get(_checkbox_key, False):
                    if _uspc_class not in _selected_ai_uspc_classes:
                        _selected_ai_uspc_classes.append(
                            _uspc_class
                        )

                    if _uspc_subclass:
                        if (
                            _uspc_subclass
                            not in _selected_ai_uspc_subclasses
                        ):
                            _selected_ai_uspc_subclasses.append(
                                _uspc_subclass
                            )

            st.session_state.ai_pending_cpc_classifications = (
                _selected_ai_cpc
            )

            st.session_state.ai_pending_uspc_classes = (
                _selected_ai_uspc_classes
            )

            st.session_state.ai_pending_uspc_subclasses = (
                _selected_ai_uspc_subclasses
            )

            st.success(
                "Reviewed AI classifications are ready for "
                "Classification Refinement. Review them there "
                "before applying the refinement."
            )

    # ── Classification Refinement ─────────────────────────

    st.divider()
    st.subheader("Classification Refinement")
    st.caption(
        "**Review the selected classifications before applying local "
        "refinement to the current USPTO results.**"
    )

    refinement_mode = st.radio(
        "Refinement mode",
        options=[
            "Classification only",
            "Original search + classification",
        ],
        horizontal=True,
        key="classification_refinement_mode",
    )

    classification_operator = st.radio(
        "Operator between selected classification groups",
        options=["AND", "OR"],
        horizontal=True,
        key="classification_refinement_operator",
    )

    selected_cpc = st.session_state.get(
        "selected_cpc_classifications",
        [],
    )

    selected_uspc_classes = st.session_state.get(
        "selected_uspc_classes",
        [],
    )

    selected_uspc_subclasses = st.session_state.get(
        "selected_uspc_subclasses",
        [],
    )

    selected_classifications = []

    # -------------------------------------------------------
    # CPC selections
    # -------------------------------------------------------

    for classification in selected_cpc:
        normalized_cpc = classification.replace(" ", "")
        selected_classifications.append(
            {
                "type": "CPC",
                "value": classification,
                "query_value": f"{normalized_cpc}.CPC.",
            }
        )

    # -------------------------------------------------------
    # USPC class selections
    # -------------------------------------------------------

    for classification in selected_uspc_classes:
        selected_classifications.append(
            {
                "type": "USPC Class",
                "value": classification,
                "query_value": f'"{classification}".CLAS.',
            }
        )

    # -------------------------------------------------------
    # USPC subclass selections
    # -------------------------------------------------------
    #
    # A standalone subclass does not identify a complete
    # USPC class/subclass pair, so we retain the selection
    # for review but do not construct a live query from it.
    # -------------------------------------------------------

    for classification in selected_uspc_subclasses:
        selected_classifications.append(
            {
                "type": "USPC Subclass",
                "value": classification,
                "query_value": None,
            }
        )

    if selected_classifications:
        st.markdown("#### Selected classifications")

        refinement_rows = []

        for item in selected_classifications:
            refinement_rows.append(
                {
                    "Type": item["type"],
                    "Classification": item["value"],
                    "Search expression": (
                        item["query_value"]
                        if item["query_value"]
                        else "Requires class/subclass pair"
                    ),
                }
            )

        st.dataframe(
            pd.DataFrame(refinement_rows),
            width="stretch",
            hide_index=True,
        )

        # ---------------------------------------------------
        # Classification query preview
        # ---------------------------------------------------

        cpc_query_values = [
            item["query_value"]
            for item in selected_classifications
            if item["type"] == "CPC"
        ]

        uspc_class_query_values = [
            item["query_value"]
            for item in selected_classifications
            if item["type"] == "USPC Class"
        ]

        preview_groups = []

        if cpc_query_values:
            if len(cpc_query_values) == 1:
                cpc_expression = cpc_query_values[0]
            else:
                cpc_expression = (
                    "("
                    + " OR ".join(cpc_query_values)
                    + ")"
                )

            preview_groups.append(cpc_expression)

        if uspc_class_query_values:
            if len(uspc_class_query_values) == 1:
                uspc_expression = uspc_class_query_values[0]
            else:
                uspc_expression = (
                    "("
                    + " OR ".join(uspc_class_query_values)
                    + ")"
                )

            preview_groups.append(uspc_expression)

        if preview_groups:
            classification_query = (
                f" {classification_operator} ".join(
                    preview_groups
                )
            )

            st.markdown("#### Classification query representation")

            st.code(
                classification_query,
                language="text",
            )

        # ---------------------------------------------------
        # Apply local classification refinement
        # ---------------------------------------------------

        st.markdown("#### Apply Classification Refinement")

        st.caption(
            "Classification only filters the current USPTO results locally. "
            "Original search + classification reruns the original USPTO search "
            "with the selected classification criteria."
        )

        if st.button(
            "Apply Classification Refinement",
            type="primary",
            key="apply_classification_refinement",
        ):
            current_patents = st.session_state.get(
                "unique_patents",
                [],
            )

            # -------------------------------------------------------
            # Validate classification selection
            # -------------------------------------------------------

            if not selected_cpc and not selected_uspc_classes:
                st.warning(
                    "Select at least one CPC classification or USPC class "
                    "before applying classification refinement."
                )

            # -------------------------------------------------------
            # Mode 1 — Classification only
            # -------------------------------------------------------

            elif refinement_mode == "Classification only":

                if not current_patents:
                    st.warning(
                        "No patent results are available for refinement."
                    )

                else:
                    refined_patents = filter_patents_by_classification(
                        current_patents,
                        selected_cpc=selected_cpc,
                        selected_uspc_class=selected_uspc_classes,
                        selected_uspc_subclass=selected_uspc_subclasses,
                        operator=classification_operator,
                    )

                    st.session_state.classification_refined_patents = (
                        refined_patents
                    )

                    st.session_state.classification_refinement_applied = True

                    st.success(
                        f"Classification refinement: "
                        f"{len(current_patents)} → "
                        f"{len(refined_patents)} patent(s)"
                    )

                    if refined_patents:
                        with st.spinner(
                            f"Ranking {len(refined_patents)} "
                            "classification-refined patents…"
                        ):
                            ranked = _timed(
                                "3. Embed + rank",
                                _run_rank,
                                query,
                                refined_patents,
                                top_k,
                            )

                        st.session_state.ranked = ranked
                        st.session_state.stage = 3

                        st.success(
                            f"Ranking complete: "
                            f"{len(ranked)} top-ranked patent(s) "
                            f"from {len(refined_patents)} refined patent(s)."
                        )

            # -------------------------------------------------------
            # Mode 2 — Original search + classification
            # -------------------------------------------------------

            else:

                if settings.search_backend.lower() != "uspto":
                    st.warning(
                        "Original search + classification is currently "
                        "implemented for the USPTO ODP backend."
                    )

                elif not query.strip():
                    st.warning(
                        "The original search query is empty."
                    )

                else:
                    from services.search_service import fetch_patents_with_fallback
                    from services.dedup_service import deduplicate

                    tokens = (
                        st.session_state
                        .get("expansion_metadata", {})
                        .get("tokens")
                    )

                    validated_queries = (
                        st.session_state
                        .get("expansion_metadata", {})
                        .get("validated_queries", [])
                    )

                    if tokens is None:
                        st.warning(
                            "Search token information is unavailable. "
                            "Please run the original search again."
                        )
                        st.stop()

                    with st.spinner(
                        "Rerunning the original USPTO search "
                        "with classification criteria…"
                    ):
                        raw_classified = _timed(
                            "2. Original search + classification",
                            fetch_patents_with_fallback,
                            tokens,
                            validated_queries,
                            15,
                            cpc_classifications=selected_cpc,
                            uspc_classes=selected_uspc_classes,
                            classification_operator=classification_operator,
                        )

                    unique_classified = deduplicate(raw_classified)

                    # Store the new USPTO result set as the active result set.
                    st.session_state.raw_patents = raw_classified
                    st.session_state.unique_patents = unique_classified

                    st.session_state.classification_refined_patents = (
                        unique_classified
                    )

                    st.session_state.classification_refinement_applied = True

                    st.success(
                        f"Original search + classification: "
                        f"{len(unique_classified)} unique patent(s) returned."
                    )

                    if unique_classified:
                        with st.spinner(
                            f"Ranking {len(unique_classified)} "
                            "classification-filtered USPTO patents…"
                        ):
                            ranked = _timed(
                                "3. Embed + rank",
                                _run_rank,
                                query,
                                unique_classified,
                                top_k,
                            )

                        st.session_state.ranked = ranked
                        st.session_state.stage = 3

                        st.success(
                            f"Ranking complete: "
                            f"{len(ranked)} top-ranked patent(s) "
                            f"from {len(unique_classified)} USPTO patent(s)."
                        )

        # ---------------------------------------------------
        # Show refined results
        # ---------------------------------------------------

        if st.session_state.get(
            "classification_refinement_applied",
            False,
        ):
            refined_patents = st.session_state.get(
                "classification_refined_patents",
                [],
            )

            st.markdown("#### Classification-refined results")

            st.write(
                f"**{len(refined_patents)}** patent(s) match "
                "the selected classifications."
            )

            if refined_patents:
                refined_rows = []

                for patent in refined_patents:
                    refined_rows.append(
                        {
                            "Patent ID": patent.patent_id,
                            "Title": patent.patent_title or "—",
                            "Date": patent.patent_date or "—",
                            "USPC": patent.uspc_class or "—",
                            "USPC Subclass": (
                                patent.uspc_subclass or "—"
                            ),
                            "CPC": ", ".join(
                                patent.cpc_classifications
                            ),
                        }
                    )

                st.dataframe(
                    pd.DataFrame(refined_rows),
                    width="stretch",
                    hide_index=True,
                )

                # Second-stage refinement: choose a CPC found
                # in the classification-refined result set.
                st.markdown("#### Refine by a specific CPC")

                available_cpc = sorted({
                    "".join(str(code).split())
                    for patent in refined_patents
                    for code in (patent.cpc_classifications or [])
                    if str(code).strip()
                })

                if available_cpc:
                    selected_result_cpc = st.selectbox(
                        "Choose a CPC classification",
                        options=available_cpc,
                        key="selected_result_cpc",
                        help=(
                            "Choose a CPC present in the current "
                            "classification-refined results."
                        ),
                    )

                    if st.button(
                        "Search original query with this CPC",
                        key="apply_selected_result_cpc",
                        type="primary",
                    ):
                        from services.search_service import (
                            fetch_patents_with_fallback,
                        )
                        from services.dedup_service import deduplicate

                        expansion_metadata = st.session_state.get(
                            "expansion_metadata", {}
                        )
                        tokens = expansion_metadata.get("tokens")
                        validated_queries = expansion_metadata.get(
                            "validated_queries", []
                        )

                        if settings.search_backend.lower() != "uspto":
                            st.warning(
                                "This refinement currently requires "
                                "the USPTO ODP backend."
                            )
                        elif tokens is None:
                            st.warning(
                                "Search token information is unavailable. "
                                "Please run the original search again."
                            )
                        else:
                            with st.spinner(
                                f"Searching with CPC {selected_result_cpc}…"
                            ):
                                raw_results = fetch_patents_with_fallback(
                                    tokens,
                                    validated_queries,
                                    15,
                                    cpc_classifications=[
                                        selected_result_cpc
                                    ],
                                    uspc_classes=[],
                                    classification_operator="AND",
                                )

                            refined_results = deduplicate(raw_results)

                            st.session_state.raw_patents = raw_results
                            st.session_state.unique_patents = refined_results
                            st.session_state.classification_refined_patents = (
                                refined_results
                            )
                            st.session_state.classification_refinement_applied = (
                                True
                            )

                            st.success(
                                f"CPC {selected_result_cpc}: "
                                f"{len(refined_results)} unique patent(s)."
                            )

                            if refined_results:
                                with st.spinner(
                                    "Ranking CPC-refined results…"
                                ):
                                    ranked = _timed(
                                        "3. Embed + rank",
                                        _run_rank,
                                        query,
                                        refined_results,
                                        top_k,
                                    )

                                st.session_state.ranked = ranked
                                st.session_state.stage = 3

                                st.dataframe(
                                    pd.DataFrame([
                                        {
                                            "Patent ID": p.patent_id,
                                            "Title": p.patent_title or "—",
                                            "CPC": ", ".join(
                                                p.cpc_classifications or []
                                            ),
                                        }
                                        for p in refined_results
                                    ]),
                                    width="stretch",
                                    hide_index=True,
                                )
                else:
                    st.info(
                        "No CPC classifications are present in these "
                        "results. Try another refinement or search."
                    )
            else:
                st.warning(
                    "No patents matched the selected classifications."
                )

    else:
        st.info(
            "Select CPC or USPC classifications above to build "
            "a classification refinement."
        )

    # ── IPC ────────────────────────────────────────────────
    ipc_data = classification_data.get("ipc", {})

    st.markdown("### IPC")

    if ipc_data:
        ipc_rows = []
        for classification, data in sorted(
            ipc_data.items(),
            key=lambda x: x[1]["count"],
            reverse=True,
        ):
            ipc_rows.append(
                {
                    "IPC": classification,
                    "Patents": data["count"],
                    "% of results": round(
                        data["count"]
                        / len(st.session_state.unique_patents)
                        * 100,
                        1,
                    ),
                }
            )

        st.dataframe(
            pd.DataFrame(ipc_rows),
            width="stretch",
            hide_index=True,
        )
    else:
        st.info(
            "No IPC classifications were returned by the current "
            "USPTO ODP application-search response."
        )

# ── Screen 3: Technical Relevance ───────────────────────────────────────────
elif result_view == "Technical Relevance":
    if st.session_state.ranked:
        ranking_source = st.session_state.get(
            "classification_refined_patents",
            st.session_state.unique_patents,
        )

        if not st.session_state.get(
            "classification_refinement_applied",
            False,
        ):
            ranking_source = st.session_state.unique_patents

        total_unique = len(ranking_source)
        shown        = len(st.session_state.ranked)
        filtered_out = total_unique - shown
        st.success(
            f"Top {shown} of {total_unique} unique patent(s) — "
            "hybrid relevance ranking applied"
            + (f" · {filtered_out} not selected" if filtered_out > 0 else "")
        )

        rows = []
        strategy = st.session_state.get("search_strategy")
        critical_names = {
            concept.name
            for concept in (getattr(strategy, "concepts", []) or [])
            if str(getattr(concept, "importance", "")).strip().lower() == "critical"
        }

        for i, rp in enumerate(st.session_state.ranked, 1):
            evaluated_critical = critical_names.intersection(rp.importance_hits)
            matched_critical = sum(
                bool(rp.importance_hits[name])
                for name in evaluated_critical
            )
            critical_match_summary = (
                f"{matched_critical}/{len(evaluated_critical)}"
                if evaluated_critical else "Not evaluated"
            )
            evidence_status = (
                "Insufficient text"
                if rp.text_status == "insufficient_text"
                else "Text available"
            )
            rows.append({
                "Patent ID": str(getattr(rp.patent, "patent_id", "") or "—"),
                "Title": str(getattr(rp.patent, "patent_title", "") or "Untitled patent"),
                "Type": getattr(rp.patent, "patent_type", None) or "—",
                "Date": getattr(rp.patent, "patent_date", None) or "—",
                "Hybrid score": round(rp.hybrid_score, 4) if rp.hybrid_score is not None else None,
                "BM25 score": round(rp.bm25_score, 4) if rp.bm25_score is not None else None,
                "Cosine score": round(rp.cosine_score, 4) if rp.cosine_score is not None else None,
            })

        relevance_df = pd.DataFrame(rows)
        table_col, preview_col = st.columns([3, 2], gap="medium")
        with table_col:
            st.subheader("Ranked patents")
            st.caption("Select a row to inspect its technical relevance and evidence.")
            table_event = st.dataframe(
                relevance_df,
                width="stretch",
                hide_index=True,
                height=560,
                on_select="rerun",
                selection_mode="single-row",
                key="technical_relevance_select_table",
                column_config={
                    "Patent ID": st.column_config.TextColumn(width="medium"),
                    "Title": st.column_config.TextColumn(width="large"),
                    "Hybrid score": st.column_config.NumberColumn(format="%.4f", width="small"),
                    "BM25 score": st.column_config.NumberColumn(format="%.4f", width="small"),
                    "Cosine score": st.column_config.NumberColumn(format="%.4f", width="small"),
                },
            )

        selected_rows = []
        try:
            selected_rows = list(table_event.selection.rows)
        except (AttributeError, TypeError):
            selected_rows = []

        selected_key = "technical_relevance_selected_index"
        if selected_rows:
            selected_index = int(selected_rows[0])
            if 0 <= selected_index < len(st.session_state.ranked):
                st.session_state[selected_key] = selected_index
        elif selected_key not in st.session_state:
            st.session_state[selected_key] = 0

        selected_index = st.session_state.get(selected_key, 0)
        if not isinstance(selected_index, int) or not (0 <= selected_index < len(st.session_state.ranked)):
            selected_index = 0
            st.session_state[selected_key] = 0

        selected_ranked = st.session_state.ranked[selected_index]
        patent = selected_ranked.patent
        with preview_col:
            st.subheader("Technical Relevance Preview")
            st.markdown(f"**{getattr(patent, 'patent_title', None) or 'Untitled patent'}**")
            st.caption(f"Patent ID: {getattr(patent, 'patent_id', None) or '—'}")

            metadata = [
                ("Type", getattr(patent, "patent_type", None)),
                ("Publication date", getattr(patent, "patent_date", None)),
                ("Text status", "Insufficient text" if selected_ranked.text_status == "insufficient_text" else "Text available"),
            ]
            for label, value in metadata:
                if value:
                    st.markdown(f"**{label}:** {value}")

            def format_score(value):
                return f"{value:.4f}" if value is not None else "N/A"

            st.markdown("##### Relevance scores")
            st.markdown(
                f"**Hybrid:** `{format_score(selected_ranked.hybrid_score)}`  \n"
                f"**Importance:** `{format_score(selected_ranked.importance_score)}`  \n"
                f"**Cosine:** `{format_score(selected_ranked.cosine_score)}`  \n"
                f"**BM25:** `{format_score(selected_ranked.bm25_score)}`"
            )

            if selected_ranked.importance_hits:
                matched_concepts = [
                    name for name, hit in selected_ranked.importance_hits.items() if hit
                ]
                unmatched_concepts = [
                    name for name, hit in selected_ranked.importance_hits.items() if not hit
                ]
                st.markdown("##### Search strategy evidence")
                st.write(
                    f"**Matched concepts:** {len(matched_concepts)} of "
                    f"{len(selected_ranked.importance_hits)}"
                )
                with st.expander("Matched concepts", expanded=True):
                    if matched_concepts:
                        for name in matched_concepts:
                            st.write(f"✅ {name}")
                    else:
                        st.write("No concepts matched.")
                with st.expander("Unmatched concepts"):
                    if unmatched_concepts:
                        for name in unmatched_concepts:
                            st.write(f"— {name}")
                    else:
                        st.write("All evaluated concepts matched.")

            json_groups = st.session_state.get("json_groups")
            if json_groups:
                from services.json_query_service import match_groups_in_text
                matches = match_groups_in_text(
                    f"{getattr(patent, 'patent_title', '')} "
                    f"{getattr(patent, 'patent_abstract', '') or ''}",
                    json_groups,
                )
                if matches:
                    badges = " ".join(
                        _group_badge_html(idx, term) for idx, term in matches
                    )
                    st.markdown(f"**Matched groups:** {badges}", unsafe_allow_html=True)

            abstract = str(getattr(patent, "patent_abstract", "") or "").strip()
            if abstract:
                with st.expander("Abstract", expanded=True):
                    st.write(abstract)
            else:
                st.info("No abstract is available for this record.")

            claims = getattr(patent, "patent_claims", None)
            if claims:
                with st.expander("Claims", expanded=False):
                    if isinstance(claims, (list, tuple)):
                        st.write("\n\n".join(str(claim) for claim in claims))
                    else:
                        st.write(str(claims))

            patent_url = (
                getattr(patent, "patent_url", None)
                or getattr(patent, "url", None)
                or getattr(patent, "google_patents_url", None)
            )
            if patent_url and str(patent_url).startswith(("http://", "https://")):
                st.link_button("Open original patent", str(patent_url), use_container_width=True)

    else:
        st.info("Run the pipeline to see ranked patents here.")

# ── Screen 4: Claude Technical Analysis ──────────────────────────────────────
elif result_view == "Claude Technical Analysis":
    if st.session_state.stage < 3:
        st.info("Complete stages 1–3 first, then send the ranked results to Claude.")
    elif st.session_state.stage == 3:
        st.success(f"Top {len(st.session_state.ranked)} ranked patents are ready for technical comparison.")
        if st.button("🤖 Send Top 20 to Claude", type="primary", key="send_top20_to_claude_screen", width="stretch"):
            try:
                st.session_state.comparison = ""
                st.session_state.report = ""
                with st.spinner("Calling Claude for technical comparison…"):
                    comparison = _timed(
                        "4. Claude comparison",
                        _run_claude,
                        query,
                        st.session_state.ranked,
                    )
                st.session_state.comparison = comparison
                st.session_state.stage = 4
                st.rerun()
            except Exception as exc:
                logger.exception("Unhandled exception during Claude comparison from analysis screen")
                st.session_state.error = f"{type(exc).__name__}: {exc}"
                st.error(st.session_state.error)
    else:
        sub1, sub2 = st.tabs(["Technical Comparison JSON", "Report"])

        with sub1:
            st.subheader("Structured technical comparison")
            try:
                parsed = json.loads(st.session_state.comparison)
                comp_df = pd.DataFrame(parsed)
                st.dataframe(comp_df, width="stretch", hide_index=True)
            except (json.JSONDecodeError, Exception):
                st.text_area("Raw comparison output", st.session_state.comparison,
                             height=400)

        with sub2:
            st.subheader("Patent intelligence report")

            if not st.session_state.comparison:
                st.info("Generate the technical comparison first.")
            else:
                if st.button("📝 Generate Report", type="primary"):
                    try:
                        from models.claude_client import ClaudeClient
                        from services.report_service import generate

                        with st.spinner("Generating patent intelligence report…"):
                            report = _timed(
                                "5. Claude report",
                                generate,
                                query,
                                st.session_state.ranked,
                                st.session_state.comparison,
                                ClaudeClient(),
                            )

                        st.session_state.report = report

                    except Exception as exc:
                        logger.exception("Unhandled exception generating report")
                        st.session_state.error = (
                            f"{type(exc).__name__}: {exc}"
                        )

                if st.session_state.report:
                    st.markdown(st.session_state.report)

                    st.divider()
                    st.download_button(
                        "⬇ Download report (.md)",
                        data=st.session_state.report.encode(),
                        file_name="patent_report.md",
                        mime="text/markdown",
                    )
                else:
                    st.caption("The report has not been generated yet.")
