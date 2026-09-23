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
from models.claude_client import ClaudeClient
from models.schemas import SearchStrategy, SearchConcept, ProximityRule
from services.ai_classification_service import (
    generate_classification_suggestions,
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
    build_uspto_search_strings,
    build_uspto_proximity_strings,
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
    for k, v in _DEFAULTS.items():
        st.session_state[k] = v


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

# ── Phase 1 — Manual Search Strategy ──────────────────────────────────────────
st.divider()
st.subheader("🧭 Phase 1 — Manual Search Strategy")
st.caption(
    "Define the important concepts and alternative patent terminology before "
    "running the search. This strategy will later also be used by document-driven search."
)

# ── AI-assisted concept generation ────────────────────────────────────────────
st.markdown("### 🤖 AI-Assisted Search Concepts")
st.caption(
    "Claude can propose 10 technical concepts and 3 patent-search keywords "
    "for each. Review and edit the suggestions before using them in the search strategy."
)

if "ai_search_concepts" not in st.session_state:
    st.session_state.ai_search_concepts = []

if "ai_search_concepts_approved" not in st.session_state:
    st.session_state.ai_search_concepts_approved = False

_ai_invention = st.text_area(
    "Invention description for Claude",
    value=st.session_state.get("query", ""),
    height=100,
    key="ai_invention_description",
    help="Describe the invention or technology you want to search for.",
)

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
            st.success(
                f"Generated {len(generated)} concepts. "
                "Review and edit them below."
            )

        except Exception as exc:
            st.error(f"Unable to generate search concepts: {exc}")

_ai_concepts = st.session_state.get("ai_search_concepts", [])

if _ai_concepts:
    st.markdown("#### Review and edit Claude's suggestions")

    edited_concepts = []

    for index, concept in enumerate(_ai_concepts, start=1):
        st.markdown(f"**Concept {index}**")

        concept_name = st.text_input(
            "Concept name",
            value=concept.get("name", ""),
            key=f"ai_concept_name_{index}",
        )

        keywords = concept.get("keywords", [])
        edited_keywords = []

        for keyword_index in range(3):
            default_keyword = (
                keywords[keyword_index]
                if keyword_index < len(keywords)
                else ""
            )

            edited_keyword = st.text_input(
                f"Keyword {keyword_index + 1}",
                value=default_keyword,
                key=f"ai_concept_{index}_keyword_{keyword_index + 1}",
            )

            edited_keywords.append(edited_keyword)

        edited_concepts.append(
            {
                "name": concept_name,
                "keywords": edited_keywords,
            }
        )

        if index < len(_ai_concepts):
            st.divider()

    st.session_state.ai_search_concepts = edited_concepts

    if st.button(
        "✅ Approve Reviewed Concepts",
        key="approve_ai_search_concepts",
    ):
        st.session_state.ai_search_concepts_approved = True

        # Populate the existing manual Search Strategy fields.
        st.session_state.strategy_concept_count = len(edited_concepts)

        for _index, _concept in enumerate(edited_concepts):
            st.session_state[f"strategy_concept_name_{_index}"] = (
                _concept.get("name", "").strip()
            )

            st.session_state[f"strategy_concept_terms_{_index}"] = "\n".join(
                keyword.strip()
                for keyword in _concept.get("keywords", [])
                if keyword.strip()
            )

            # Let the user decide importance in the existing UI.
            st.session_state[f"strategy_concept_importance_{_index}"] = "important"

            # Claude's three keywords are alternatives by default.
            st.session_state[f"strategy_concept_operator_{_index}"] = "OR"

        st.success(
            "Reviewed concepts approved and loaded into the Search Strategy."
        )

    if st.session_state.get("ai_search_concepts_approved", False):
        st.info(
            "The reviewed concepts are approved. "
            "No patent search has been executed yet."
        )

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

            st.session_state.ai_invention_structure = _structure
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

with st.expander("Build Search Strategy", expanded=True):

    st.markdown("### Key Inventive Points")

    _kip_text = st.text_area(
        "Enter the key inventive points, one per line",
        value="\n".join(
            st.session_state.search_strategy.key_inventive_points
            if st.session_state.search_strategy
            else []
        ),
        height=100,
        key="strategy_kip",
        placeholder=(
            "Example:\n"
            "rechargeable battery\n"
            "charging circuit"
        ),
    )

    st.markdown("### Concepts")

    _concept_count = st.number_input(
        "Number of concepts",
        min_value=1,
        max_value=15,
        value=3,
        step=1,
        key="strategy_concept_count",
    )

    _concepts = []

    for _i in range(_concept_count):
        st.markdown(f"#### Concept {_i + 1}")

        _c1, _c2 = st.columns([3, 1])

        with _c1:
            _name = st.text_input(
                "Concept name",
                key=f"strategy_concept_name_{_i}",
                placeholder="e.g. Lane Detection",
            )

        with _c2:
            _importance = st.selectbox(
                "Importance",
                ["critical", "important", "supporting"],
                key=f"strategy_concept_importance_{_i}",
            )

        _terms_raw = st.text_area(
            "Alternative / similar patent terms — one per line",
            key=f"strategy_concept_terms_{_i}",
            height=100,
            placeholder=(
                "lane detection\n"
                "lane recognition\n"
                "lane boundary detection"
            ),
        )
        st.caption(
            "Enter one search term per line. Wildcards such as comput$ "
            "can be entered directly."
        )

        _operator = st.selectbox(
            "Operator within this concept",
            ["OR", "AND"],
            key=f"strategy_concept_operator_{_i}",
        )

        _terms = [
            term.strip()
            for term in _terms_raw.splitlines()
            if term.strip()
        ]

        if _name.strip() or _terms:
            _concepts.append(
                SearchConcept(
                    name=_name.strip(),
                    importance=_importance,
                    terms=_terms,
                    operator=_operator,
                )
            )

    st.markdown("### Relationship Between Concepts")

    _concept_operator = st.selectbox(
        "Combine concept groups with",
        ["AND", "OR"],
        key="strategy_concept_operator",
    )

    st.markdown("### Search Fields")

    _f1, _f2, _f3 = st.columns(3)

    with _f1:
        _use_title = st.checkbox("Title", value=True, key="strategy_field_title")

    with _f2:
        _use_abstract = st.checkbox(
            "Abstract",
            value=True,
            key="strategy_field_abstract",
        )

    with _f3:
        _use_claims = st.checkbox(
            "Claims",
            value=True,
            key="strategy_field_claims",
        )

    _search_fields = []

    if _use_title:
        _search_fields.append("title")

    if _use_abstract:
        _search_fields.append("abstract")

    if _use_claims:
        _search_fields.append("claims")

    if st.button(
        "💾 Build Search Strategy",
        type="primary",
        use_container_width=True,
        key="build_search_strategy",
    ):
        _key_points = [
            line.strip()
            for line in _kip_text.splitlines()
            if line.strip()
        ]

        st.session_state.search_strategy = SearchStrategy(
            original_input=query,
            key_inventive_points=_key_points,
            concepts=_concepts,
            concept_operator=_concept_operator,
            search_fields=_search_fields,
        )

        st.session_state.search_strategy_approved = False

    if st.session_state.search_strategy:

        st.divider()
        st.markdown("### Current Search Strategy")

        _strategy = st.session_state.search_strategy

        st.write(
            f"**Original input:** {_strategy.original_input}"
        )

        if _strategy.key_inventive_points:
            st.write("**Key inventive points:**")
            for _point in _strategy.key_inventive_points:
                st.write(f"- {_point}")

        for _i, _concept in enumerate(_strategy.concepts, 1):
            st.markdown(
                f"**Concept {_i}: {_concept.name}** "
                f"({_concept.importance})"
            )

            if _concept.terms:
                st.write(
                    f"{' ' + _concept.operator + ' '.join([])}".strip()
                    if False
                    else f"Terms ({_concept.operator}): "
                    + " · ".join(_concept.terms)
                )

        st.write(
            f"**Between concepts:** {_strategy.concept_operator}"
        )

        st.write(
            f"**Search fields:** "
            f"{', '.join(_strategy.search_fields) if _strategy.search_fields else 'None'}"
        )

        _boolean_search = build_boolean_search(_strategy)

        if _boolean_search:
            st.markdown("### Generated Boolean Search")
            st.code(_boolean_search, language="text")

        _uspto_strings = build_uspto_search_strings(_strategy)

        if _uspto_strings:
            st.markdown("### 🇺🇸 USPTO Search Strings")

            for _field, _expression in _uspto_strings.items():
                st.markdown(f"**{_field.title()}**")
                st.code(_expression, language="text")

        if st.button(
            "✅ Approve Search Strategy",
            use_container_width=True,
            key="approve_search_strategy",
        ):
            st.session_state.search_strategy_approved = True

        # ── Proximity Rules ────────────────────────────────────

        st.markdown("### Proximity Rules")
        st.caption(
            "Define relationships between search terms. "
            "Distance applies to ADJ and NEAR."
        )

        all_strategy_terms = []

        for concept in st.session_state.search_strategy.concepts:
            for term in concept.terms:
                term = term.strip()
                if term and term not in all_strategy_terms:
                    all_strategy_terms.append(term)

        if all_strategy_terms:
            proximity_count = st.number_input(
                "Number of proximity rules",
                min_value=0,
                max_value=10,
                value=len(st.session_state.search_strategy.proximity_rules),
                step=1,
                key="proximity_rule_count",
            )

            proximity_rules = []

            for i in range(proximity_count):
                st.markdown(f"**Proximity Rule {i + 1}**")

                col1, col2, col3 = st.columns(3)

                existing_rule = (
                    st.session_state.search_strategy.proximity_rules[i]
                    if i < len(st.session_state.search_strategy.proximity_rules)
                    else None
                )

                with col1:
                    default_left_index = 0

                    if existing_rule:
                        if existing_rule.left_term in all_strategy_terms:
                            default_left_index = all_strategy_terms.index(
                                existing_rule.left_term
                            )

                    left_term = st.selectbox(
                        "Left term",
                        all_strategy_terms,
                        index=default_left_index,
                        key=f"proximity_left_{i}",
                    )

                with col2:
                    operators = [
                        "ADJ",
                        "NEAR",
                        "WITH",
                        "SAME",
                    ]

                    default_operator_index = 0

                    if existing_rule:
                        if existing_rule.operator in operators:
                            default_operator_index = operators.index(
                                existing_rule.operator
                            )

                    operator = st.selectbox(
                        "Operator",
                        operators,
                        index=default_operator_index,
                        key=f"proximity_operator_{i}",
                    )

                with col3:
                    default_right_index = (
                        1 if len(all_strategy_terms) > 1 else 0
                    )

                    if existing_rule:
                        if existing_rule.right_term in all_strategy_terms:
                            default_right_index = all_strategy_terms.index(
                                existing_rule.right_term
                            )

                    right_term = st.selectbox(
                        "Right term",
                        all_strategy_terms,
                        index=default_right_index,
                        key=f"proximity_right_{i}",
                    )

                distance = 1

                if operator in {"ADJ", "NEAR"}:
                    distance = st.number_input(
                        "Distance",
                        min_value=1,
                        max_value=50,
                        value=(
                            existing_rule.distance
                            if existing_rule and existing_rule.distance
                            else 1
                        ),
                        step=1,
                        key=f"proximity_distance_{i}",
                    )

                field = st.selectbox(
                    "Field",
                    ["title", "abstract", "claims"],
                    index=2,
                    key=f"proximity_field_{i}",
                )

                proximity_rules.append(
                    ProximityRule(
                        left_term=left_term,
                        right_term=right_term,
                        operator=operator,
                        distance=distance,
                        field=field,
                    )
                )

                st.divider()

            if st.button(
                "Save Proximity Rules",
                key="save_proximity_rules",
            ):
                if st.session_state.search_strategy:
                    st.session_state.search_strategy.proximity_rules = (
                        proximity_rules
                    )

                    st.session_state.search_strategy_approved = False

                    st.success(
                        f"Saved {len(proximity_rules)} proximity rule(s)."
                    )

            # ── Generated proximity expressions ────────────────────

            if st.session_state.search_strategy.proximity_rules:
                st.markdown("### Generated USPTO Proximity Search")

                _proximity_strings = build_uspto_proximity_strings(
                    st.session_state.search_strategy
                )

                if _proximity_strings:
                    for _expression in _proximity_strings:
                        st.code(_expression, language="text")

        else:
            st.info(
                "Add search concepts and terms before defining "
                "proximity rules."
            )

        if st.session_state.search_strategy_approved:
            st.success(
                "Search strategy approved — ready for search-string generation."
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
run_claude   = col2.button("🤖 Send Ranked Results to Claude", use_container_width=True,
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
    return rank(query=q, patents=patents, doc_vecs=doc_vecs,
                query_vec=query_vec, top_k=k, critical_tokens=critical or None,
                tokens=tokens)


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
        with st.spinner("Calling Claude for technical comparison & report…"):
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
    "3 · Technical Relevance",
    "4 · Claude Technical Analysis",
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
            # ── Primary token panel ──────────────────────────────────────
            meta = st.session_state.expansion_metadata
            _primary      = meta.get("primary_token", "")
            _supporting   = meta.get("supporting_tokens", [])
            _epo_order    = meta.get("epo_search_order", [])
            _domain       = meta.get("domain_concepts", [])

            if _primary:
                _epo_str = " → ".join(f"`{t}`" for t in _epo_order) if _epo_order else "—"
                st.info(
                    f"🎯 **Primary anchor (inventive concept):** `{_primary}`  \n"
                    f"**Supporting tokens:** {', '.join(f'`{t}`' for t in _supporting) if _supporting else '—'}  \n"
                    f"**EPO search order** *(most → least discriminating)*: {_epo_str}  \n"
                    f"**Domain concepts:** {', '.join(_domain) if _domain else '—'}  \n\n"
                    "_Every expanded query is constrained to keep the primary anchor. "
                    "EPO CQL uses the discriminating order above — multi-word phrases filter first._"
                )
            else:
                st.warning(
                    "⚠️ No primary anchor identified — expansions may drift. "
                    "Try making the inventive concept more explicit in your query."
                )

            st.divider()
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

    st.divider()
    st.subheader("Classification Analysis")
    st.caption(
        "Classification frequency across the unique USPTO search results. "
        "Patent IDs are retained for later refinement."
    )

    classification_data = aggregate_classifications(
        st.session_state.unique_patents
    )

    # ── CPC ────────────────────────────────────────────────
    cpc_data = classification_data.get("cpc", {})

    if cpc_data:
        st.markdown("### CPC")

        cpc_rows = []
        for classification, data in sorted(
            cpc_data.items(),
            key=lambda x: x[1]["count"],
            reverse=True,
        ):
            cpc_rows.append(
                {
                    "CPC": classification,
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
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("#### Review CPC Classifications")

        # Load reviewed AI classifications into the refinement
        # widgets before those widgets are instantiated.
        if "ai_pending_cpc_classifications" in st.session_state:
            st.session_state.selected_cpc_classifications = (
                st.session_state.pop("ai_pending_cpc_classifications")
            )

        if "ai_pending_uspc_classes" in st.session_state:
            st.session_state.selected_uspc_classes = (
                st.session_state.pop("ai_pending_uspc_classes")
            )

        if "ai_pending_uspc_subclasses" in st.session_state:
            st.session_state.selected_uspc_subclasses = (
                st.session_state.pop("ai_pending_uspc_subclasses")
            )

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
                    use_container_width=True,
                    hide_index=True,
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
            use_container_width=True,
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
            use_container_width=True,
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
                use_container_width=True,
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
                use_container_width=True,
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
            use_container_width=True,
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
            "Classification refinement is applied locally to the "
            "currently retrieved USPTO results."
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
                    # Re-rank the refined patent set immediately.
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
                    use_container_width=True,
                    hide_index=True,
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
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(
            "No IPC classifications were returned by the current "
            "USPTO ODP application-search response."
        )

# ── Tab 3: Technical Relevance ─────────────────────────────────────────────────────
with tab_rank:
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

# ── Tab 4: Claude Technical Analysis ───────────────────────────────────────────────────
with tab_claude:
    if st.session_state.stage < 3:
        st.info("Complete stages 1–3 first, then click **Send Ranked Results to Claude**.")
    elif st.session_state.stage == 3:
        st.warning("Top 20 are ready. Click **Send Top 20 to Claude** to continue.")
    else:
        sub1, sub2 = st.tabs(["Technical Comparison JSON", "Report"])

        with sub1:
            st.subheader("Structured technical comparison")
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
