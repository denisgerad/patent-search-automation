"""Integration layer for the bounded patent-search feedback experiment."""

from __future__ import annotations

from typing import Any

from models.claude_client import ClaudeClient
from services.ai_search_strategy_service import (
    _extract_json,
    generate_query_plan,
    normalize_query_plan,
)
from services.search_strategy_service import build_odp_queries_from_plan
from services.search_feedback_service import (
    review_search_results,
    select_top_patents,
)
from services.search_feedback_controller import run_search_feedback_loop
from services.uspto_search_service import search_patents_by_strategy


REVISION_SYSTEM_PROMPT = """
You are revising a patent prior-art search plan.

Return valid JSON only, without Markdown fences.
Use only the supplied approved concepts and their approved search terms.
Do not introduce new terms, concepts, classifications, or unsupported facts.
Preserve the required query-plan structure:
- 3 to 6 queries.
- Each query has a unique name, purpose, concept_indices, and groups.
- Each group has concept_index, terms, and operator (AND or OR).
- Terms must be copied exactly from the approved concepts.
- concept_indices must match the groups.
Focus revisions on the specific evidence-based coverage gaps supplied.
Do not claim that a search is exhaustive or establish patentability.
"""


def _patent_to_dict(patent: Any) -> dict[str, Any]:
    """Convert a PatentRecord or compatible object to reviewer evidence."""
    if isinstance(patent, dict):
        get_value = patent.get
    else:
        get_value = lambda key, default=None: getattr(patent, key, default)

    patent_id = str(get_value("patent_id", "") or "").strip()
    title = str(get_value("patent_title", "") or "").strip()

    return {
        "patent_id": patent_id,
        "title": title,
        "abstract": str(get_value("patent_abstract", "") or "").strip(),
        "claims": str(get_value("patent_claims", "") or "").strip(),
        "patent_date": get_value("patent_date"),
        "search_queries": [],
    }


def _execute_plan(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Execute each query and preserve query provenance for each result."""
    odp_queries = build_odp_queries_from_plan(plan)
    by_patent_id: dict[str, dict[str, Any]] = {}

    for query in odp_queries:
        records = search_patents_by_strategy(
            query["expression"],
            limit=25,
            offset=0,
        )

        for record in records:
            patent = _patent_to_dict(record)
            patent_id = patent["patent_id"]

            if not patent_id:
                continue

            if patent_id not in by_patent_id:
                by_patent_id[patent_id] = patent

            existing = by_patent_id[patent_id]
            if query["name"] not in existing["search_queries"]:
                existing["search_queries"].append(query["name"])

    return list(by_patent_id.values())


def _revise_plan(
    current_plan: dict[str, Any],
    review: dict[str, Any],
    concepts: list[dict[str, Any]],
    invention: str,
    client: ClaudeClient,
) -> dict[str, Any]:
    """Ask Claude for a constrained plan revision and validate it."""
    import json

    prompt = (
        "INVENTION DESCRIPTION:\n"
        f"{invention}\n\n"
        "APPROVED CONCEPTS:\n"
        f"{json.dumps(concepts, ensure_ascii=False, indent=2)}\n\n"
        "CURRENT VALIDATED PLAN:\n"
        f"{json.dumps(current_plan, ensure_ascii=False, indent=2)}\n\n"
        "REVIEW AND COVERAGE GAPS:\n"
        f"{json.dumps(review, ensure_ascii=False, indent=2)}\n\n"
        "Revise the plan only to address the identified gaps. "
        "Return the required query-plan JSON."
    )

    raw = client.complete(
        system=REVISION_SYSTEM_PROMPT,
        user=prompt,
        max_tokens=2400,
    )
    return normalize_query_plan(_extract_json(raw), concepts)


def run_patent_search_feedback_workflow(
    invention: str,
    approved_concepts: list[dict[str, Any]],
    client: ClaudeClient | None = None,
    max_runs: int = 3,
) -> dict[str, Any]:
    """
    Generate a plan, search, review the top 20 results, and optionally revise.

    Search/API errors are allowed to propagate: they must not be misreported
    as evidence that no relevant prior art exists.
    """
    if not invention.strip():
        raise ValueError("Invention description cannot be empty.")
    if not approved_concepts:
        raise ValueError("Approved search concepts are required.")
    if not 1 <= max_runs <= 3:
        raise ValueError("max_runs must be between 1 and 3.")

    active_client = client or ClaudeClient()
    initial_plan = generate_query_plan(
        invention,
        approved_concepts,
        active_client,
    )

    def execute(plan: dict[str, Any]) -> list[dict[str, Any]]:
        return _execute_plan(plan)

    def review(
        plan: dict[str, Any],
        patents: list[dict[str, Any]],
    ) -> dict[str, Any]:
        queries = build_odp_queries_from_plan(plan)

        if not patents:
            return {
                "decision": "insufficient_evidence",
                "rationale": (
                    "The executed queries returned no patent records. "
                    "This does not establish that relevant prior art is absent."
                ),
                "coverage_gaps": [],
                "queries_to_revise": [],
                "patent_reviews": [],
            }

        # The reviewer must assess the bounded result set, not an unbounded list.
        top_patents = select_top_patents(patents, limit=20)

        return review_search_results(
            invention=invention,
            concepts=approved_concepts,
            queries=queries,
            patents=top_patents,
            client=active_client,
        )

    def revise(
        plan: dict[str, Any],
        feedback: dict[str, Any],
    ) -> dict[str, Any]:
        return _revise_plan(
            current_plan=plan,
            review=feedback,
            concepts=approved_concepts,
            invention=invention,
            client=active_client,
        )

    return run_search_feedback_loop(
        initial_plan=initial_plan,
        execute_search=execute,
        review_results=review,
        revise_plan=revise,
        max_runs=max_runs,
    )
