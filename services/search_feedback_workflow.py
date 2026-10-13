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
from services.search_feedback_service import review_search_results
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
If the previous search returned zero patent records:
- Treat this as a retrieval failure, not evidence that prior art is absent.
- Review the previous query expressions and approved terminology.
- Broaden or redirect the next search using only approved terms.
- Consider relaxing an AND between concept groups, or searching a concept independently.
- Do not merely repeat the previous plan.
- Explain the search-strategy change in each revised query's purpose.

If the previous search returned records but the review found
poor relevance or incomplete coverage:
- Treat retrieval success and technical relevance as separate outcomes.
- Preserve useful evidence from the previous run.
- Use the review's patent labels, matched mechanisms, missing
  mechanisms, and coverage_gaps to guide revision.
- Do not repeat a query strategy that produced mostly unrelated
  patents without a specific reason.
- Search individual concepts independently when this may recover
  relevant terminology.
- Use combination queries to investigate relationships between
  important technical features.
- Prioritize gaps supported by the review rather than inventing
  new gaps.
- Explain the intended coverage improvement in each revised
  query's purpose.
- Do not claim that zero results establish the absence of prior art.
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
        "continuity_family_id": get_value("continuity_family_id"),
        "search_queries": [],
    }


def _rank_candidates(
    invention: str,
    patents: list[dict[str, Any]],
    concepts: list[dict[str, Any]],
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Rank feedback-loop candidates with the existing ranking service."""
    from models.embedding_model import EmbeddingModel
    from models.schemas import PatentRecord, SearchConcept, SearchStrategy
    from services.embedding_service import embed_patents, embed_query
    from services.ranking_service import rank

    if not patents:
        return []

    records = [
        PatentRecord(
            patent_id=item["patent_id"],
            patent_title=item.get("title", ""),
            patent_abstract=item.get("abstract") or None,
            patent_claims=item.get("claims") or None,
            patent_date=item.get("patent_date"),
            continuity_family_id=item.get("continuity_family_id"),
            search_paths=item.get("search_queries", []),
        )
        for item in patents
    ]

    search_strategy = SearchStrategy(
        original_input=invention,
        concepts=[
            SearchConcept(
                name=str(concept.get("name") or concept.get("primary_term") or "Concept"),
                importance=str(concept.get("importance", "important")),
                terms=list(dict.fromkeys(
                    term
                    for term in [
                        concept.get("primary_term"),
                        *(concept.get("alternative_terms") or []),
                    ]
                    if isinstance(term, str) and term.strip()
                )),
                operator="OR",
            )
            for concept in concepts
        ],
    )

    model = EmbeddingModel()
    doc_vecs = embed_patents(records, model)
    query_vec = embed_query(invention, model)

    ranked = rank(
        query=invention,
        patents=records,
        doc_vecs=doc_vecs,
        query_vec=query_vec,
        top_k=limit,
        search_strategy=search_strategy,
    )

    results = []
    seen_ids: set[str] = set()
    for item in ranked:
        patent = item.patent
        patent_id = patent.patent_id
        if patent_id in seen_ids:
            continue
        seen_ids.add(patent_id)
        results.append({
            "patent_id": patent_id,
            "title": patent.patent_title,
            "abstract": patent.patent_abstract or "",
            "claims": patent.patent_claims or "",
            "patent_date": patent.patent_date,
            "search_queries": list(patent.search_paths),
            "cosine_score": item.cosine_score,
            "bm25_score": item.bm25_score,
            "hybrid_score": item.hybrid_score,
            "coverage": item.coverage,
            "importance_score": item.importance_score,
        })

    return results[:limit]


def _execute_plan(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Execute planned queries with a bounded unquoted fallback."""
    odp_queries = build_odp_queries_from_plan(plan)
    by_patent_id: dict[str, dict[str, Any]] = {}

    for query in odp_queries:
        records = search_patents_by_strategy(
            query["expression"],
            limit=25,
            offset=0,
        )
        provenance = query["name"]

        # Retry only when the original expression returned no records.
        fallback_expression = query["expression"].replace('"', "")
        if not records and fallback_expression != query["expression"]:
            records = search_patents_by_strategy(
                fallback_expression,
                limit=25,
                offset=0,
            )
            if records:
                provenance = f'{query["name"]} (unquoted fallback)'

        for record in records:
            patent = _patent_to_dict(record)
            patent_id = patent["patent_id"]

            if not patent_id:
                continue

            if patent_id not in by_patent_id:
                by_patent_id[patent_id] = patent

            existing = by_patent_id[patent_id]
            if provenance not in existing["search_queries"]:
                existing["search_queries"].append(provenance)

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
        top_patents = _rank_candidates(
            invention=invention,
            patents=patents,
            concepts=approved_concepts,
            limit=8,
        )

        if not top_patents:
            return {
                "decision": "insufficient_evidence",
                "rationale": (
                    f"The search retrieved {len(patents)} patent records, "
                    "but the ranking stage returned no candidates for review. "
                    "This is a ranking or candidate-selection limitation, "
                    "not evidence that relevant prior art is absent."
                ),
                "coverage_gaps": [
                    "Retrieved records could not be assessed because "
                    "no candidates survived the ranking stage."
                ],
                "queries_to_revise": [],
                "patent_reviews": [],
            }

        try:
            return review_search_results(
                invention=invention,
                concepts=approved_concepts,
                queries=queries,
                patents=top_patents,
                client=active_client,
            )
        except ValueError as exc:
            # Never convert an invalid AI review into a positive match.
            return {
                "decision": "insufficient_evidence",
                "rationale": (
                    "Patent candidates were retrieved and ranked, but the "
                    "AI evidence review could not be validated after its "
                    "correction attempt. No mechanism-match conclusion "
                    "has been accepted. Review validation error: "
                    f"{exc}"
                ),
                "coverage_gaps": [
                    "The ranked candidates remain unreviewed for verified "
                    "mechanism overlap because the AI evidence review failed "
                    "validation. Inspect the patent evidence before drawing "
                    "a conclusion."
                ],
                "queries_to_revise": [],
                "patent_reviews": [],
            }

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
