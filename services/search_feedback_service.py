
"""Evidence-based review of patent search results using Claude."""

from __future__ import annotations

import json
from typing import Any, Callable

from services.ai_search_strategy_service import _extract_json, _clean_text


REVIEW_SYSTEM_PROMPT = """
You are reviewing patent search results for technical relevance and search
coverage. You are not making a legal patentability determination.

Return valid JSON only, without Markdown fences.

Assess the results against the supplied invention and approved concepts.
Use only the supplied patent evidence. Do not invent missing facts.

Return this structure:
{
  "decision": "accept | revise | insufficient_evidence",
  "rationale": "Evidence-based explanation",
  "coverage_gaps": ["Specific technical mechanism not adequately covered"],
  "queries_to_revise": [
    {
      "query_name": "Name of an existing query",
      "reason": "Evidence-based reason for revision",
      "suggested_focus": "Approved concepts or terms to focus on"
    }
  ],
  "patent_reviews": [
    {
      "patent_id": "Supplied patent identifier",
      "label": "relevant | partially_relevant | not_relevant | insufficient_evidence",
      "matched_mechanisms": ["Mechanisms supported by the supplied evidence"],
      "missing_mechanisms": ["Important mechanisms not shown in the evidence"],
      "evidence": "Brief explanation grounded in supplied title, abstract or claims"
    }
  ]
}

Decision rules:
- accept means the results appear adequate for the stated search objective,
  not that the search is exhaustive or the invention is patentable.
- revise means there is a specific, actionable search-coverage gap.
- insufficient_evidence means the supplied result text is too limited to
  make a meaningful coverage assessment.
- Do not recommend revision merely because results are not exact matches.
- Do not call a patent relevant based only on a shared generic term.
- Do not invent patent identifiers or facts.
- Review every supplied patent and preserve its exact identifier.
- Return an empty queries_to_revise list unless revision is justified.
"""


def normalize_search_review(
    payload: dict[str, Any],
    patents: list[dict[str, Any]],
    query_names: set[str],
) -> dict[str, Any]:
    """Validate a Claude review against the supplied search results."""
    decisions = {"accept", "revise", "insufficient_evidence"}
    labels = {
        "relevant",
        "partially_relevant",
        "not_relevant",
        "insufficient_evidence",
    }

    decision = _clean_text(payload.get("decision")).lower()
    if decision not in decisions:
        raise ValueError("Review decision is invalid.")

    rationale = _clean_text(payload.get("rationale"))
    if not rationale:
        raise ValueError("Review rationale is required.")

    def clean_string_list(value: Any, field: str) -> list[str]:
        if not isinstance(value, list):
            raise ValueError(f"{field} must be a list.")
        cleaned = []
        for item in value:
            text = _clean_text(item)
            if text:
                cleaned.append(text)
        return cleaned

    gaps = clean_string_list(payload.get("coverage_gaps", []), "coverage_gaps")

    raw_reviews = payload.get("patent_reviews")
    if not isinstance(raw_reviews, list):
        raise ValueError("patent_reviews must be a list.")

    expected_ids = {
        _clean_text(patent.get("patent_id"))
        for patent in patents
        if _clean_text(patent.get("patent_id"))
    }
    reviewed_ids: set[str] = set()
    normalized_reviews = []

    for review in raw_reviews:
        if not isinstance(review, dict):
            raise ValueError("Each patent review must be an object.")

        patent_id = _clean_text(review.get("patent_id"))
        if patent_id not in expected_ids:
            raise ValueError(f"Review contains an unknown patent ID: {patent_id}")
        if patent_id in reviewed_ids:
            raise ValueError(f"Duplicate patent review: {patent_id}")

        label = _clean_text(review.get("label")).lower()
        if label not in labels:
            raise ValueError(f"Invalid relevance label for {patent_id}.")

        evidence = _clean_text(review.get("evidence"))
        if not evidence:
            raise ValueError(f"Evidence is required for patent {patent_id}.")

        normalized_reviews.append(
            {
                "patent_id": patent_id,
                "label": label,
                "matched_mechanisms": clean_string_list(
                    review.get("matched_mechanisms", []), "matched_mechanisms"
                ),
                "missing_mechanisms": clean_string_list(
                    review.get("missing_mechanisms", []), "missing_mechanisms"
                ),
                "evidence": evidence,
            }
        )
        reviewed_ids.add(patent_id)

    if reviewed_ids != expected_ids:
        missing = sorted(expected_ids - reviewed_ids)
        raise ValueError(f"Review is missing supplied patents: {missing}")

    raw_revisions = payload.get("queries_to_revise", [])
    if not isinstance(raw_revisions, list):
        raise ValueError("queries_to_revise must be a list.")

    revisions = []
    for revision in raw_revisions:
        if not isinstance(revision, dict):
            raise ValueError("Each query revision must be an object.")

        query_name = _clean_text(revision.get("query_name"))
        reason = _clean_text(revision.get("reason"))
        suggested_focus = _clean_text(revision.get("suggested_focus"))

        if query_name not in query_names:
            raise ValueError(f"Revision references an unknown query: {query_name}")
        if not reason or not suggested_focus:
            raise ValueError("Query revisions need a reason and suggested_focus.")

        revisions.append(
            {
                "query_name": query_name,
                "reason": reason,
                "suggested_focus": suggested_focus,
            }
        )

    if decision == "revise" and not revisions and not gaps:
        raise ValueError(
            "A revise decision must identify an actionable coverage gap or query revision."
        )

    return {
        "decision": decision,
        "rationale": rationale,
        "coverage_gaps": gaps,
        "queries_to_revise": revisions,
        "patent_reviews": normalized_reviews,
    }


def review_search_results(
    invention: str,
    concepts: list[dict[str, Any]],
    queries: list[dict[str, str]],
    patents: list[dict[str, Any]],
    client: Any,
) -> dict[str, Any]:
    """Ask Claude to review the supplied patents and validate its response."""
    if not _clean_text(invention):
        raise ValueError("Invention description cannot be empty.")
    if not patents:
        raise ValueError("At least one patent is required for review.")

    query_names = {
        _clean_text(query.get("name"))
        for query in queries
        if _clean_text(query.get("name"))
    }
    if not query_names:
        raise ValueError("At least one named search query is required.")

    prompt = (
        "INVENTION:\n"
        f"{invention}\n\n"
        "APPROVED CONCEPTS:\n"
        f"{json.dumps(concepts, ensure_ascii=False, indent=2)}\n\n"
        "QUERIES EXECUTED:\n"
        f"{json.dumps(queries, ensure_ascii=False, indent=2)}\n\n"
        "PATENT RESULTS TO REVIEW:\n"
        f"{json.dumps(patents, ensure_ascii=False, indent=2)}\n\n"
        "Review all supplied patents and return the required JSON."
    )

    raw = client.complete(
        system=REVIEW_SYSTEM_PROMPT,
        user=prompt,
        max_tokens=5000,
    )
    payload = _extract_json(raw)
    return normalize_search_review(payload, patents, query_names)


def select_top_patents(
    patents: list[dict[str, Any]],
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Select the first N patents from an already-ranked list."""
    if limit < 1:
        raise ValueError("limit must be at least 1.")
    return patents[:limit]
