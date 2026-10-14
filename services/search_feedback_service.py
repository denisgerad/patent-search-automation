
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
      "evidence": "Brief explanation grounded in supplied title, abstract or claims",
      "evidence_excerpts": [
        {
          "source": "title | abstract | claims",
          "excerpt": "Exact text copied from the supplied patent field",
          "supports": "The matched mechanism this excerpt supports"
        }
      ]
    }
  ]
}

Evidence assessment requirements:
- Assess each patent against the invention's approved concepts and
  the complete invention description.
- Distinguish a shared generic term from a shared technical mechanism.
- Identify which approved concepts are supported by the patent's
  title, abstract, or claims.
- Identify which important concepts are not established by the
  available patent evidence.
- Treat dependency graphs in software testing, data processing,
  and other domains as adjacent evidence unless the disclosed
  mechanism materially relates to the invention being searched.
- Do not infer incremental score recalculation merely because a
  patent describes dependency tracking.
- Do not infer weighted criterion scoring, eligibility gating,
  or contractor/vendor evaluation unless the supplied evidence
  supports those mechanisms.
- A patent can be partially relevant even when it does not
  disclose the complete invention.
- Provide evidence_excerpts for each patent review.
- Each excerpt must be copied verbatim from the supplied title, abstract,
  or claims. Do not paraphrase inside an excerpt.
- The source must identify the field from which the excerpt was copied:
  title, abstract, or claims.
- The supports value must identify one of the review's matched_mechanisms.
- Every matched mechanism must have at least one corresponding excerpt.
- If the supplied text does not establish a mechanism, do not mark it
  as matched merely because related terminology appears.
- If no adequate excerpt exists, use insufficient_evidence when the
  available text is too limited for a meaningful assessment.
- An empty or unavailable source field must never be treated as evidence.
- When the abstract or claims are empty or inadequate, use
  insufficient_evidence rather than guessing.

Coverage assessment:
- Assess coverage across the approved concepts, not only the
  number of retrieved patents.
- Distinguish gaps in individual technical features from gaps
  in combinations of features.
- Recommend revision when the evidence reveals a specific,
  actionable search gap.
- Do not recommend revision solely because fewer than 20
  patents were retrieved.
- Do not interpret embedding or hybrid ranking scores as
  probabilities of relevance.

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
    patents_by_id = {
        _clean_text(patent.get("patent_id")): patent
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

        matched = clean_string_list(
            review.get("matched_mechanisms", []), "matched_mechanisms"
        )
        missing = clean_string_list(
            review.get("missing_mechanisms", []), "missing_mechanisms"
        )

        raw_excerpts = review.get("evidence_excerpts")
        if not isinstance(raw_excerpts, list):
            raise ValueError(
                f"evidence_excerpts must be a list for patent {patent_id}."
            )

        patent = patents_by_id[patent_id]
        source_fields = {
            "title": _clean_text(patent.get("title")),
            "abstract": _clean_text(patent.get("abstract")),
            "claims": _clean_text(patent.get("claims")),
        }

        excerpts = []
        supported_mechanisms = set()

        for excerpt_item in raw_excerpts:
            if not isinstance(excerpt_item, dict):
                raise ValueError("Each evidence excerpt must be an object.")

            source = _clean_text(excerpt_item.get("source")).lower()
            excerpt = _clean_text(excerpt_item.get("excerpt"))
            supports = _clean_text(excerpt_item.get("supports"))

            if source not in source_fields:
                raise ValueError(
                    f"Invalid evidence source for patent {patent_id}: {source}"
                )

            if not excerpt or excerpt.casefold() not in source_fields[source].casefold():
                raise ValueError(
                    f"Evidence excerpt does not match supplied {source} text "
                    f"for patent {patent_id}."
                )

            if supports not in matched:
                raise ValueError(
                    f"Evidence excerpt for {patent_id} must support a "
                    "listed matched mechanism."
                )

            excerpts.append({
                "source": source,
                "excerpt": excerpt,
                "supports": supports,
            })
            supported_mechanisms.add(supports)

        if label != "insufficient_evidence" and not excerpts:
            raise ValueError(
                f"At least one verified evidence excerpt is required "
                f"for patent {patent_id}."
            )

        unsupported_matches = set(matched) - supported_mechanisms
        if unsupported_matches:
            raise ValueError(
                f"Matched mechanisms lack verified excerpts for {patent_id}: "
                f"{sorted(unsupported_matches)}"
            )

        normalized_reviews.append(
            {
                "patent_id": patent_id,
                "label": label,
                "matched_mechanisms": matched,
                "missing_mechanisms": missing,
                "evidence": evidence,
                "evidence_excerpts": excerpts,
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

    if hasattr(client, "set_usage_stage"):
        client.set_usage_stage("Evidence review")

    last_error = None
    previous_raw = None

    for attempt in range(2):
        correction_prompt = prompt

        if attempt == 1:
            correction_prompt += (
                "\n\nCORRECTION REQUIRED:\n"
                "Your previous response failed validation.\n"
                f"Validation error: {last_error}\n\n"
                "Return the complete corrected JSON object only. "
                "Do not include Markdown fences or explanatory text. "
                "Use only the supplied patent evidence. Every evidence "
                "excerpt must be verbatim from its source field, and its "
                "'supports' value must exactly match a string in that "
                "patent's 'matched_mechanisms' list. Remove unsupported "
                "matches and their excerpts. If the evidence is inadequate, "
                "use 'insufficient_evidence' where appropriate. "
                "Do not invent evidence or unsupported matches."
            )

        raw = client.complete(
            system=REVIEW_SYSTEM_PROMPT,
            user=correction_prompt,
            max_tokens=5000,
        )

        try:
            payload = _extract_json(raw)
            return normalize_search_review(payload, patents, query_names)
        except ValueError as exc:
            last_error = str(exc)
            previous_raw = raw

    raise ValueError(
        "Claude could not produce a valid evidence review after one correction. "
        f"Last validation error: {last_error}"
    )


def select_top_patents(
    patents: list[dict[str, Any]],
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Select the first N patents from an already-ranked list."""
    if limit < 1:
        raise ValueError("limit must be at least 1.")
    return patents[:limit]
