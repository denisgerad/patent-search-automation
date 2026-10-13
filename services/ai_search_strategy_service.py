"""Claude-assisted generation of patent search concepts and keywords.

Claude proposes concepts and patent-search keywords only.
The user reviews/edits these suggestions before they become a
SearchStrategy.
"""

from __future__ import annotations

import json
import re
from typing import Any

from models.claude_client import ClaudeClient


SYSTEM_PROMPT = """
You are an expert patent search strategist.

Your task is to decompose an invention description into exactly
10 distinct technical search concepts for prior-art retrieval.

The goal is to understand the invention accurately and identify
search terminology that is likely to retrieve relevant patent
documents.

For each concept, identify:

1. A clear technical concept name.
2. One PRIMARY SEARCH TERM.
3. Exactly two ALTERNATIVE SEARCH TERMS.
4. An importance classification selected from exactly one of:
   - critical: central to the invention's proposed technical contribution;
   - important: a meaningful mechanism or feature that strengthens or
     distinguishes the approach but may not be essential to its core;
   - supporting: supplementary context or implementation detail.
5. A concise classification_reason grounded in the invention description.
6. A confidence value of high, medium, or low.

The PRIMARY SEARCH TERM is the most important requirement.

A primary search term must:
- be short and searchable;
- use terminology likely to appear in patent documents;
- represent the core technical meaning of the concept;
- prefer established technical terminology over polished
  descriptions of the invention;
- avoid unnecessary words;
- avoid artificial phrases created only by combining words
  from the invention description.

Prefer terms such as:
- "facial recognition"
- "attendance"
- "parking space"
- "parking sensor"
- "occupancy detection"
- "parking guidance"

Avoid unnecessarily long phrases such as:
- "automatic attendance logging system"
- "nearest available parking space selection mechanism"
- "real-time dynamic parking availability guidance"

A primary term should NOT simply restate an entire sentence
from the invention.

Alternative search terms should:
- be genuine terminology variants;
- provide useful additional retrieval coverage;
- be reasonably likely to occur in patent documents;
- remain concise;
- not merely repeat the primary term.

Distinguish between:
- core inventive concepts;
- important technical mechanisms;
- supporting implementation details.

Base classification on the supplied invention description.
Do not assume that a feature is essential merely because it is
mentioned first or repeated frequently.
Do not make legal conclusions about patentability or claim scope.

Do not introduce technical features that are not stated or
reasonably supported by the invention description.

Concepts should cover different technical aspects of the
invention and should not duplicate one another.

The first concepts should focus on the strongest technical
ideas that distinguish the invention.

Later concepts may cover supporting mechanisms and
implementation details.

Do not generate Boolean expressions.
Do not generate proximity rules.
Do not generate a complete search strategy.

Return ONLY valid JSON.
No markdown fences.
No commentary.
"""


USER_TEMPLATE = """
Analyze this invention description:

{query}

Return exactly this JSON structure:

{{
  "concepts": [
    {{
      "name": "Concept name",
      "primary_term": "primary patent search term",
      "alternative_terms": [
        "alternative search term 1",
        "alternative search term 2"
      ],
      "importance": "critical",
      "classification_reason": "Why this concept has this importance",
      "confidence": "medium"
    }}
  ]
}}

Rules:

- Exactly 10 concepts.
- Exactly 1 primary_term for every concept.
- Exactly 2 alternative_terms for every concept.
- Each concept must describe a distinct technical aspect.
- The primary_term must be the strongest concise retrieval
  anchor for that concept.
- Primary terms should normally be short technical terms or
  established technical phrases.
- Prefer terminology likely to occur literally in patent
  titles, abstracts, claims, or technical descriptions.
- Alternatives should provide genuine terminology variation.
- Do not make all three terms long phrases.
- Do not simply copy long phrases from the invention description.
- Do not create artificial patent terminology.
- Do not introduce unsupported features.
- Do not duplicate concepts.
- Do not duplicate the primary term in the alternatives.
- Avoid generic words such as "system", "method", "device",
  or "technology" unless they are technically meaningful.
- Do not include CPC, USPC, IPC, or other classifications.
- Do not include Boolean operators.
- Do not include explanations outside the JSON.
- Classify every concept as critical, important, or supporting.
- Include a concise classification_reason for every concept.
- Set confidence to high, medium, or low.
- Do not classify concepts solely by their order in the output.
- Ensure the importance and reason are consistent with the invention.
- Keep the existing rules requiring exactly ten concepts and exactly two alternatives per concept.
"""


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _extract_json(raw: str) -> dict[str, Any]:
    """Extract a JSON object from Claude's response."""
    text = _clean_text(raw)

    if not text:
        raise ValueError("Claude returned an empty response.")

    # Handle accidental markdown fences.
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Claude returned invalid JSON: {exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError("Claude response must be a JSON object.")

    return payload


def _normalize_search_terms(raw: Any) -> tuple[str, list[str]]:
    """Normalize one primary search term and two alternatives."""

    if not isinstance(raw, dict):
        return "", []

    primary = str(raw.get("primary_term") or "").strip()

    alternatives_raw = raw.get("alternative_terms")

    if not isinstance(alternatives_raw, list):
        return primary, []

    alternatives: list[str] = []

    for value in alternatives_raw:
        value = str(value).strip()

        if not value:
            continue

        if value.lower() == primary.lower():
            continue

        if any(existing.lower() == value.lower() for existing in alternatives):
            continue

        alternatives.append(value)

    return primary, alternatives


def _normalize_concepts(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate and normalize ten concepts and their classifications."""
    raw_concepts = payload.get("concepts")

    if not isinstance(raw_concepts, list):
        raise ValueError(
            "Claude response does not contain a valid concepts list."
        )

    valid_importance = {"critical", "important", "supporting"}
    valid_confidence = {"high", "medium", "low"}
    concepts: list[dict[str, Any]] = []

    for item in raw_concepts:
        if not isinstance(item, dict):
            continue

        primary_term, alternative_terms = _normalize_search_terms(item)

        if not primary_term or len(alternative_terms) != 2:
            continue

        name = str(item.get("name") or "").strip()
        if not name:
            continue

        importance = str(item.get("importance") or "").strip().lower()
        if importance not in valid_importance:
            importance = "important"

        confidence = str(item.get("confidence") or "").strip().lower()
        if confidence not in valid_confidence:
            confidence = "low"

        reason = str(item.get("classification_reason") or "").strip()
        if not reason:
            reason = "Claude did not provide a classification rationale."
            confidence = "low"

        concepts.append(
            {
                "name": name,
                "primary_term": primary_term,
                "alternative_terms": alternative_terms,
                "importance": importance,
                "classification_reason": reason,
                "confidence": confidence,
            }
        )

    if len(concepts) != 10:
        raise ValueError(
            f"Claude returned {len(concepts)} valid concepts; "
            "exactly 10 are required."
        )

    return concepts


def generate_search_concepts(
    invention: str,
    client: ClaudeClient,
) -> list[dict[str, Any]]:
    """Generate exactly 10 concepts with 3 keywords each."""
    query = _clean_text(invention)

    if not query:
        raise ValueError("Invention description cannot be empty.")

    prompt = USER_TEMPLATE.format(query=query)

    raw = client.complete(
        system=SYSTEM_PROMPT,
        user=prompt,
        max_tokens=1200,
    )

    payload = _extract_json(raw)
    return _normalize_concepts(payload)


QUERY_PLAN_SYSTEM_PROMPT = """
You are a patent search strategist.

Design a search plan for the supplied invention using only the
approved concepts and their terms.

Return valid JSON only. Do not use Markdown fences.

Rules:
- Produce 3 to 6 distinct search queries.
- Each query must have a name, purpose, concept_indices, and groups.
- concept_indices are 1-based indices into the supplied concepts list.
- Each group must identify one concept_index and one or more terms
  drawn from that concept's primary_term and alternative_terms.
- Each group must use operator AND or OR.
- Use OR to express terminology alternatives within a concept.
- Use AND between groups when the combination is technically meaningful.
- Include targeted queries for the most important inventive mechanisms.
- Every concept classified as critical must have its own independent
  single-concept query.
- A query combining multiple concepts does not satisfy this requirement.
- Also include combination queries where the relationship between
  mechanisms is technically meaningful.
- Prefer coverage of distinct mechanisms over redundant queries.
- The complete plan must satisfy all validation rules and contain
  between 3 and 6 queries.
- Do not make every query depend on every concept.
- Cover different aspects of the invention across the plan.
- Do not invent concepts or terminology.
- Do not return raw USPTO query syntax or field names. Python will
  translate this plan into the search provider's supported syntax.
- Do not make patentability or legal conclusions.

Required structure:
{
  "queries": [
    {
      "name": "Short descriptive name",
      "purpose": "What this query is intended to retrieve",
      "concept_indices": [1, 2],
      "groups": [
        {
          "concept_index": 1,
          "terms": ["term one", "term alternative"],
          "operator": "OR"
        },
        {
          "concept_index": 2,
          "terms": ["another term"],
          "operator": "OR"
        }
      ]
    }
  ]
}
"""


def normalize_query_plan(
    payload: dict[str, Any],
    concepts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate Claude's structured query plan without executing it."""
    raw_queries = payload.get("queries")
    if not isinstance(raw_queries, list):
        raise ValueError("Query plan must contain a queries list.")

    if not 3 <= len(raw_queries) <= 6:
        raise ValueError("Query plan must contain between 3 and 6 queries.")

    normalized_queries: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    allowed_operators = {"AND", "OR"}

    for position, item in enumerate(raw_queries, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Query {position} must be an object.")

        name = _clean_text(item.get("name"))
        purpose = _clean_text(item.get("purpose"))
        groups_raw = item.get("groups")

        if not name or not purpose:
            raise ValueError(f"Query {position} needs a name and purpose.")

        name_key = name.casefold()
        if name_key in seen_names:
            raise ValueError(f"Duplicate query name: {name}")
        seen_names.add(name_key)

        if not isinstance(groups_raw, list) or not groups_raw:
            raise ValueError(f"Query {position} must contain groups.")

        groups: list[dict[str, Any]] = []
        used_indices: set[int] = set()

        for group in groups_raw:
            if not isinstance(group, dict):
                raise ValueError(f"Query {position} contains an invalid group.")

            try:
                concept_index = int(group.get("concept_index"))
            except (TypeError, ValueError) as exc:
                raise ValueError("Each group needs a valid concept_index.") from exc

            if not 1 <= concept_index <= len(concepts):
                raise ValueError(
                    f"Concept index {concept_index} is outside the approved concepts."
                )

            if concept_index in used_indices:
                raise ValueError(
                    f"Query {position} repeats concept index {concept_index}."
                )

            operator = _clean_text(group.get("operator")).upper()
            if operator not in allowed_operators:
                raise ValueError("Group operator must be AND or OR.")

            terms_raw = group.get("terms")
            if not isinstance(terms_raw, list) or not terms_raw:
                raise ValueError("Each group must contain at least one term.")

            concept_entry = concepts[concept_index - 1]
            approved_terms = {
                _clean_text(concept_entry.get("primary_term")).casefold(),
                *(
                    _clean_text(term).casefold()
                    for term in concept_entry.get("alternative_terms", [])
                ),
            }
            approved_terms.discard("")

            terms: list[str] = []
            for raw_term in terms_raw:
                term = _clean_text(raw_term)
                if not term:
                    continue
                if term.casefold() not in approved_terms:
                    raise ValueError(
                        f"Query {position} uses an unapproved term "
                        f"for concept {concept_index}: {term}"
                    )
                if term.casefold() not in {value.casefold() for value in terms}:
                    terms.append(term)

            if not terms:
                raise ValueError("Each group must contain an approved non-empty term.")

            used_indices.add(concept_index)
            groups.append(
                {
                    "concept_index": concept_index,
                    "terms": terms,
                    "operator": operator,
                }
            )

        declared_indices = item.get("concept_indices")
        if not isinstance(declared_indices, list):
            raise ValueError(f"Query {position} needs concept_indices.")

        try:
            declared = {int(value) for value in declared_indices}
        except (TypeError, ValueError) as exc:
            raise ValueError("concept_indices must contain integers.") from exc

        if declared != used_indices:
            raise ValueError(
                f"Query {position}: concept_indices must match the groups."
            )

        normalized_queries.append(
            {
                "name": name,
                "purpose": purpose,
                "concept_indices": sorted(used_indices),
                "groups": groups,
            }
        )

    critical_indices = {
        index
        for index, concept in enumerate(concepts, start=1)
        if _clean_text(concept.get("importance")).lower() == "critical"
    }

    covered_indices = {
        index
        for query in normalized_queries
        for index in query["concept_indices"]
    }

    missing_critical = critical_indices - covered_indices
    if missing_critical:
        raise ValueError(
            "Critical concepts missing from query plan: "
            f"{sorted(missing_critical)}."
        )

    standalone_indices = {
        query["concept_indices"][0]
        for query in normalized_queries
        if len(query["concept_indices"]) == 1
    }

    critical_without_standalone = critical_indices - standalone_indices
    if critical_without_standalone:
        raise ValueError(
            "Critical concepts require an independent single-concept query: "
            f"{sorted(critical_without_standalone)}."
        )

    return {"queries": normalized_queries}


def _repair_critical_standalone_queries(
    payload: dict[str, Any],
    concepts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Ensure critical concepts have standalone queries within the plan limit."""
    queries = payload.get("queries")
    if not isinstance(queries, list):
        return payload

    critical_indices = {
        index
        for index, concept in enumerate(concepts, start=1)
        if _clean_text(concept.get("importance")).lower() == "critical"
    }

    standalone_indices = set()
    for query in queries:
        if not isinstance(query, dict):
            continue
        groups = query.get("groups")
        if not isinstance(groups, list):
            continue

        indices = []
        for group in groups:
            if not isinstance(group, dict):
                continue
            try:
                indices.append(int(group.get("concept_index")))
            except (TypeError, ValueError):
                continue

        if len(set(indices)) == 1 and len(indices) == 1:
            standalone_indices.add(indices[0])

    missing = sorted(critical_indices - standalone_indices)

    if len(critical_indices) > 6:
        raise ValueError(
            "Cannot create independent queries for all critical concepts: "
            "the plan limit is six queries, but there are more than six "
            "critical concepts."
        )

    for index in missing:
        concept = concepts[index - 1]
        primary_term = _clean_text(concept.get("primary_term"))

        if not primary_term:
            raise ValueError(
                f"Critical concept {index} has no approved primary_term."
            )

        queries.append({
            "name": f"Critical concept {index} - standalone",
            "purpose": (
                f"Search for the critical mechanism: {primary_term}."
            ),
            "concept_indices": [index],
            "groups": [{
                "concept_index": index,
                "terms": [primary_term],
                "operator": "OR",
            }],
        })

    while len(queries) > 6:
        combination_position = next(
            (
                position
                for position in range(len(queries) - 1, -1, -1)
                if isinstance(queries[position], dict)
                and isinstance(queries[position].get("groups"), list)
                and len(queries[position]["groups"]) > 1
            ),
            None,
        )

        if combination_position is None:
            raise ValueError(
                "Cannot repair query plan within the six-query limit "
                "without removing standalone searches."
            )

        queries.pop(combination_position)

    return payload


def generate_query_plan(
    invention: str,
    concepts: list[dict[str, Any]],
    client: ClaudeClient,
) -> dict[str, Any]:
    """Generate a query plan, retrying once when validation fails."""
    query = _clean_text(invention)
    if not query:
        raise ValueError("Invention description cannot be empty.")

    if not isinstance(concepts, list) or not concepts:
        raise ValueError("Approved concepts are required.")

    prompt = (
        "INVENTION DESCRIPTION:\n"
        f"{query}\n\n"
        "APPROVED CONCEPTS (indices are 1-based):\n"
        f"{json.dumps(concepts, ensure_ascii=False, indent=2)}\n\n"
        "Create the required structured query plan."
    )

    last_error = None

    for attempt in range(2):
        if attempt == 1:
            prompt += (
                "\n\nCORRECTION REQUIRED:\n"
                f"The previous plan failed validation: {last_error}\n"
                "Return a corrected complete JSON query plan. "
                "Preserve all approved concepts. Every critical concept "
                "must have its own single-concept query. Keep useful "
                "combination queries where the query limit permits. "
                "Follow every original validation rule."
            )

        raw = client.complete(
            system=QUERY_PLAN_SYSTEM_PROMPT,
            user=prompt,
            max_tokens=2400,
        )

        try:
            payload = _extract_json(raw)
            payload = _repair_critical_standalone_queries(payload, concepts)
            return normalize_query_plan(payload, concepts)
        except ValueError as exc:
            last_error = str(exc)

    raise ValueError(
        "Claude could not produce a valid query plan after one correction. "
        f"Last validation error: {last_error}"
    )
