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

Do not assume every implementation detail is essential.

Do not introduce technical features that are not stated or
reasonably supported by the invention description.

Concepts should cover different technical aspects of the
invention and should not duplicate one another.

The first concepts should focus on the strongest technical
ideas that distinguish the invention.

Later concepts may cover supporting mechanisms and
implementation details.

Do not generate classifications.
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
      ]
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
    raw_concepts = payload.get("concepts")

    if not isinstance(raw_concepts, list):
        raise ValueError("Claude response does not contain a valid concepts list.")

    concepts: list[dict[str, Any]] = []

    for item in raw_concepts:
        if not isinstance(item, dict):
            continue

        primary_term, alternative_terms = _normalize_search_terms(item)

        if not primary_term:
            continue

        if len(alternative_terms) != 2:
            continue

        normalized = {
            "name": str(item.get("name") or "").strip(),
            "primary_term": primary_term,
            "alternative_terms": alternative_terms,
        }

        if not normalized["name"]:
            continue

        concepts.append(normalized)

    if len(concepts) != 10:
        raise ValueError(
            f"Claude returned {len(concepts)} valid concepts; exactly 10 are required."
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
