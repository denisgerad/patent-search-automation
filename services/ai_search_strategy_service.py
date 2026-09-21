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
10 distinct technical search concepts.

For each concept, provide exactly 3 strong patent-search keywords
or phrases.

The concepts should represent different technical aspects of the
invention, such as:
- core function
- sensing/input
- processing/algorithm
- technical mechanism
- components
- data/features
- control
- output/result
- implementation technique
- application-specific aspect

Do not generate classifications.
Do not generate Boolean expressions.
Do not generate proximity rules.
Do not generate a complete search strategy.

Return ONLY valid JSON. No markdown fences and no commentary.
"""


USER_TEMPLATE = """
Analyze this invention description:

{query}

Return exactly this JSON structure:

{{
  "concepts": [
    {{
      "name": "Concept name",
      "keywords": [
        "keyword or phrase 1",
        "keyword or phrase 2",
        "keyword or phrase 3"
      ]
    }}
  ]
}}

Rules:
- Exactly 10 concepts.
- Exactly 3 keywords for every concept.
- Each concept must describe a distinct technical aspect.
- Keywords should be useful for patent searching.
- Prefer technical phrases and patent terminology over isolated generic words.
- Keywords within a concept should be related synonyms, variants, or closely related terminology.
- Avoid generic words such as "system", "method", "device", or "technology" unless they are technically meaningful.
- Do not duplicate concepts.
- Do not duplicate keywords unnecessarily.
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


def _normalize_keywords(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []

    keywords: list[str] = []

    for item in raw:
        value = _clean_text(item)

        if value and value.lower() not in {
            existing.lower() for existing in keywords
        }:
            keywords.append(value)

    return keywords


def _normalize_concepts(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_concepts = payload.get("concepts")

    if not isinstance(raw_concepts, list):
        raise ValueError("Claude response does not contain a valid concepts list.")

    concepts: list[dict[str, Any]] = []

    for item in raw_concepts:
        if not isinstance(item, dict):
            continue

        name = _clean_text(item.get("name"))
        keywords = _normalize_keywords(item.get("keywords"))

        if not name:
            continue

        if len(keywords) != 3:
            continue

        concepts.append(
            {
                "name": name,
                "keywords": keywords,
            }
        )

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
