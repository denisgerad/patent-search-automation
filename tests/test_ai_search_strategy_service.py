
import json

import pytest

from services.ai_search_strategy_service import (
    _normalize_concepts,
    generate_search_concepts,
)


def make_payload():
    return {
        "concepts": [
            {
                "name": f"Concept {i}",
                "primary_term": f"term {i}",
                "alternative_terms": [
                    f"alternative {i}a",
                    f"alternative {i}b",
                ],
                "importance": ["critical", "important", "supporting"][
                    (i - 1) % 3
                ],
                "classification_reason": f"Reason for concept {i}.",
                "confidence": "medium",
            }
            for i in range(1, 11)
        ]
    }


def test_normalize_preserves_classification():
    concepts = _normalize_concepts(make_payload())

    assert len(concepts) == 10
    assert concepts[0]["importance"] == "critical"
    assert concepts[1]["importance"] == "important"
    assert concepts[2]["importance"] == "supporting"
    assert concepts[0]["classification_reason"] == "Reason for concept 1."
    assert concepts[0]["confidence"] == "medium"


def test_invalid_classification_uses_safe_fallback():
    payload = make_payload()
    payload["concepts"][0]["importance"] = "essential"
    payload["concepts"][0]["confidence"] = "certain"

    concepts = _normalize_concepts(payload)

    assert concepts[0]["importance"] == "important"
    assert concepts[0]["confidence"] == "low"


def test_generation_requests_and_returns_classification():
    class FakeClaudeClient:
        def complete(self, *, system, user, max_tokens):
            assert "Do not generate classifications." not in system
            assert "classification_reason" in system
            assert "confidence" in system
            return json.dumps(make_payload())

    concepts = generate_search_concepts(
        "A system that evaluates contractors using weighted criteria.",
        FakeClaudeClient(),
    )

    assert len(concepts) == 10
    assert all(c["importance"] in {
        "critical", "important", "supporting"
    } for c in concepts)
    assert all(c["classification_reason"] for c in concepts)
