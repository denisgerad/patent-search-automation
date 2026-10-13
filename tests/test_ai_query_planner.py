
import json

import pytest

from services.ai_search_strategy_service import (
    generate_query_plan,
    normalize_query_plan,
)


@pytest.fixture
def concepts():
    return [
        {
            "name": "Contractor matching",
            "primary_term": "contractor matching",
            "alternative_terms": ["contractor selection", "supplier matching"],
            "importance": "critical",
        },
        {
            "name": "Dependency tracking",
            "primary_term": "dependency graph",
            "alternative_terms": ["dependency tracking", "dependency relationship"],
            "importance": "critical",
        },
        {
            "name": "Incremental scoring",
            "primary_term": "incremental score",
            "alternative_terms": ["score recalculation", "incremental ranking"],
            "importance": "important",
        },
    ]


@pytest.fixture
def valid_plan():
    return {
        "queries": [
            {
                "name": "Matching and dependencies",
                "purpose": "Retrieve systems combining matching with dependency tracking.",
                "concept_indices": [1, 2],
                "groups": [
                    {
                        "concept_index": 1,
                        "terms": ["contractor matching", "contractor selection"],
                        "operator": "OR",
                    },
                    {
                        "concept_index": 2,
                        "terms": ["dependency graph"],
                        "operator": "OR",
                    },
                ],
            },
            {
                "name": "Incremental scoring",
                "purpose": "Retrieve incremental scoring mechanisms.",
                "concept_indices": [3],
                "groups": [
                    {
                        "concept_index": 3,
                        "terms": ["incremental score", "score recalculation"],
                        "operator": "OR",
                    }
                ],
            },
            {
                "name": "Supplier dependency tracking",
                "purpose": "Retrieve supplier matching using dependency relationships.",
                "concept_indices": [1, 2],
                "groups": [
                    {
                        "concept_index": 1,
                        "terms": ["supplier matching"],
                        "operator": "OR",
                    },
                    {
                        "concept_index": 2,
                        "terms": ["dependency relationship"],
                        "operator": "OR",
                    },
                ],
            },
            {
                "name": "Dependency tracking",
                "purpose": "Retrieve dependency tracking mechanisms independently.",
                "concept_indices": [2],
                "groups": [
                    {
                        "concept_index": 2,
                        "terms": ["dependency graph"],
                        "operator": "OR",
                    }
                ],
            },
            {
                "name": "Contractor matching",
                "purpose": "Retrieve contractor matching mechanisms independently.",
                "concept_indices": [1],
                "groups": [
                    {
                        "concept_index": 1,
                        "terms": ["contractor matching"],
                        "operator": "OR",
                    }
                ],
            },
        ]
    }


def test_normalize_query_plan_accepts_valid_plan(concepts, valid_plan):
    result = normalize_query_plan(valid_plan, concepts)

    assert len(result["queries"]) == 5
    assert result["queries"][0]["concept_indices"] == [1, 2]
    assert result["queries"][0]["groups"][0]["operator"] == "OR"


def test_normalize_query_plan_rejects_unapproved_term(concepts, valid_plan):
    valid_plan["queries"][0]["groups"][0]["terms"][0] = "invented terminology"

    with pytest.raises(ValueError, match="unapproved term"):
        normalize_query_plan(valid_plan, concepts)


def test_normalize_query_plan_rejects_invalid_concept_index(concepts, valid_plan):
    valid_plan["queries"][0]["groups"][0]["concept_index"] = 99

    with pytest.raises(ValueError, match="outside the approved concepts"):
        normalize_query_plan(valid_plan, concepts)


def test_normalize_query_plan_rejects_index_group_mismatch(concepts, valid_plan):
    valid_plan["queries"][0]["concept_indices"] = [1]

    with pytest.raises(ValueError, match="must match the groups"):
        normalize_query_plan(valid_plan, concepts)


def test_normalize_query_plan_requires_three_to_six_queries(concepts, valid_plan):
    valid_plan["queries"] = valid_plan["queries"][:2]

    with pytest.raises(ValueError, match="between 3 and 6"):
        normalize_query_plan(valid_plan, concepts)


def test_generate_query_plan_calls_client_and_validates(concepts, valid_plan):
    class FakeClaudeClient:
        called = False

        def complete(self, system, user, max_tokens=None):
            self.called = True
            assert "INVENTION DESCRIPTION" in user
            assert "APPROVED CONCEPTS" in user
            return json.dumps(valid_plan)

    client = FakeClaudeClient()
    result = generate_query_plan(
        "A contractor evaluation system with incremental ranking.",
        concepts,
        client,
    )

    assert client.called
    assert len(result["queries"]) == 5


def test_generate_query_plan_rejects_empty_invention(concepts):
    class UnusedClient:
        def complete(self, **kwargs):
            raise AssertionError("Client should not be called")

    with pytest.raises(ValueError, match="cannot be empty"):
        generate_query_plan("  ", concepts, UnusedClient())


def test_normalize_query_plan_rejects_missing_critical_concept(
    concepts, valid_plan
):
    # Remove every query that includes critical concept 2.
    valid_plan["queries"] = [
        query
        for query in valid_plan["queries"]
        if 2 not in query["concept_indices"]
    ]

    # Retain the minimum of three queries without restoring concept 2.
    valid_plan["queries"].append({
        "name": "Alternative contractor search",
        "purpose": "Search contractor selection independently.",
        "concept_indices": [1],
        "groups": [
            {
                "concept_index": 1,
                "terms": ["contractor selection"],
                "operator": "OR",
            }
        ],
    })

    with pytest.raises(
        ValueError,
        match="Critical concepts missing from query plan",
    ):
        normalize_query_plan(valid_plan, concepts)


def test_normalize_query_plan_rejects_combination_only_critical_concept(
    concepts, valid_plan
):
    # Concept 2 remains in combination queries, but loses its standalone query.
    valid_plan["queries"] = [
        query
        for query in valid_plan["queries"]
        if query["name"] != "Dependency tracking"
    ]

    with pytest.raises(
        ValueError,
        match="independent single-concept query",
    ):
        normalize_query_plan(valid_plan, concepts)


def test_generate_query_plan_repairs_missing_critical_standalone_query(
    concepts, valid_plan
):
    from services.ai_search_strategy_service import (
        normalize_query_plan,
    )

    invalid_plan = json.loads(json.dumps(valid_plan))
    invalid_plan["queries"] = [
        query
        for query in invalid_plan["queries"]
        if query["name"] != "Dependency tracking"
    ]

    class FakeClaudeClient:
        calls = 0

        def complete(self, system, user, max_tokens=None):
            self.calls += 1
            return json.dumps(invalid_plan)

    client = FakeClaudeClient()
    result = generate_query_plan(
        "A contractor evaluation system with incremental ranking.",
        concepts,
        client,
    )

    # The deterministic repair should avoid a second Claude call.
    assert client.calls == 1

    # The repaired plan must still pass all validation rules.
    normalized = normalize_query_plan(result, concepts)
    critical_indices = {
        index
        for index, concept in enumerate(concepts, start=1)
        if concept["importance"] == "critical"
    }
    standalone_indices = {
        query["concept_indices"][0]
        for query in normalized["queries"]
        if len(query["concept_indices"]) == 1
    }

    assert critical_indices <= standalone_indices
