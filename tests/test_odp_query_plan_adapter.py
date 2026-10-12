
import pytest

from services.search_strategy_service import build_odp_queries_from_plan


def test_builds_boolean_query_from_plan():
    plan = {
        "queries": [
            {
                "name": "Matching and dependencies",
                "purpose": "Find matching systems with dependency tracking.",
                "concept_indices": [1, 2],
                "groups": [
                    {
                        "concept_index": 1,
                        "terms": ["contractor matching", "supplier matching"],
                        "operator": "OR",
                    },
                    {
                        "concept_index": 2,
                        "terms": ["dependency graph"],
                        "operator": "OR",
                    },
                ],
            }
        ]
    }

    result = build_odp_queries_from_plan(plan)

    assert len(result) == 1
    assert result[0]["name"] == "Matching and dependencies"
    assert result[0]["expression"] == (
        '("contractor matching" OR "supplier matching") '
        'AND ("dependency graph")'
    )


def test_builds_multiple_queries():
    plan = {
        "queries": [
            {
                "name": "First",
                "purpose": "First purpose",
                "groups": [
                    {
                        "concept_index": 1,
                        "terms": ["dependency graph"],
                        "operator": "OR",
                    }
                ],
            },
            {
                "name": "Second",
                "purpose": "Second purpose",
                "groups": [
                    {
                        "concept_index": 2,
                        "terms": ["incremental score"],
                        "operator": "OR",
                    }
                ],
            },
        ]
    }

    result = build_odp_queries_from_plan(plan)

    assert len(result) == 2
    assert result[1]["expression"] == '("incremental score")'


def test_rejects_empty_plan():
    with pytest.raises(ValueError, match="non-empty validated query plan"):
        build_odp_queries_from_plan({"queries": []})


def test_rejects_empty_term():
    plan = {
        "queries": [
            {
                "name": "Invalid",
                "purpose": "Invalid test",
                "groups": [
                    {
                        "concept_index": 1,
                        "terms": [""],
                        "operator": "OR",
                    }
                ],
            }
        ]
    }

    with pytest.raises(ValueError, match="cannot be empty"):
        build_odp_queries_from_plan(plan)
