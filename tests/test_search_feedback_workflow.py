from types import SimpleNamespace

import pytest

from services import search_feedback_workflow as workflow


def patent(patent_id, title="Test patent", abstract="Test abstract", claims="Test claim"):
    return SimpleNamespace(
        patent_id=patent_id,
        patent_title=title,
        patent_abstract=abstract,
        patent_claims=claims,
        patent_date="2025-01-01",
    )


def test_patent_mapping_preserves_evidence_fields():
    result = workflow._patent_to_dict(
        patent("P1", "A title", "An abstract", "A claim")
    )

    assert result["patent_id"] == "P1"
    assert result["title"] == "A title"
    assert result["abstract"] == "An abstract"
    assert result["claims"] == "A claim"
    assert result["search_queries"] == []


def test_execute_plan_deduplicates_and_preserves_query_provenance(monkeypatch):
    monkeypatch.setattr(
        workflow,
        "build_odp_queries_from_plan",
        lambda plan: [
            {"name": "Query A", "purpose": "First query", "expression": "A"},
            {"name": "Query B", "purpose": "Second query", "expression": "B"},
        ],
    )

    def fake_search(expression, limit, offset):
        assert limit == 25
        assert offset == 0

        if expression == "A":
            return [patent("P1"), patent("P2")]
        return [patent("P1"), patent("P3")]

    monkeypatch.setattr(workflow, "search_patents_by_strategy", fake_search)

    results = workflow._execute_plan({"queries": []})
    by_id = {item["patent_id"]: item for item in results}

    assert set(by_id) == {"P1", "P2", "P3"}
    assert by_id["P1"]["search_queries"] == ["Query A", "Query B"]
    assert by_id["P2"]["search_queries"] == ["Query A"]
    assert by_id["P3"]["search_queries"] == ["Query B"]


def test_execute_plan_propagates_search_api_failure(monkeypatch):
    monkeypatch.setattr(
        workflow,
        "build_odp_queries_from_plan",
        lambda plan: [
            {"name": "Query A", "purpose": "Test", "expression": "A"}
        ],
    )

    def fail_search(*args, **kwargs):
        raise RuntimeError("USPTO unavailable")

    monkeypatch.setattr(workflow, "search_patents_by_strategy", fail_search)

    with pytest.raises(RuntimeError, match="USPTO unavailable"):
        workflow._execute_plan({"queries": []})


def test_empty_results_trigger_revision_without_calling_reviewer(monkeypatch):
    plan = {
        "queries": [
            {
                "name": "Query A",
                "purpose": "Initial precision search",
                "concept_indices": [1],
                "groups": [
                    {
                        "concept_index": 1,
                        "terms": ["sensor"],
                        "operator": "OR",
                    }
                ],
            }
        ]
    }

    revised_plan = {
        "queries": [
            {
                "name": "Query B",
                "purpose": "Broaden search after zero results",
                "concept_indices": [1],
                "groups": [
                    {
                        "concept_index": 1,
                        "terms": ["detector"],
                        "operator": "OR",
                    }
                ],
            }
        ]
    }

    executed_plans = []
    revision_calls = []

    monkeypatch.setattr(
        workflow,
        "generate_query_plan",
        lambda *args: plan,
    )

    def fake_execute(current_plan):
        executed_plans.append(current_plan)
        return []

    monkeypatch.setattr(workflow, "_execute_plan", fake_execute)

    def fake_revise(**kwargs):
        revision_calls.append(kwargs)
        return revised_plan

    monkeypatch.setattr(workflow, "_revise_plan", fake_revise)

    def unexpected_review(**kwargs):
        pytest.fail("Claude reviewer must not be called for empty results")

    monkeypatch.setattr(
        workflow,
        "review_search_results",
        unexpected_review,
    )

    result = workflow.run_patent_search_feedback_workflow(
        invention="Test invention",
        approved_concepts=[
            {
                "primary_term": "sensor",
                "alternative_terms": ["detector"],
            }
        ],
        client=object(),
        max_runs=2,
    )

    assert len(revision_calls) == 1
    assert len(executed_plans) == 2
    assert executed_plans[0] == plan
    assert executed_plans[1] == revised_plan

    assert result["run_count"] == 2
    assert result["max_runs"] == 2
    assert result["decision"] == "insufficient_evidence"
    assert result["stop_reason"] == "insufficient_evidence"
    assert result["history"][0]["results"] == []
    assert result["history"][1]["results"] == []


def test_api_failure_propagates_instead_of_becoming_empty_results(monkeypatch):
    plan = {"queries": [{"name": "Query A"}]}
    monkeypatch.setattr(workflow, "generate_query_plan", lambda *args: plan)

    def fail_search(plan):
        raise RuntimeError("USPTO unavailable")

    monkeypatch.setattr(workflow, "_execute_plan", fail_search)

    with pytest.raises(RuntimeError, match="USPTO unavailable"):
        workflow.run_patent_search_feedback_workflow(
            invention="Test invention",
            approved_concepts=[{"primary_term": "sensor", "alternative_terms": ["detector"]}],
            client=object(),
        )


def test_workflow_rejects_more_than_three_runs():
    with pytest.raises(ValueError, match="max_runs must be between 1 and 3"):
        workflow.run_patent_search_feedback_workflow(
            invention="Test invention",
            approved_concepts=[{"primary_term": "sensor", "alternative_terms": ["detector"]}],
            client=object(),
            max_runs=4,
        )


def test_review_receives_at_most_twenty_patents(monkeypatch):
    plan = {"queries": [{"name": "Query A"}]}
    concepts = [{"primary_term": "sensor", "alternative_terms": ["detector"]}]
    results = [
        {"patent_id": f"P{i}", "title": f"Patent {i}", "search_queries": ["Query A"]}
        for i in range(25)
    ]

    monkeypatch.setattr(workflow, "generate_query_plan", lambda *args: plan)
    monkeypatch.setattr(workflow, "_execute_plan", lambda current_plan: results)
    monkeypatch.setattr(
        workflow,
        "build_odp_queries_from_plan",
        lambda current_plan: [
            {"name": "Query A", "purpose": "Test", "expression": "A"}
        ],
    )

    observed = {}

    def fake_review(**kwargs):
        observed["patents"] = kwargs["patents"]
        return {
            "decision": "accept",
            "rationale": "Mocked acceptance",
            "coverage_gaps": [],
            "queries_to_revise": [],
            "patent_reviews": [],
        }

    monkeypatch.setattr(workflow, "review_search_results", fake_review)
    monkeypatch.setattr(
        workflow,
        "_rank_candidates",
        lambda invention, patents, concepts, limit=20: patents[:limit],
    )

    result = workflow.run_patent_search_feedback_workflow(
        invention="Test invention",
        approved_concepts=concepts,
        client=object(),
    )

    assert len(observed["patents"]) <= 20
    assert result["run_count"] == 1
    assert result["decision"] == "accept"


def test_rank_candidates_returns_fewer_than_limit_when_ranker_has_fewer_results(monkeypatch):
    patents = [
        {"patent_id": "P1", "title": "Patent 1", "abstract": "Abstract 1", "claims": "Claim 1"},
        {"patent_id": "P2", "title": "Patent 2", "abstract": "Abstract 2", "claims": "Claim 2"},
    ]
    concepts = [{"primary_term": "sensor", "alternative_terms": ["detector"]}]

    class DummyModel:
        def __init__(self):
            pass

    def fake_embed_patents(records, model):
        return [[0.1, 0.2], [0.3, 0.4]]

    def fake_embed_query(query, model):
        return [0.5, 0.6]

    def fake_rank(**kwargs):
        return [
            SimpleNamespace(
                patent=SimpleNamespace(
                    patent_id="P1",
                    patent_title="Patent 1",
                    patent_abstract="Abstract 1",
                    patent_claims="Claim 1",
                    patent_date="2025-01-01",
                    search_paths=["Query A"],
                ),
                cosine_score=0.9,
                bm25_score=0.8,
                hybrid_score=0.85,
                coverage=0.7,
                importance_score=0.6,
            )
        ]

    monkeypatch.setattr("models.embedding_model.EmbeddingModel", DummyModel)
    monkeypatch.setattr("services.embedding_service.embed_patents", fake_embed_patents)
    monkeypatch.setattr("services.embedding_service.embed_query", fake_embed_query)
    monkeypatch.setattr("services.ranking_service.rank", fake_rank)

    result = workflow._rank_candidates(
        invention="Test invention",
        patents=patents,
        concepts=concepts,
        limit=20,
    )

    assert len(result) == 1
    assert result[0]["patent_id"] == "P1"


def test_empty_first_run_then_revised_search_recovers_patent(monkeypatch):
    initial_plan = {
        "queries": [
            {
                "name": "Precision",
                "purpose": "Initial search",
                "concept_indices": [1],
                "groups": [
                    {
                        "concept_index": 1,
                        "terms": ["sensor"],
                        "operator": "OR",
                    }
                ],
            }
        ]
    }
    revised_plan = {
        "queries": [
            {
                "name": "Broadened",
                "purpose": "Broaden after zero results",
                "concept_indices": [1],
                "groups": [
                    {
                        "concept_index": 1,
                        "terms": ["detector"],
                        "operator": "OR",
                    }
                ],
            }
        ]
    }

    candidate = {
        "patent_id": "US1234567",
        "patent_title": "Example sensor system",
        "patent_abstract": "An example detection system.",
        "search_queries": ["Broadened"],
    }

    executed_plans = []
    review_calls = []

    monkeypatch.setattr(
        workflow,
        "generate_query_plan",
        lambda *args: initial_plan,
    )

    def fake_execute(plan):
        executed_plans.append(plan)
        return [] if len(executed_plans) == 1 else [candidate]

    monkeypatch.setattr(workflow, "_execute_plan", fake_execute)

    monkeypatch.setattr(
        workflow,
        "_revise_plan",
        lambda **kwargs: revised_plan,
    )

    monkeypatch.setattr(
        workflow,
        "_rank_candidates",
        lambda invention, patents, concepts, limit=20: patents[:limit],
    )

    def fake_review(**kwargs):
        review_calls.append(kwargs)
        return {
            "decision": "accept",
            "rationale": "The retrieved evidence was accepted for this test.",
            "coverage_gaps": [],
            "query_revisions": [],
            "patent_reviews": [],
        }

    monkeypatch.setattr(
        workflow,
        "review_search_results",
        fake_review,
    )

    result = workflow.run_patent_search_feedback_workflow(
        invention="Test invention",
        approved_concepts=[
            {
                "primary_term": "sensor",
                "alternative_terms": ["detector"],
            }
        ],
        client=object(),
        max_runs=3,
    )

    assert len(executed_plans) == 2
    assert executed_plans[0] == initial_plan
    assert executed_plans[1] == revised_plan
    assert len(review_calls) == 1

    assert result["run_count"] == 2
    assert result["decision"] == "accept"
    assert result["history"][0]["results"] == []
    assert result["history"][1]["results"] == [candidate]


def test_empty_invention_and_missing_concepts_are_rejected():
    with pytest.raises(ValueError, match="Invention description cannot be empty"):
        workflow.run_patent_search_feedback_workflow(
            invention=" ",
            approved_concepts=[{"primary_term": "sensor"}],
            client=object(),
        )

    with pytest.raises(ValueError, match="Approved search concepts are required"):
        workflow.run_patent_search_feedback_workflow(
            invention="Test invention",
            approved_concepts=[],
            client=object(),
        )


def test_execute_plan_retries_unquoted_query_and_tracks_provenance(monkeypatch):
    monkeypatch.setattr(
        workflow,
        "build_odp_queries_from_plan",
        lambda plan: [
            {
                "name": "Dependency search",
                "purpose": "Find dependency mechanisms",
                "expression": '("dependency graph")',
            }
        ],
    )

    calls = []

    def fake_search(expression, limit, offset):
        calls.append(expression)
        if expression == '("dependency graph")':
            return []
        if expression == "(dependency graph)":
            return [patent("P1")]
        pytest.fail(f"Unexpected query expression: {expression}")

    monkeypatch.setattr(workflow, "search_patents_by_strategy", fake_search)

    results = workflow._execute_plan({"queries": []})

    assert calls == ['("dependency graph")', "(dependency graph)"]
    assert len(results) == 1
    assert "Dependency search (unquoted fallback)" in results[0]["search_queries"]


def test_execute_plan_does_not_retry_when_precise_query_succeeds(monkeypatch):
    monkeypatch.setattr(
        workflow,
        "build_odp_queries_from_plan",
        lambda plan: [
            {
                "name": "Dependency search",
                "purpose": "Find dependency mechanisms",
                "expression": '("dependency graph")',
            }
        ],
    )

    calls = []

    def fake_search(expression, limit, offset):
        calls.append(expression)
        return [patent("P1")]

    monkeypatch.setattr(workflow, "search_patents_by_strategy", fake_search)

    results = workflow._execute_plan({"queries": []})

    assert calls == ['("dependency graph")']
    assert "Dependency search" in results[0]["search_queries"]
    assert "Dependency search (unquoted fallback)" not in results[0]["search_queries"]


def test_execute_plan_keeps_empty_result_when_both_queries_miss(monkeypatch):
    monkeypatch.setattr(
        workflow,
        "build_odp_queries_from_plan",
        lambda plan: [
            {
                "name": "Dependency search",
                "purpose": "Find dependency mechanisms",
                "expression": '("dependency graph")',
            }
        ],
    )

    calls = []

    def fake_search(expression, limit, offset):
        calls.append(expression)
        return []

    monkeypatch.setattr(workflow, "search_patents_by_strategy", fake_search)

    results = workflow._execute_plan({"queries": []})

    assert calls == ['("dependency graph")', "(dependency graph)"]
    assert results == []


def test_ranked_empty_results_are_reported_as_insufficient_evidence(monkeypatch):
    plan = {
        "queries": [
            {
                "name": "Mechanism search",
                "purpose": "Find patents describing the mechanism",
                "concept_indices": [1],
                "groups": [
                    {
                        "concept_index": 1,
                        "terms": ["sensor"],
                        "operator": "OR",
                    }
                ],
            }
        ]
    }

    retrieved = [
        {
            "patent_id": "US1234567",
            "title": "Example patent",
            "abstract": "An example technical system.",
            "claims": "A method comprising...",
            "search_queries": ["Mechanism search"],
        }
    ]

    monkeypatch.setattr(
        workflow, "generate_query_plan", lambda *args: plan
    )
    monkeypatch.setattr(
        workflow, "_execute_plan", lambda current_plan: retrieved
    )
    monkeypatch.setattr(
        workflow,
        "build_odp_queries_from_plan",
        lambda current_plan: [
            {
                "name": "Mechanism search",
                "purpose": "Find patents describing the mechanism",
                "expression": '"sensor"',
            }
        ],
    )
    monkeypatch.setattr(
        workflow, "_rank_candidates", lambda *args, **kwargs: []
    )

    def unexpected_review(**kwargs):
        pytest.fail("Reviewer must not receive an empty candidate list")

    monkeypatch.setattr(
        workflow, "review_search_results", unexpected_review
    )

    result = workflow.run_patent_search_feedback_workflow(
        invention="Test invention",
        approved_concepts=[
            {
                "name": "Sensor",
                "primary_term": "sensor",
                "alternative_terms": ["detector"],
                "importance": "critical",
            }
        ],
        client=object(),
        max_runs=1,
    )

    assert result["decision"] == "insufficient_evidence"
    assert result["history"][0]["results"] == retrieved
    assert "ranking stage returned no candidates" in (
        result["history"][0]["review"]["rationale"]
    )


def test_review_validation_failure_is_reported_as_insufficient_evidence(
    monkeypatch,
):
    plan = {
        "queries": [
            {
                "name": "Mechanism search",
                "purpose": "Find patents describing the mechanism",
                "concept_indices": [1],
                "groups": [
                    {
                        "concept_index": 1,
                        "terms": ["sensor"],
                        "operator": "OR",
                    }
                ],
            }
        ]
    }

    retrieved = [
        {
            "patent_id": "US1234567",
            "title": "Example patent",
            "abstract": "An example technical system.",
            "claims": "A method comprising...",
            "search_queries": ["Mechanism search"],
        }
    ]

    monkeypatch.setattr(workflow, "generate_query_plan", lambda *args: plan)
    monkeypatch.setattr(workflow, "_execute_plan", lambda current_plan: retrieved)
    monkeypatch.setattr(
        workflow,
        "build_odp_queries_from_plan",
        lambda current_plan: [
            {
                "name": "Mechanism search",
                "purpose": "Find patents describing the mechanism",
                "expression": '"sensor"',
            }
        ],
    )
    monkeypatch.setattr(
        workflow,
        "_rank_candidates",
        lambda invention, patents, concepts, limit=8: retrieved[:1],
    )

    def fail_review(**kwargs):
        raise ValueError("unsupported excerpt match")

    monkeypatch.setattr(workflow, "review_search_results", fail_review)

    result = workflow.run_patent_search_feedback_workflow(
        invention="Test invention",
        approved_concepts=[
            {
                "name": "Sensor",
                "primary_term": "sensor",
                "alternative_terms": ["detector"],
                "importance": "critical",
            }
        ],
        client=object(),
        max_runs=1,
    )

    assert result["decision"] == "insufficient_evidence"
    assert result["history"]
    assert result["history"][-1]["review"]["decision"] == "insufficient_evidence"
    assert result["history"][-1]["review"]["patent_reviews"] == []
