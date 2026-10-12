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


def test_empty_results_return_insufficient_evidence_without_review(monkeypatch):
    plan = {
        "queries": [
            {
                "name": "Query A",
                "purpose": "Test query",
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
    monkeypatch.setattr(workflow, "generate_query_plan", lambda *args: plan)
    monkeypatch.setattr(workflow, "_execute_plan", lambda plan: [])

    def unexpected_review(**kwargs):
        pytest.fail("Claude reviewer must not be called for empty results")

    monkeypatch.setattr(workflow, "review_search_results", unexpected_review)

    result = workflow.run_patent_search_feedback_workflow(
        invention="Test invention",
        approved_concepts=[{"primary_term": "sensor", "alternative_terms": ["detector"]}],
        client=object(),
    )

    assert result["decision"] == "insufficient_evidence"
    assert result["stop_reason"] == "insufficient_evidence"
    assert result["run_count"] == 1
    assert "no patent records" in result["history"][0]["review"]["rationale"]


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

    result = workflow.run_patent_search_feedback_workflow(
        invention="Test invention",
        approved_concepts=concepts,
        client=object(),
    )

    assert len(observed["patents"]) == 20
    assert result["run_count"] == 1
    assert result["decision"] == "accept"


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
