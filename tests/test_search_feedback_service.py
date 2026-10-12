
import json

import pytest

from services.search_feedback_service import (
    normalize_search_review,
    review_search_results,
    select_top_patents,
)


@pytest.fixture
def patents():
    return [
        {
            "patent_id": "US1234567A1",
            "title": "Dependency-aware evaluation",
            "abstract": "An evaluation system tracks dependent states.",
        },
        {
            "patent_id": "US7654321A1",
            "title": "Incremental score calculation",
            "abstract": "Scores are recalculated after input changes.",
        },
    ]


@pytest.fixture
def queries():
    return [
        {
            "name": "Dependency search",
            "purpose": "Find dependency-aware evaluation systems.",
            "expression": '"dependency graph"',
        }
    ]


@pytest.fixture
def valid_review():
    return {
        "decision": "revise",
        "rationale": "The evidence covers dependencies but not contractor eligibility.",
        "coverage_gaps": ["Mandatory contractor eligibility gating"],
        "queries_to_revise": [
            {
                "query_name": "Dependency search",
                "reason": "The current query focuses on dependencies.",
                "suggested_focus": "Contractor matching and eligibility concepts",
            }
        ],
        "patent_reviews": [
            {
                "patent_id": "US1234567A1",
                "label": "partially_relevant",
                "matched_mechanisms": ["Dependency tracking"],
                "missing_mechanisms": ["Contractor eligibility gating"],
                "evidence": "The supplied abstract describes dependent evaluation states.",
            },
            {
                "patent_id": "US7654321A1",
                "label": "partially_relevant",
                "matched_mechanisms": ["Incremental score calculation"],
                "missing_mechanisms": ["Contractor matching"],
                "evidence": "The supplied abstract describes score recalculation after changes.",
            },
        ],
    }


def test_normalize_search_review_accepts_valid_review(
    patents, queries, valid_review
):
    result = normalize_search_review(
        valid_review, patents, {"Dependency search"}
    )

    assert result["decision"] == "revise"
    assert len(result["patent_reviews"]) == 2
    assert result["coverage_gaps"] == [
        "Mandatory contractor eligibility gating"
    ]


def test_rejects_unknown_patent_id(patents, queries, valid_review):
    valid_review["patent_reviews"][0]["patent_id"] = "UNKNOWN"

    with pytest.raises(ValueError, match="unknown patent ID"):
        normalize_search_review(
            valid_review, patents, {"Dependency search"}
        )


def test_rejects_missing_patent_review(patents, queries, valid_review):
    valid_review["patent_reviews"].pop()

    with pytest.raises(ValueError, match="missing supplied patents"):
        normalize_search_review(
            valid_review, patents, {"Dependency search"}
        )


def test_rejects_revision_of_unknown_query(patents, valid_review):
    valid_review["queries_to_revise"][0]["query_name"] = "Not an executed query"

    with pytest.raises(ValueError, match="unknown query"):
        normalize_search_review(
            valid_review, patents, {"Dependency search"}
        )


def test_revise_requires_actionable_gap(patents, valid_review):
    valid_review["coverage_gaps"] = []
    valid_review["queries_to_revise"] = []

    with pytest.raises(ValueError, match="actionable coverage gap"):
        normalize_search_review(
            valid_review, patents, {"Dependency search"}
        )


def test_review_search_results_uses_client(patents, queries, valid_review):
    class FakeClaudeClient:
        def complete(self, system, user, max_tokens=None):
            assert "PATENT RESULTS TO REVIEW" in user
            return json.dumps(valid_review)

    result = review_search_results(
        invention="Contractor matching with incremental scoring.",
        concepts=[{"primary_term": "contractor matching"}],
        queries=queries,
        patents=patents,
        client=FakeClaudeClient(),
    )

    assert result["decision"] == "revise"


def test_select_top_patents_limits_results(patents):
    assert len(select_top_patents(patents, limit=1)) == 1


def test_select_top_patents_rejects_invalid_limit(patents):
    with pytest.raises(ValueError, match="at least 1"):
        select_top_patents(patents, limit=0)
