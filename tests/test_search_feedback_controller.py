
import pytest

from services.search_feedback_controller import run_search_feedback_loop


def make_review(decision):
    return {
        "decision": decision,
        "rationale": f"Test decision: {decision}",
    }


def test_accept_stops_after_first_run():
    searches = []

    def execute(plan):
        searches.append(plan)
        return [{"patent_id": f"P{len(searches)}"}]

    result = run_search_feedback_loop(
        initial_plan={"version": 1},
        execute_search=execute,
        review_results=lambda plan, patents: make_review("accept"),
        revise_plan=lambda plan, review: {"version": plan["version"] + 1},
    )

    assert result["decision"] == "accept"
    assert result["run_count"] == 1
    assert len(searches) == 1
    assert result["stop_reason"] == "accepted"


def test_revision_runs_at_most_three_times_and_preserves_history():
    searches = []

    def execute(plan):
        result = [{"patent_id": f"P{len(searches) + 1}"}]
        searches.append({"plan": plan, "result": result})
        return result

    def revise(plan, review):
        return {"version": plan["version"] + 1}

    result = run_search_feedback_loop(
        initial_plan={"version": 1},
        execute_search=execute,
        review_results=lambda plan, patents: make_review("revise"),
        revise_plan=revise,
    )

    assert result["run_count"] == 3
    assert len(searches) == 3
    assert result["decision"] == "revise"
    assert result["stop_reason"] == "run_limit_reached"
    assert [entry["run_number"] for entry in result["history"]] == [1, 2, 3]
    assert [entry["plan"]["version"] for entry in result["history"]] == [1, 2, 3]
    assert [
        entry["results"][0]["patent_id"] for entry in result["history"]
    ] == ["P1", "P2", "P3"]


def test_insufficient_evidence_stops_without_revision():
    revised = []

    result = run_search_feedback_loop(
        initial_plan={"version": 1},
        execute_search=lambda plan: [{"patent_id": "P1"}],
        review_results=lambda plan, patents: make_review("insufficient_evidence"),
        revise_plan=lambda plan, review: revised.append(True),
    )

    assert result["run_count"] == 1
    assert result["stop_reason"] == "insufficient_evidence"
    assert revised == []


def test_empty_results_trigger_revision_and_second_search():
    plans = []
    revisions = []

    def execute(plan):
        plans.append(plan)
        if len(plans) == 1:
            return []
        return [{"patent_id": "P1"}]

    def review(plan, patents):
        if not patents:
            return {
                "decision": "insufficient_evidence",
                "rationale": "The first query plan returned no records.",
                "coverage_gaps": [],
                "queries_to_revise": [],
            }
        return make_review("accept")

    def revise(plan, feedback):
        revisions.append(feedback)
        return {"version": 2}

    result = run_search_feedback_loop(
        initial_plan={"version": 1},
        execute_search=execute,
        review_results=review,
        revise_plan=revise,
        max_runs=3,
    )

    assert result["decision"] == "accept"
    assert result["run_count"] == 2
    assert result["stop_reason"] == "accepted"
    assert [plan["version"] for plan in plans] == [1, 2]
    assert len(revisions) == 1
    assert result["history"][0]["results"] == []
    assert result["history"][1]["results"] == [{"patent_id": "P1"}]


def test_revision_failure_stops_safely():
    result = run_search_feedback_loop(
        initial_plan={"version": 1},
        execute_search=lambda plan: [{"patent_id": "P1"}],
        review_results=lambda plan, patents: make_review("revise"),
        revise_plan=lambda plan, review: {},
    )

    assert result["run_count"] == 1
    assert result["decision"] == "insufficient_evidence"
    assert result["stop_reason"] == "revision_failed"


@pytest.mark.parametrize("limit", [0, 4, -1])
def test_run_limit_cannot_be_disabled_or_increased(limit):
    with pytest.raises(ValueError, match="max_runs must be between"):
        run_search_feedback_loop(
            initial_plan={"version": 1},
            execute_search=lambda plan: [],
            review_results=lambda plan, patents: make_review("accept"),
            revise_plan=lambda plan, review: {"version": 2},
            max_runs=limit,
        )


def test_rejects_invalid_review_decision():
    with pytest.raises(ValueError, match="Unsupported review decision"):
        run_search_feedback_loop(
            initial_plan={"version": 1},
            execute_search=lambda plan: [],
            review_results=lambda plan, patents: {"decision": "keep searching"},
            revise_plan=lambda plan, review: {"version": 2},
        )
