
"""Control a bounded patent-search feedback loop."""

from __future__ import annotations

from typing import Any, Callable


MAX_SEARCH_RUNS = 3


def run_search_feedback_loop(
    initial_plan: dict[str, Any],
    execute_search: Callable[[dict[str, Any]], list[dict[str, Any]]],
    review_results: Callable[
        [dict[str, Any], list[dict[str, Any]]], dict[str, Any]
    ],
    revise_plan: Callable[
        [dict[str, Any], dict[str, Any]], dict[str, Any]
    ],
    max_runs: int = MAX_SEARCH_RUNS,
) -> dict[str, Any]:
    """
    Execute and review a search plan up to three times.

    Each run's plan, results and review are retained in history.
    The controller never creates a fourth run.
    """
    if not 1 <= max_runs <= MAX_SEARCH_RUNS:
        raise ValueError(f"max_runs must be between 1 and {MAX_SEARCH_RUNS}.")

    if not isinstance(initial_plan, dict) or not initial_plan:
        raise ValueError("initial_plan must be a non-empty dictionary.")

    current_plan = initial_plan
    history: list[dict[str, Any]] = []
    final_decision = "insufficient_evidence"
    stop_reason = "run_limit_reached"

    for run_number in range(1, max_runs + 1):
        # Search execution is a separate dependency, allowing deterministic tests.
        results = execute_search(current_plan)
        if not isinstance(results, list):
            raise TypeError("execute_search must return a list.")

        review = review_results(current_plan, results)
        if not isinstance(review, dict):
            raise TypeError("review_results must return a dictionary.")

        decision = review.get("decision")
        if decision not in {"accept", "revise", "insufficient_evidence"}:
            raise ValueError(f"Unsupported review decision: {decision!r}")

        # Store the exact result set for this run; do not overwrite earlier runs.
        history.append(
            {
                "run_number": run_number,
                "plan": current_plan,
                "results": results,
                "review": review,
            }
        )

        final_decision = decision

        if decision == "accept":
            stop_reason = "accepted"
            break

        if decision == "insufficient_evidence":
            stop_reason = "insufficient_evidence"
            break

        # A revision is only allowed if another run remains.
        if run_number == max_runs:
            stop_reason = "run_limit_reached"
            break

        next_plan = revise_plan(current_plan, review)
        if not isinstance(next_plan, dict) or not next_plan:
            stop_reason = "revision_failed"
            final_decision = "insufficient_evidence"
            break

        current_plan = next_plan

    return {
        "decision": final_decision,
        "stop_reason": stop_reason,
        "run_count": len(history),
        "max_runs": max_runs,
        "history": history,
        "results_by_run": [
            {
                "run_number": item["run_number"],
                "results": item["results"],
            }
            for item in history
        ],
    }
