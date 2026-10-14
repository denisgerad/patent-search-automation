import unittest

from models.schemas import PatentRecord, SearchConcept, SearchStrategy
from services.retrieval_diagnostics_service import (
    calculate_retrieval_concept_coverage,
)


def make_strategy():
    return SearchStrategy(
        original_input="Context-aware fraud detection",
        concepts=[
            SearchConcept(
                name="Fraud score",
                importance="critical",
                terms=["fraud score", "risk score"],
            ),
            SearchConcept(
                name="Adaptive threshold",
                importance="critical",
                terms=["adaptive threshold", "dynamic threshold"],
            ),
            SearchConcept(
                name="Device profiling",
                importance="important",
                terms=["device fingerprinting"],
            ),
            SearchConcept(
                name="Historical patterns",
                importance="supporting",
                terms=["historical pattern learning"],
            ),
        ],
    )


class TestRetrievalDiagnosticsService(unittest.TestCase):
    def test_coverage_counts_only_critical_concepts(self):
        patents = [
            PatentRecord(
                patent_id="P1",
                patent_title="Fraud score with adaptive threshold",
                patent_abstract="Uses a risk score.",
            ),
            PatentRecord(
                patent_id="P2",
                patent_title="Dynamic threshold for risk evaluation",
                patent_abstract="No fraud score terminology.",
            ),
            PatentRecord(
                patent_id="P3",
                patent_title="Unrelated transaction processing",
            ),
        ]

        result = calculate_retrieval_concept_coverage(
            patents,
            make_strategy(),
        )

        self.assertEqual(result["total_records"], 3)
        self.assertEqual(result["critical_concept_count"], 2)
        self.assertEqual(
            result["records_with_any_critical_match"], 2
        )

        coverage = {
            item["concept"]: item
            for item in result["per_concept"]
        }

        self.assertEqual(
            coverage["Fraud score"]["matched_records"], 1
        )
        self.assertEqual(
            coverage["Adaptive threshold"]["matched_records"], 2
        )
        self.assertNotIn(
            "Device profiling", coverage
        )
        self.assertNotIn(
            "Historical patterns", coverage
        )

    def test_unmatched_critical_concept_is_reported(self):
        strategy = SearchStrategy(
            original_input="Test",
            concepts=[
                SearchConcept(
                    name="Critical mechanism",
                    importance="critical",
                    terms=["quantum calibration"],
                ),
            ],
        )
        patents = [
            PatentRecord(
                patent_id="P1",
                patent_title="Ordinary signal processing",
            )
        ]

        result = calculate_retrieval_concept_coverage(
            patents,
            strategy,
        )

        self.assertEqual(
            result["unmatched_concepts"], ["Critical mechanism"]
        )

    def test_empty_results_return_zero_coverage(self):
        result = calculate_retrieval_concept_coverage(
            [],
            make_strategy(),
        )

        self.assertEqual(result["total_records"], 0)
        self.assertEqual(
            result["records_with_any_critical_match"], 0
        )
        self.assertTrue(
            all(
                item["coverage_pct"] == 0
                for item in result["per_concept"]
            )
        )

    def test_missing_strategy_has_no_critical_concepts(self):
        result = calculate_retrieval_concept_coverage([], None)

        self.assertEqual(result["critical_concept_count"], 0)
        self.assertEqual(result["per_concept"], [])


if __name__ == "__main__":
    unittest.main()
