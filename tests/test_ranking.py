
import unittest

from models.schemas import PatentRecord, SearchConcept, SearchStrategy
from services.ranking_service import (
    _calculate_importance_score,
    _apply_importance_score,
    _apply_importance_score_if_text_available,
)


def make_patent(title="", abstract="", patent_id="TEST001"):
    return PatentRecord(
        patent_id=patent_id,
        patent_title=title,
        patent_abstract=abstract,
    )


def make_strategy(concepts):
    return SearchStrategy(
        original_input="Context-aware fraud detection",
        concepts=concepts,
    )


class TestImportanceScoring(unittest.TestCase):
    def test_exact_critical_concept_match(self):
        patent = make_patent(
            title="Dynamic fraud scoring",
            abstract="A system generates a fraud score for each transaction.",
        )
        strategy = make_strategy([
            SearchConcept(
                name="Dynamic fraud scoring",
                importance="critical",
                terms=["fraud score", "risk score"],
            ),
        ])

        score, hits = _calculate_importance_score(patent, strategy)

        self.assertEqual(score, 1.0)
        self.assertTrue(hits["Dynamic fraud scoring"])

    def test_no_literal_match_returns_zero(self):
        patent = make_patent(
            title="Transaction anomaly analysis",
            abstract="The system evaluates customer activity against a baseline.",
        )
        strategy = make_strategy([
            SearchConcept(
                name="Adaptive detection threshold",
                importance="important",
                terms=[
                    "adaptive threshold",
                    "dynamic threshold",
                    "variable detection threshold",
                ],
            ),
        ])

        score, hits = _calculate_importance_score(patent, strategy)

        self.assertEqual(score, 0.0)
        self.assertFalse(hits["Adaptive detection threshold"])

    def test_weighted_coverage(self):
        patent = make_patent(
            abstract="The system generates a fraud score.",
        )
        strategy = make_strategy([
            SearchConcept(
                name="Dynamic fraud scoring",
                importance="critical",
                terms=["fraud score"],
            ),
            SearchConcept(
                name="Device profiling",
                importance="important",
                terms=["device fingerprinting"],
            ),
        ])

        score, hits = _calculate_importance_score(patent, strategy)

        # One critical concept matched: 15 / (15 + 5).
        self.assertAlmostEqual(score, 15 / 20)
        self.assertTrue(hits["Dynamic fraud scoring"])
        self.assertFalse(hits["Device profiling"])

    def test_critical_concept_penalty(self):
        strategy = make_strategy([
            SearchConcept(
                name="Dynamic fraud scoring",
                importance="critical",
                terms=["fraud score"],
            ),
            SearchConcept(
                name="Context-aware detection",
                importance="critical",
                terms=["context-aware fraud detection"],
            ),
        ])

        score_with_missing_critical = _apply_importance_score(
            hybrid=0.8,
            importance_score=0.5,
            importance_hits={
                "Dynamic fraud scoring": True,
                "Context-aware detection": False,
            },
            search_strategy=strategy,
        )

        # Base multiplier: 0.60 + (0.40 * 0.5) = 0.80.
        # Critical-missing penalty: 0.80 * 0.75 = 0.60.
        self.assertAlmostEqual(score_with_missing_critical, 0.8 * 0.6)

    def test_no_strategy_preserves_hybrid_score(self):
        self.assertEqual(
            _apply_importance_score(
                hybrid=0.8,
                importance_score=0.0,
                importance_hits={},
                search_strategy=None,
            ),
            0.8,
        )

    def test_matching_normalizes_hyphens_and_case(self):
        patent = make_patent(
            title="CONTEXT AWARE FRAUD DETECTION",
        )
        strategy = make_strategy([
            SearchConcept(
                name="Context-aware detection",
                importance="critical",
                terms=["context-aware fraud detection"],
            ),
        ])

        score, hits = _calculate_importance_score(
            patent,
            strategy,
        )

        self.assertEqual(score, 1.0)
        self.assertTrue(hits["Context-aware detection"])

    def test_related_terminology_needs_explicit_variant(self):
        patent = make_patent(
            title="Transaction Risk Evaluation",
            abstract=(
                "The system compares customer activity against "
                "an established personal baseline."
            ),
        )
        strategy = make_strategy([
            SearchConcept(
                name="Adaptive detection threshold",
                importance="important",
                terms=[
                    "adaptive threshold",
                    "dynamic threshold",
                    "variable detection threshold",
                ],
            ),
        ])

        score, hits = _calculate_importance_score(
            patent,
            strategy,
        )

        self.assertEqual(score, 0.0)
        self.assertFalse(hits["Adaptive detection threshold"])

    def test_explicit_terminology_variant_matches_patent(self):
        from services.search_terminology import add_terminology_variants

        patent = make_patent(
            title="Transaction Risk Evaluation",
            abstract=(
                "The system compares customer activity against "
                "an established personal baseline and applies a "
                "risk based threshold to the comparison."
            ),
        )

        original_terms = [
            "adaptive threshold",
            "dynamic threshold",
            "variable detection threshold",
        ]

        expanded_terms = add_terminology_variants(
            "Adaptive detection threshold",
            original_terms,
        )

        strategy = make_strategy([
            SearchConcept(
                name="Adaptive detection threshold",
                importance="important",
                terms=expanded_terms,
            ),
        ])

        score, hits = _calculate_importance_score(
            patent,
            strategy,
        )

        self.assertGreater(score, 0.0)
        self.assertTrue(
            hits["Adaptive detection threshold"]
        )

    def test_title_only_patent_preserves_hybrid_score(self):
        patent = make_patent(
            title="Adaptive transaction fraud detection",
        )
        strategy = make_strategy([
            SearchConcept(
                name="Dynamic fraud scoring",
                importance="critical",
                terms=["fraud score"],
            ),
        ])

        result = _apply_importance_score_if_text_available(
            patent=patent,
            hybrid=0.8,
            importance_score=0.0,
            importance_hits={"Dynamic fraud scoring": False},
            search_strategy=strategy,
        )

        self.assertEqual(result, 0.8)

    def test_patent_with_text_keeps_existing_importance_adjustment(self):
        patent = make_patent(
            title="Transaction analysis",
            abstract="The system analyses transaction records.",
        )
        strategy = make_strategy([
            SearchConcept(
                name="Dynamic fraud scoring",
                importance="critical",
                terms=["fraud score"],
            ),
        ])

        result = _apply_importance_score_if_text_available(
            patent=patent,
            hybrid=0.8,
            importance_score=0.0,
            importance_hits={"Dynamic fraud scoring": False},
            search_strategy=strategy,
        )

        self.assertAlmostEqual(result, 0.8 * 0.60 * 0.75)


if __name__ == "__main__":
    unittest.main()
