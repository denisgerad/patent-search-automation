import unittest

from models.schemas import SearchConcept, SearchStrategy
from services.search_strategy_service import (
    build_boolean_search,
    build_core_pair_queries,
    build_queries_for_mode,
    build_search_queries,
)


def make_strategy():
    return SearchStrategy(
        original_input="Context-aware fraud detection",
        concepts=[
            SearchConcept(
                name="Fraud score",
                importance="critical",
                terms=["fraud score", "risk score"],
                operator="OR",
            ),
            SearchConcept(
                name="Adaptive threshold",
                importance="critical",
                terms=["adaptive threshold", "dynamic threshold"],
                operator="OR",
            ),
            SearchConcept(
                name="Behavioral analysis",
                importance="critical",
                terms=["behavioral profiling"],
                operator="OR",
            ),
            SearchConcept(
                name="Transaction information",
                importance="important",
                terms=["transaction information"],
                operator="OR",
            ),
            SearchConcept(
                name="Historical patterns",
                importance="supporting",
                terms=["historical pattern learning"],
                operator="OR",
            ),
        ],
    )


class TestSearchStrategyService(unittest.TestCase):
    def test_individual_queries_include_only_critical_concepts(self):
        queries = build_search_queries(make_strategy())

        self.assertEqual(len(queries), 3)
        self.assertIn('"fraud score"', queries[0])
        self.assertIn('"adaptive threshold"', queries[1])
        self.assertIn('"behavioral profiling"', queries[2])
        self.assertFalse(
            any("transaction information" in q for q in queries)
        )
        self.assertFalse(
            any("historical pattern learning" in q for q in queries)
        )

    def test_pair_queries_cover_each_critical_concept_pair(self):
        queries = build_core_pair_queries(make_strategy())

        self.assertEqual(len(queries), 3)
        self.assertTrue(
            any('"fraud score"' in q and '"adaptive threshold"' in q
                for q in queries)
        )
        self.assertTrue(
            any('"fraud score"' in q and '"behavioral profiling"' in q
                for q in queries)
        )
        self.assertTrue(
            any('"adaptive threshold"' in q
                and '"behavioral profiling"' in q for q in queries)
        )

    def test_strict_boolean_search_requires_all_critical_concepts(self):
        query = build_boolean_search(make_strategy())

        self.assertIn(" AND ", query)
        self.assertIn('"fraud score"', query)
        self.assertIn('"adaptive threshold"', query)
        self.assertIn('"behavioral profiling"', query)
        self.assertNotIn('"transaction information"', query)
        self.assertNotIn('"historical pattern learning"', query)

    def test_broad_mode_runs_individual_critical_queries(self):
        queries = build_queries_for_mode(make_strategy(), "Broad")
        self.assertEqual(len(queries), 3)

    def test_balanced_mode_runs_individual_and_pair_queries(self):
        queries = build_queries_for_mode(make_strategy(), "Balanced")
        self.assertEqual(len(queries), 6)

    def test_strict_mode_runs_one_query_requiring_all_critical_concepts(self):
        queries = build_queries_for_mode(make_strategy(), "Strict")
        self.assertEqual(len(queries), 1)

        query = queries[0]
        self.assertIn(" AND ", query)
        self.assertIn('"fraud score"', query)
        self.assertIn('"adaptive threshold"', query)
        self.assertIn('"behavioral profiling"', query)

    def test_mode_names_are_case_insensitive(self):
        self.assertEqual(
            build_queries_for_mode(make_strategy(), "broad"),
            build_queries_for_mode(make_strategy(), "Broad"),
        )

    def test_individual_queries_return_empty_without_critical_concepts(self):
        strategy = SearchStrategy(
            original_input="Test",
            concepts=[
                SearchConcept(
                    name="Supporting concept",
                    importance="supporting",
                    terms=["supporting term"],
                )
            ],
        )

        self.assertEqual(build_search_queries(strategy), [])
        self.assertEqual(build_core_pair_queries(strategy), [])
        self.assertEqual(build_boolean_search(strategy), "")


if __name__ == "__main__":
    unittest.main()
