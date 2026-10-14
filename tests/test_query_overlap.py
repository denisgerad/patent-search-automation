import unittest

from services.retrieval_diagnostics_service import (
    calculate_pairwise_query_overlap,
)


class TestPairwiseQueryOverlap(unittest.TestCase):
    def test_identical_family_sets_have_full_overlap(self):
        queries = [
            (1, "query A", {"F1", "F2"}),
            (2, "query B", {"F1", "F2"}),
        ]

        result = calculate_pairwise_query_overlap(queries)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["Shared families"], 2)
        self.assertEqual(result[0]["Jaccard overlap (%)"], 100.0)
        self.assertEqual(result[0]["Unique to A"], 0)
        self.assertEqual(result[0]["Unique to B"], 0)

    def test_disjoint_family_sets_have_zero_overlap(self):
        queries = [
            (1, "query A", {"F1", "F2"}),
            (2, "query B", {"F3", "F4"}),
        ]

        result = calculate_pairwise_query_overlap(queries)

        self.assertEqual(result[0]["Shared families"], 0)
        self.assertEqual(result[0]["Jaccard overlap (%)"], 0.0)
        self.assertEqual(result[0]["Unique to A"], 2)
        self.assertEqual(result[0]["Unique to B"], 2)

    def test_partial_overlap_uses_jaccard_percentage(self):
        queries = [
            (1, "query A", {"F1", "F2", "F3"}),
            (2, "query B", {"F2", "F3", "F4"}),
        ]

        result = calculate_pairwise_query_overlap(queries)

        self.assertEqual(result[0]["Shared families"], 2)
        self.assertEqual(result[0]["Jaccard overlap (%)"], 50.0)
        self.assertEqual(result[0]["Unique to A"], 1)
        self.assertEqual(result[0]["Unique to B"], 1)

    def test_empty_family_sets_do_not_divide_by_zero(self):
        queries = [
            (1, "query A", set()),
            (2, "query B", set()),
        ]

        result = calculate_pairwise_query_overlap(queries)

        self.assertEqual(result[0]["Jaccard overlap (%)"], 0.0)
        self.assertEqual(result[0]["Shared families"], 0)

    def test_missing_family_ids_are_not_counted(self):
        queries = [
            (1, "query A", {"F1"}),
            (2, "query B", set()),
        ]

        result = calculate_pairwise_query_overlap(queries)

        self.assertEqual(result[0]["Families A"], 1)
        self.assertEqual(result[0]["Families B"], 0)
        self.assertEqual(result[0]["Jaccard overlap (%)"], 0.0)

    def test_three_queries_produce_three_pairs(self):
        queries = [
            (1, "query A", {"F1"}),
            (2, "query B", {"F1", "F2"}),
            (3, "query C", {"F3"}),
        ]

        result = calculate_pairwise_query_overlap(queries)

        self.assertEqual(len(result), 3)

    def test_single_query_produces_no_pairs(self):
        queries = [(1, "query A", {"F1"})]

        self.assertEqual(
            calculate_pairwise_query_overlap(queries),
            [],
        )


if __name__ == "__main__":
    unittest.main()
