import unittest

from services.search_terminology import add_terminology_variants


class TestSearchTerminology(unittest.TestCase):
    def test_original_terms_are_preserved(self):
        original = ["fraud score", "risk score"]

        result = add_terminology_variants(
            "Dynamic fraud scoring", original
        )

        self.assertEqual(result[:2], original)
        self.assertIn("transaction risk rating", result)

    def test_duplicate_terms_are_removed(self):
        result = add_terminology_variants(
            "Dynamic fraud scoring",
            ["fraud score", "transaction risk rating", "fraud score"],
        )

        self.assertEqual(
            len([t for t in result if t.casefold() == "fraud score"]),
            1,
        )
        self.assertEqual(
            len([
                t for t in result
                if t.casefold() == "transaction risk rating"
            ]),
            1,
        )

    def test_unrecognized_concept_is_unchanged(self):
        original = ["customer transaction"]

        self.assertEqual(
            add_terminology_variants("Unrelated concept", original),
            original,
        )

    def test_variants_are_concept_specific(self):
        device_terms = add_terminology_variants(
            "Device characteristics fingerprinting", []
        )
        threshold_terms = add_terminology_variants(
            "Adaptive detection threshold", []
        )

        self.assertIn("device profiling", device_terms)
        self.assertNotIn("device profiling", threshold_terms)
        self.assertIn("adaptive decision threshold", threshold_terms)


if __name__ == "__main__":
    unittest.main()
