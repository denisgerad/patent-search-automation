import unittest

from models.schemas import PatentRecord


def get_text_status(patent):
    abstract = patent.patent_abstract or ""
    claims = patent.patent_claims or ""

    text_characters = len(abstract) + len(claims)

    if abstract.strip() and claims.strip():
        status = "full_text"
    elif abstract.strip() or claims.strip():
        status = "partial_text"
    else:
        status = "insufficient_text"

    return status, text_characters


class TestPatentTextStatus(unittest.TestCase):
    def test_full_text(self):
        patent = PatentRecord(
            patent_id="TEST001",
            patent_title="Test patent",
            patent_abstract="A transaction fraud detection system.",
            patent_claims="1. A method for detecting fraud.",
        )

        status, characters = get_text_status(patent)

        self.assertEqual(status, "full_text")
        self.assertGreater(characters, 0)

    def test_partial_text(self):
        patent = PatentRecord(
            patent_id="TEST002",
            patent_title="Test patent",
            patent_abstract="A transaction fraud detection system.",
        )

        status, characters = get_text_status(patent)

        self.assertEqual(status, "partial_text")
        self.assertGreater(characters, 0)

    def test_insufficient_text(self):
        patent = PatentRecord(
            patent_id="TEST003",
            patent_title="Test patent",
        )

        status, characters = get_text_status(patent)

        self.assertEqual(status, "insufficient_text")
        self.assertEqual(characters, 0)


if __name__ == "__main__":
    unittest.main()