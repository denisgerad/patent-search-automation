
import unittest
from unittest.mock import Mock, patch

from services.uspto_search_service import (
    _build_odp_query,
    search_patents_with_classification,
)


class TestODPClassificationQuery(unittest.TestCase):
    def test_cpc_requires_title_and_selected_class(self):
        result = _build_odp_query(
            "neural network",
            cpc_classifications=["G06N3/045"],
        )

        self.assertIn(
            'applicationMetaData.inventionTitle:"neural network" AND',
            result,
        )
        self.assertIn(
            "applicationMetaData.cpcClassificationBag:G06N3/045",
            result,
        )
        self.assertNotIn(" OR ", result)

    def test_multiple_cpc_codes_use_or(self):
        result = _build_odp_query(
            "neural network",
            cpc_classifications=["G06N3/045", "G06N3/08"],
        )

        self.assertIn(
            "(applicationMetaData.cpcClassificationBag:G06N3/045 "
            "OR applicationMetaData.cpcClassificationBag:G06N3/08)",
            result,
        )
        self.assertIn(' AND (', result)

    def test_cpc_and_uspc_groups_can_use_or(self):
        result = _build_odp_query(
            "neural network",
            cpc_classifications=["G06N3/045"],
            uspc_classes=["706"],
            classification_operator="OR",
        )

        self.assertIn(
            'applicationMetaData.inventionTitle:"neural network" AND (',
            result,
        )
        self.assertIn(
            "(applicationMetaData.cpcClassificationBag:G06N3/045) OR "
            "(applicationMetaData.class:706)",
            result,
        )

    def test_empty_classification_keeps_title_search(self):
        self.assertEqual(
            _build_odp_query("neural network"),
            'applicationMetaData.inventionTitle:"neural network"',
        )

    def test_invalid_operator_is_rejected(self):
        with self.assertRaises(ValueError):
            _build_odp_query(
                "neural network",
                cpc_classifications=["G06N3/045"],
                classification_operator="XOR",
            )


class TestClassifiedSearchPayload(unittest.TestCase):
    @patch("services.uspto_search_service.requests.get")
    @patch("services.uspto_search_service._build_headers")
    def test_classification_filter_is_sent_in_get_params(
        self, mock_headers, mock_get
    ):
        mock_headers.return_value = {}

        response = Mock()
        response.status_code = 200
        response.json.return_value = {
            "count": 0,
            "patentFileWrapperDataBag": [],
        }
        mock_get.return_value = response

        search_patents_with_classification(
            query="neural network",
            cpc_classifications=["G06N3/045"],
        )

        params = mock_get.call_args.kwargs["params"]
        self.assertEqual(
            params["q"],
            'applicationMetaData.inventionTitle:"neural network"',
        )
        self.assertEqual(
            params["filters"],
            "applicationMetaData.cpcClassificationBag G06N3/045",
        )
        self.assertNotIn("filters", params["q"])


if __name__ == "__main__":
    unittest.main()
