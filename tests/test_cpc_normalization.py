"""Regression tests for USPTO ODP CPC extraction."""

from services.uspto_search_service import _normalize_cpc_classifications


def test_cpc_strings_are_preserved():
    assert _normalize_cpc_classifications(["G06V 40/16", "G06V 10/74"]) == [
        "G06V 40/16", "G06V 10/74"
    ]


def test_cpc_objects_extract_symbol_not_dict_repr():
    payload = [
        {"classificationSymbol": "G06V 40/16", "sequenceNumber": 1},
        {"classificationSymbol": "G06V 10/74", "sequenceNumber": 2},
    ]
    assert _normalize_cpc_classifications(payload) == ["G06V 40/16", "G06V 10/74"]


def test_cpc_nested_and_duplicate_values_are_normalized():
    payload = {
        "cpcClassification": [
            {"symbol": "G06V 40/16"},
            {"classificationCode": "G06V 10/74"},
            {"symbol": "G06V 40/16"},
        ]
    }
    assert _normalize_cpc_classifications(payload) == ["G06V 40/16", "G06V 10/74"]


def test_empty_or_unrecognized_cpc_payload_returns_empty_list():
    assert _normalize_cpc_classifications(None) == []
    assert _normalize_cpc_classifications([{"sequenceNumber": 1}]) == []
