from types import SimpleNamespace

import pytest

from services import uspto_search_service as service


class FakeResponse:
    def __init__(self, status_code, data=None, text=""):
        self.status_code = status_code
        self._data = data or {}
        self.text = text

    def json(self):
        return self._data


def test_strategy_search_retries_413_with_smaller_limit(monkeypatch):
    monkeypatch.setattr(
        service,
        "_build_headers",
        lambda: {"X-API-KEY": "test-key"},
    )
    monkeypatch.setattr(
        service,
        "_build_patent_record",
        lambda raw: SimpleNamespace(patent_id=raw["id"]),
    )

    calls = []

    def fake_post(url, headers, json, timeout):
        calls.append(json)
        if len(calls) == 1:
            return FakeResponse(413, text="Payload Too Large")
        return FakeResponse(
            200,
            {"patentFileWrapperDataBag": [{"id": "P1"}]},
        )

    monkeypatch.setattr(service.requests, "post", fake_post)

    records = service.search_patents_by_strategy(
        "fraud score",
        limit=100,
        offset=20,
    )

    assert [call["pagination"]["limit"] for call in calls] == [100, 25]
    assert all(call["q"] == "fraud score" for call in calls)
    assert all(call["pagination"]["offset"] == 20 for call in calls)
    assert [record.patent_id for record in records] == ["P1"]


def test_strategy_search_raises_clear_error_if_limit_one_fails(monkeypatch):
    monkeypatch.setattr(
        service,
        "_build_headers",
        lambda: {"X-API-KEY": "test-key"},
    )

    calls = []

    def fake_post(url, headers, json, timeout):
        calls.append(json)
        return FakeResponse(413, text="Payload Too Large")

    monkeypatch.setattr(service.requests, "post", fake_post)

    with pytest.raises(RuntimeError, match="page size 1"):
        service.search_patents_by_strategy(
            "fraud score",
            limit=1,
            offset=0,
        )

    assert [call["pagination"]["limit"] for call in calls] == [1]
