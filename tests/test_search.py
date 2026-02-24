"""
tests/test_search.py

Unit tests for services/search_service.py and services/dedup_service.py.

All tests are offline — no live API calls are made.
requests.post is fully mocked using the fixture JSON files in tests/fixtures/.

Covered scenarios (per architecture spec):
  - Pagination loop fires when a full page is returned
  - Cursor (after) value is forwarded correctly on page 2+
  - MAX_RESULTS cap stops fetching early and notifies the user
  - MAX_PAGES cap stops fetching after N requests
  - HTTP 400 / network error returns an empty list gracefully
  - API-level error payload {error: true} returns an empty list
  - Zero results returns an empty list
  - dedup_service removes records with duplicate patent_id values
"""

import sys
from pathlib import Path
from unittest.mock import patch, call, Mock

import pytest

# Ensure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.search_service import fetch_patents_by_keywords  # noqa: E402
from services.dedup_service import deduplicate                  # noqa: E402
from models.schemas import PatentRecord                         # noqa: E402


# ===========================================================================
# Helpers
# ===========================================================================

def _resp(data: dict, status: int = 200, headers: dict | None = None) -> Mock:
    """Build a minimal mock requests.Response."""
    r = Mock()
    r.status_code = status
    r.reason = "OK" if status == 200 else "Bad Request"
    r.url = "https://search.patentsview.org/api/v1/patent/"
    r.headers = headers or {}
    r.text = str(data)
    r.json.return_value = data
    return r


# ===========================================================================
# search_service: error handling
# ===========================================================================

class TestSearchServiceErrors:

    def test_returns_empty_list_on_http_400(self, fx_page_full):
        """A 400 response must return [] without raising."""
        with patch("services.search_service.requests.post") as mock_post:
            mock_post.return_value = _resp(
                {},
                status=400,
                headers={"X-Status-Reason": "bad query", "Content-Type": "application/json"},
            )
            result = fetch_patents_by_keywords("laser", "", "")
        assert result == []
        assert mock_post.call_count == 1

    def test_returns_empty_list_on_network_exception(self):
        """A connection-level exception must be caught and return []."""
        import requests as req_lib
        with patch("services.search_service.requests.post",
                   side_effect=req_lib.exceptions.ConnectionError("timeout")):
            result = fetch_patents_by_keywords("laser", "", "")
        assert result == []

    def test_returns_empty_list_on_api_level_error_payload(self, fx_error):
        """HTTP 200 with {error: true} body must return []."""
        with patch("services.search_service.requests.post") as mock_post:
            mock_post.return_value = _resp(fx_error)
            result = fetch_patents_by_keywords("laser", "", "")
        assert result == []

    def test_returns_empty_list_on_zero_results(self, fx_empty):
        """An empty patents array must return []."""
        with patch("services.search_service.requests.post") as mock_post:
            mock_post.return_value = _resp(fx_empty)
            result = fetch_patents_by_keywords("xyzzy_nonexistent", "", "")
        assert result == []
        assert mock_post.call_count == 1


# ===========================================================================
# search_service: pagination
# ===========================================================================

class TestSearchServicePagination:

    def test_single_page_when_results_below_page_size(self, fx_page_partial):
        """
        When the API returns fewer records than the page size, only one
        request should be made (no cursor follow-up needed).
        """
        with patch("services.search_service.requests.post") as mock_post, \
             patch("services.search_service.MAX_RESULTS", None), \
             patch("services.search_service.MAX_PAGES", None), \
             patch("services.search_service.PATENT_PAGE_SIZE", 1000):
            mock_post.return_value = _resp(fx_page_partial)
            result = fetch_patents_by_keywords("neural", "", "")

        assert mock_post.call_count == 1
        assert len(result) == 3  # all records from fx_page_partial

    def test_pagination_fires_and_cursor_is_forwarded(
        self, fx_page_full, fx_page_partial
    ):
        """
        When a full page is returned, fetch_patents_by_keywords must:
          1. Make a second API request (pagination fires).
          2. Pass the last patent_id from page 1 as the 'after' cursor.
          3. Return all records from both pages combined.
        """
        with patch("services.search_service.requests.post") as mock_post, \
             patch("services.search_service.MAX_RESULTS", None), \
             patch("services.search_service.MAX_PAGES", None), \
             patch("services.search_service.PATENT_PAGE_SIZE", 5):
            # Page 1 → 5 patents (full); page 2 → 3 patents (partial → stop)
            mock_post.side_effect = [
                _resp(fx_page_full),
                _resp(fx_page_partial),
            ]
            result = fetch_patents_by_keywords("machine learning", "", "")

        # Both pages fetched
        assert mock_post.call_count == 2, "Expected exactly 2 API calls"
        assert len(result) == 8, f"Expected 8 patents (5+3), got {len(result)}"

        # The second call must carry the 'after' cursor from the last record on page 1
        last_id_page1 = fx_page_full["patents"][-1]["patent_id"]
        _, second_call_kwargs = mock_post.call_args_list[1]
        options = second_call_kwargs["json"]["o"]
        assert "after" in options, "Second request must include an 'after' cursor"
        assert options["after"] == last_id_page1, (
            f"Expected after='{last_id_page1}', got '{options['after']}'"
        )

    def test_first_request_has_no_cursor(self, fx_page_partial):
        """The very first request must not include an 'after' key in options."""
        with patch("services.search_service.requests.post") as mock_post, \
             patch("services.search_service.MAX_RESULTS", None), \
             patch("services.search_service.MAX_PAGES", None):
            mock_post.return_value = _resp(fx_page_partial)
            fetch_patents_by_keywords("neural network", "", "")

        _, first_call_kwargs = mock_post.call_args_list[0]
        options = first_call_kwargs["json"]["o"]
        assert "after" not in options, "First request must not contain an 'after' cursor"


# ===========================================================================
# search_service: result limits
# ===========================================================================

class TestSearchServiceLimits:

    def test_max_results_cap_stops_after_one_page(self, fx_page_full):
        """
        With MAX_RESULTS=5 and a 5-patent response, exactly one API call is
        made.  The second loop iteration detects remaining==0 and breaks
        before issuing another request.
        """
        with patch("services.search_service.requests.post") as mock_post, \
             patch("services.search_service.MAX_RESULTS", 5), \
             patch("services.search_service.MAX_PAGES", None), \
             patch("services.search_service.PATENT_PAGE_SIZE", 1000):
            mock_post.return_value = _resp(fx_page_full)
            result = fetch_patents_by_keywords("deep learning", "", "")

        assert mock_post.call_count == 1
        assert len(result) == 5

    def test_max_results_limits_per_page_size_requested(self, fx_page_partial):
        """
        When MAX_RESULTS=2, the size sent to the API must be 2 (not PATENT_PAGE_SIZE).
        """
        with patch("services.search_service.requests.post") as mock_post, \
             patch("services.search_service.MAX_RESULTS", 2), \
             patch("services.search_service.MAX_PAGES", None), \
             patch("services.search_service.PATENT_PAGE_SIZE", 1000):
            # Return only 2 records so the partial-page check stops pagination
            two_patent_body = {
                "error": False,
                "count": 2,
                "total_hits": 10,
                "patents": fx_page_partial["patents"][:2],
            }
            mock_post.return_value = _resp(two_patent_body)
            result = fetch_patents_by_keywords("transformer", "", "")

        _, kwargs = mock_post.call_args_list[0]
        assert kwargs["json"]["o"]["size"] == 2, "size sent to API should match MAX_RESULTS"
        assert len(result) == 2

    def test_max_pages_cap_stops_early(self, fx_page_full):
        """
        With MAX_PAGES=2 and PATENT_PAGE_SIZE=5, pagination should stop
        after exactly 2 API calls even though each page is full.
        """
        with patch("services.search_service.requests.post") as mock_post, \
             patch("services.search_service.MAX_RESULTS", None), \
             patch("services.search_service.MAX_PAGES", 2), \
             patch("services.search_service.PATENT_PAGE_SIZE", 5):
            # All calls return a full page — pagination would never end without the cap
            mock_post.side_effect = [_resp(fx_page_full), _resp(fx_page_full), _resp(fx_page_full)]
            result = fetch_patents_by_keywords("transformer", "", "")

        assert mock_post.call_count == 2, f"Expected 2 calls (MAX_PAGES=2), got {mock_post.call_count}"
        assert len(result) == 10  # 5 patents × 2 pages


# ===========================================================================
# dedup_service
# ===========================================================================

# ---------------------------------------------------------------------------
# Helpers for dedup tests: convert raw fixture dicts → PatentRecord objects
# ---------------------------------------------------------------------------

def _to_records(raw_list: list[dict]) -> list[PatentRecord]:
    """Convert a list of fixture dicts to PatentRecord objects."""
    return [PatentRecord(**d) for d in raw_list]


class TestDedupService:

    def test_removes_duplicate_patent_ids(self, fx_with_duplicates):
        """
        fx_with_duplicates contains 4 records where IDs 11000001 and 11000002
        each appear twice.  deduplicate must return exactly 2 unique records.
        """
        patents = _to_records(fx_with_duplicates["patents"])
        assert len(patents) == 4  # sanity-check fixture

        result = deduplicate(patents)

        assert len(result) == 2
        assert isinstance(result[0], PatentRecord)
        returned_ids = [p.patent_id for p in result]
        assert returned_ids == ["11000001", "11000002"]

    def test_preserves_order_of_first_occurrence(self, fx_with_duplicates):
        """First occurrence of each patent_id must be the one kept."""
        patents = _to_records(fx_with_duplicates["patents"])
        result = deduplicate(patents)
        assert result[0].patent_id == "11000001"
        assert result[1].patent_id == "11000002"

    def test_no_duplicates_unchanged(self, fx_page_full):
        """A list with no duplicates must pass through untouched."""
        patents = _to_records(fx_page_full["patents"])
        result = deduplicate(patents)
        assert len(result) == len(patents)
        assert [p.patent_id for p in result] == [p.patent_id for p in patents]

    def test_empty_list_returns_empty(self):
        assert deduplicate([]) == []

    def test_returns_patent_record_objects(self):
        """Return type must be list[PatentRecord], not list[dict]."""
        records = [
            PatentRecord(patent_id="X001", patent_title="Alpha"),
            PatentRecord(patent_id="X002", patent_title="Beta"),
            PatentRecord(patent_id="X001", patent_title="Alpha duplicate"),
        ]
        result = deduplicate(records)
        assert len(result) == 2
        assert all(isinstance(p, PatentRecord) for p in result)
