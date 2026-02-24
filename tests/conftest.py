"""
tests/conftest.py

Shared pytest fixtures for the patent-search test suite.
All JSON fixture files live in tests/fixtures/ — import them here so
individual test modules can use them without re-reading the disk.
"""
import json
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def _load_fixture(filename: str) -> dict:
    return json.loads((FIXTURES_DIR / filename).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# PatentsView API response fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fx_page_full():
    """
    Full page: 5 patents, count == 5.
    Used with a patched PATENT_PAGE_SIZE=5 to trigger the pagination loop.
    """
    return _load_fixture("patentsview_page_full.json")


@pytest.fixture
def fx_page_partial():
    """
    Partial page: 3 patents (fewer than a full page).
    Signals the final page — pagination should stop after receiving this.
    """
    return _load_fixture("patentsview_page_partial.json")


@pytest.fixture
def fx_empty():
    """Zero-result response: patents array is empty."""
    return _load_fixture("patentsview_empty.json")


@pytest.fixture
def fx_error():
    """API-level error payload returned with HTTP 200 (error: true)."""
    return _load_fixture("patentsview_error.json")


@pytest.fixture
def fx_with_duplicates():
    """
    Four records where patent_id 11000001 and 11000002 each appear twice.
    Used to verify dedup_service reduces 4 inputs to 2 unique records.
    """
    return _load_fixture("patentsview_with_duplicates.json")


# ---------------------------------------------------------------------------
# Helper: build a mock requests.Response
# ---------------------------------------------------------------------------

def make_mock_response(data: dict, status_code: int = 200, headers: dict | None = None):
    """
    Return a unittest.mock.Mock that behaves like a requests.Response.

    Args:
        data:        Dict that .json() will return and .text will serialise.
        status_code: HTTP status code (default 200).
        headers:     Optional dict of response headers.
    """
    import json as _json
    from unittest.mock import Mock
    resp = Mock()
    resp.status_code = status_code
    resp.reason = "OK" if status_code == 200 else "Bad Request"
    resp.url = "https://search.patentsview.org/api/v1/patent/"
    resp.headers = headers or {}
    resp.text = _json.dumps(data)
    resp.json.return_value = data
    return resp


@pytest.fixture
def mock_response_factory():
    """Expose make_mock_response to test modules as a pytest fixture."""
    return make_mock_response
