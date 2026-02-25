"""Patent search service.

Public API
----------
fetch_all_patents(query_terms)
    Takes a list of expanded query terms (strings) and returns a deduplicated
    combined list of PatentRecord objects across all terms.

fetch_patents_by_keywords(patent_title, patent_abstract, patent_type)
    Legacy entry point kept for backward compatibility (tests, CLI demo).
    Builds the query from the three keyword arguments and delegates to the
    internal pagination engine.  Returns list[PatentRecord].

Internal helpers
----------------
_build_headers()           -> dict
_build_term_query(term)    -> dict
_fetch_all_for_query(query) -> list[PatentRecord]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import requests

# Ensure the project root is importable regardless of the launch directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings, PATENT_PAGE_SIZE, MAX_RESULTS, MAX_PAGES  # noqa: E402
from models.schemas import PatentRecord  # noqa: E402
from utils.logger import get_logger  # noqa: E402

log = get_logger(__name__)

# Words that carry no technical meaning for patent search.
# Keep ONLY true linguistic stop words — never domain terms such as
# "system", "method", "device" which are meaningful in patent queries.
_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "of", "in", "for", "to", "with",
    "by", "on", "at", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "that", "this",
    "these", "those", "it", "its", "which", "who", "what", "where",
    "when", "how", "using", "used", "use", "via",
})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_headers() -> dict:
    """Return HTTP headers, injecting the API key when available."""
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    if settings.patentsview_api_key:
        headers["X-Api-Key"] = settings.patentsview_api_key
    return headers


def _strip_stop_words(term: str) -> str:
    """
    Remove stop words and single-character tokens from *term*.

    PatentsView _text_any tokenises the query string and matches patents
    containing ANY of those tokens.  Keeping stop words causes common English
    words to dominate and return completely irrelevant results.

    Returns the original term unchanged if stripping would leave nothing.
    """
    tokens = [
        w for w in term.lower().split()
        if w not in _STOP_WORDS and len(w) > 1  # allow 2-char terms like IR, AI
    ]
    cleaned = " ".join(tokens)
    if not cleaned:
        log.warning("Stop-word stripping left empty query; using original",
                    extra={"original": term})
        return term
    log.debug("Query cleaned", extra={"original": term, "cleaned": cleaned})
    return cleaned


def _build_term_query(term: str) -> dict:
    """Build a PatentsView query that searches *term* in title and abstract.

    Stop words are stripped first so that broad sentences (e.g. from the raw
    user input or Mistral output) are reduced to their technical keywords
    before being sent to the API.
    """
    cleaned = _strip_stop_words(term)
    return {
        "_or": [
            {"_text_any": {"patent_title": cleaned}},
            {"_text_any": {"patent_abstract": cleaned}},
        ]
    }


def build_token_aware_query(critical_tokens: list[str]) -> dict:
    """Build a structured ``_and`` query from a list of critical tokens.

    Each token becomes a ``_text_any`` clause so individual words within the
    token string can match anywhere in the abstract.  All clauses are wrapped
    in ``_and``, meaning *every* token group must appear somewhere in the
    patent — providing domain co-occurrence without phrase-match strictness.

    Example for ``["distributed ledger", "interbank"]``::

        {
            "_and": [
                {"_text_any": {"patent_abstract": "distributed ledger"}},
                {"_text_any": {"patent_abstract": "interbank"}},
            ]
        }

    Falls back to a plain ``_text_any`` OR query when fewer than two tokens
    are supplied (an ``_and`` of one clause is unnecessary overhead).

    Args:
        critical_tokens: Non-empty list of token strings from
            :func:`~services.token_extractor.extract_critical_tokens`.

    Returns:
        A ready-to-send PatentsView ``q`` payload dict.
    """
    if not critical_tokens:
        return {"_text_any": {"patent_abstract": ""}}

    # Single token — no _and needed, plain OR across title/abstract.
    if len(critical_tokens) == 1:
        token = critical_tokens[0]
        return {
            "_or": [
                {"_text_any": {"patent_title": token}},
                {"_text_any": {"patent_abstract": token}},
            ]
        }

    # Multiple tokens — require all to co-occur (abstract searched only to
    # avoid double-OR nesting; abstract contains all title text in practice).
    and_clauses = [
        {"_text_any": {"patent_abstract": token}}
        for token in critical_tokens
    ]
    return {"_and": and_clauses}


def _fetch_all_for_query(query: dict) -> list[PatentRecord]:
    """Paginate the PatentsView API for *query* and return PatentRecord objects.

    Uses cursor-based pagination (``after`` = last ``patent_id`` on each page).
    Stops when:
      - a partial page is returned (no more results), or
      - MAX_RESULTS documents have been collected, or
      - MAX_PAGES requests have been made.
    """
    url = "https://search.patentsview.org/api/v1/patent/"
    headers = _build_headers()

    all_patents: list[PatentRecord] = []
    after_cursor: Optional[str] = None  # None on the first request
    page_number = 0

    while True:
        # ── Page cap ────────────────────────────────────────────────────────
        if MAX_PAGES is not None and page_number >= MAX_PAGES:
            log.warning(
                "Reached MAX_PAGES limit",
                extra={"max_pages": MAX_PAGES, "collected": len(all_patents)},
            )
            print(
                f"⚠️  Results limited: reached the maximum of {MAX_PAGES} page(s). "
                f"Returning {len(all_patents)} document(s). "
                "To fetch more, increase MAX_PAGES in config.py."
            )
            break

        # ── Document cap ─────────────────────────────────────────────────────
        if MAX_RESULTS is not None:
            remaining = MAX_RESULTS - len(all_patents)
            if remaining <= 0:
                log.warning(
                    "Reached MAX_RESULTS limit",
                    extra={"max_results": MAX_RESULTS},
                )
                print(
                    f"⚠️  Results limited to {MAX_RESULTS} document(s). "
                    "To fetch more, increase MAX_RESULTS in config.py."
                )
                break
            size = min(PATENT_PAGE_SIZE, remaining)
        else:
            size = PATENT_PAGE_SIZE

        # ── Build request options (cursor-based pagination) ──────────────────
        # First page:  {"size": N}
        # Later pages: {"size": N, "after": "<last_patent_id_from_previous_page>"}
        options: dict = {"size": size}
        if after_cursor is not None:
            options["after"] = after_cursor

        # Field list from settings.patent_fields so all services share the schema.
        params = {
            "q": query,
            "f": settings.patent_fields,
            "o": options,
        }

        try:
            log.info("Final USPTO Query: %s", json.dumps(query, indent=2))
            response = requests.post(url, headers=headers, json=params)
            page_number += 1

            if response.status_code != 200:
                log.error(
                    "API HTTP error",
                    extra={"status": response.status_code, "reason": response.reason},
                )
                print(
                    f"❌ API Request Error: {response.status_code} "
                    f"{response.reason} for url: {response.url}"
                )
                for h in ("X-Status-Reason", "X-Status-Reason-Code", "Content-Type"):
                    if h in response.headers:
                        print(f"  {h}: {response.headers[h]}")
                print((response.text or "")[:1000])
                break

            try:
                data = response.json()
            except json.JSONDecodeError:
                log.error("Failed to decode JSON response")
                print("❌ Failed to decode JSON response.")
                break

            if isinstance(data, dict) and data.get("error"):
                reason = data.get("reason") or data.get("message") or "unknown"
                log.error("API-level error payload", extra={"reason": reason})
                print(f"❌ API returned error: {reason}")
                break

            raw_page: list[dict] = (
                data.get("patents", []) if isinstance(data, dict) else []
            )

            # Convert raw dicts → PatentRecord objects; skip invalid records.
            page_records: list[PatentRecord] = []
            for raw in raw_page:
                try:
                    page_records.append(PatentRecord(**raw))
                except Exception as exc:
                    log.warning(
                        "Skipping invalid patent record",
                        extra={"error": str(exc), "patent_id": raw.get("patent_id")},
                    )

            all_patents.extend(page_records)
            log.debug(
                "Fetched page",
                extra={"page": page_number, "count": len(page_records), "total": len(all_patents)},
            )

            # A page shorter than requested → no more results available.
            if len(raw_page) < size:
                break

            # Advance the cursor to the last patent_id on this page.
            after_cursor = raw_page[-1].get("patent_id")
            if after_cursor is None:
                log.warning("patent_id missing on last record; cannot paginate further")
                print("⚠️  Could not determine cursor for next page (patent_id missing). Stopping.")
                break

        except requests.exceptions.RequestException as exc:
            log.error("Network error", extra={"error": str(exc)})
            print(f"❌ API Request Error: {exc}")
            break

    log.info(
        "Search complete",
        extra={"pages_fetched": page_number, "patents_returned": len(all_patents)},
    )
    return all_patents


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_patents_for_query_dict(query_dict: dict) -> list[PatentRecord]:
    """Execute a single pre-built PatentsView query dict.

    Used by :mod:`services.query_builder` / the tiered search strategy in the
    pipeline.  Separates query *construction* from query *execution* cleanly,
    and keeps all pagination logic in one place.

    Parameters
    ----------
    query_dict:
        A ready-to-send PatentsView ``q`` payload, e.g.::

            {"_text_any": {"patent_abstract": "lane detection"}}

    Returns
    -------
    list[PatentRecord]
    """
    return _fetch_all_for_query(query_dict)


def fetch_all_patents(query_terms: list[str]) -> list[PatentRecord]:
    """Fetch patents for every term in *query_terms* and return the combined list.

    This is the primary entry point for the pipeline.  The caller is responsible
    for deduplicating the results (use ``services.dedup_service.deduplicate``).

    Parameters
    ----------
    query_terms:
        Typically the output of ``services.query_expansion.expand_query`` —
        a list of search strings including the original query and alternatives.

    Returns
    -------
    list[PatentRecord]
        All PatentRecord objects collected across all terms, in fetch order.
    """
    log.info("fetch_all_patents started", extra={"num_terms": len(query_terms)})
    all_patents: list[PatentRecord] = []
    for term in query_terms:
        log.debug("Fetching term", extra={"term": term})
        results = _fetch_all_for_query(_build_term_query(term))
        all_patents.extend(results)
        log.info(
            "Term fetch complete",
            extra={"term": term, "fetched": len(results), "running_total": len(all_patents)},
        )
    log.info("fetch_all_patents complete", extra={"total": len(all_patents)})
    return all_patents


def fetch_patents_by_keywords(
    patent_title: str,
    patent_abstract: str,
    patent_type: str,
) -> list[PatentRecord]:
    """Fetch patents using separate keyword arguments (legacy / CLI).

    Builds the PatentsView ``_or`` query from the three keyword fields and
    delegates to the shared pagination engine.

    Returns
    -------
    list[PatentRecord]
    """
    # Use documented PatentsView endpoint
    url = "https://search.patentsview.org/api/v1/patent/"  # noqa: F841 (kept for clarity)

    # Build query using operator-first format per API docs
    or_clauses = []
    if patent_title:
        or_clauses.append({"_text_any": {"patent_title": patent_title}})
    if patent_abstract:
        or_clauses.append({"_text_any": {"patent_abstract": patent_abstract}})
    if patent_type:
        or_clauses.append({"_eq": {"patent_type": patent_type}})

    query = {"_or": or_clauses} if or_clauses else {"_text_any": {"patent_title": ""}}
    return _fetch_all_for_query(query)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Usage:
    #   python search_service.py                           # default terms
    #   python search_service.py "machine learning"        # single term
    #   python search_service.py "ML" "neural network"     # multiple terms
    terms = sys.argv[1:] if len(sys.argv) > 1 else ["machine learning", "neural network"]
    patents = fetch_all_patents(terms)
    print(json.dumps([p.model_dump() for p in patents], indent=2))
