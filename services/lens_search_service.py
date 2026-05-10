"""
services/lens_search_service.py

Lens.org Patent API search backend — drop-in replacement for the defunct
PatentsView search.patentsview.org endpoint.

Public API
----------
fetch_all_patents(query_terms)
    Same signature as search_service.fetch_all_patents — the pipeline calls
    this transparently when lens_api_token is set in .env.

fetch_result_count(schema)
    Returns total patent count for a JsonQuerySchema (used by the JSON mode
    panel in the UI).

Lens.org API docs: https://docs.api.lens.org/patent.html
Free trial: https://www.lens.org/lens/user/subscriptions
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from models.schemas import PatentRecord  # noqa: E402
from utils.logger import get_logger  # noqa: E402

log = get_logger(__name__)

_LENS_URL = settings.lens_api_url
_PAGE_SIZE = 50   # Lens.org max per request on trial plan
_MAX_RESULTS = 500


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _headers() -> dict:
    return {
        "Authorization": f"Bearer {settings.lens_api_token}",
        "Content-Type": "application/json",
    }


def _build_query(term: str) -> dict:
    """
    Build a Lens.org query payload for a single search term.

    Searches title + abstract with a simple query_string.
    Multi-word terms are phrase-matched automatically by Lens when quoted.
    """
    # Quote multi-word terms for phrase matching
    q_term = f'"{term}"' if " " in term else term
    return {
        "query": {
            "query_string": {
                "query": q_term,
                "fields": ["title", "abstract"],
                "default_operator": "AND",
            }
        },
        "include": [
            "lens_id",
            "title",
            "abstract",
            "publication_type",
            "date_published",
            "doc_number",
            "jurisdiction",
        ],
        "size": _PAGE_SIZE,
        "scroll": "1m",
    }


def _build_group_query(groups: list[list[str]], combine: str = "AND") -> dict:
    """
    Build a Lens.org query from JSON-mode groups (AND/OR of term groups).
    Each group becomes a should (OR) clause; groups are joined with must (AND).
    """
    bool_op = "must" if combine.upper() == "AND" else "should"

    group_clauses = []
    for group_terms in groups:
        term_parts = []
        for t in group_terms:
            qt = f'"{t}"' if " " in t else t
            term_parts.append(qt)
        group_qs = " OR ".join(term_parts)
        group_clauses.append({
            "query_string": {
                "query": group_qs,
                "fields": ["title", "abstract"],
            }
        })

    return {
        "query": {"bool": {bool_op: group_clauses}},
        "include": [
            "lens_id",
            "title",
            "abstract",
            "publication_type",
            "date_published",
            "doc_number",
            "jurisdiction",
        ],
        "size": _PAGE_SIZE,
        "scroll": "1m",
    }


def _record_from_hit(hit: dict) -> Optional[PatentRecord]:
    """Convert a Lens.org hit dict to a PatentRecord."""
    try:
        lens_id = hit.get("lens_id", "")
        doc_no  = hit.get("doc_number", "")
        jur     = hit.get("jurisdiction", "")
        # Use doc_number+jurisdiction as the patent_id so it's human-readable
        patent_id = f"{jur}-{doc_no}" if doc_no and jur else lens_id

        title    = hit.get("title") or ""
        abstract = hit.get("abstract") or ""
        pub_type = hit.get("publication_type") or ""
        date     = hit.get("date_published") or ""

        if not patent_id or not title:
            return None

        return PatentRecord(
            patent_id=patent_id,
            patent_title=title,
            patent_abstract=abstract,
            patent_type=pub_type,
            patent_date=date,
        )
    except Exception as exc:
        log.warning("Skipping invalid Lens hit: %s", exc)
        return None


def _fetch_for_payload(payload: dict) -> list[PatentRecord]:
    """POST a Lens query and paginate using scroll_id until done or cap reached."""
    results: list[PatentRecord] = []
    scroll_id: Optional[str] = None

    while True:
        if len(results) >= _MAX_RESULTS:
            break

        if scroll_id:
            # Subsequent pages use the scroll endpoint
            body = {"scroll_id": scroll_id, "include": payload.get("include", [])}
            resp = requests.post(
                f"{_LENS_URL}/scroll",
                headers=_headers(),
                json=body,
                timeout=20,
            )
        else:
            resp = requests.post(
                _LENS_URL,
                headers=_headers(),
                json=payload,
                timeout=20,
            )

        if resp.status_code == 401:
            log.error("Lens API: unauthorised — check lens_api_token in .env")
            print("❌ Lens API: Unauthorised. Add your token to .env as lens_api_token=...")
            break

        if resp.status_code == 429:
            log.warning("Lens API: rate limited")
            print("⚠️  Lens API rate limit hit — pausing is not supported in sync mode.")
            break

        if resp.status_code != 200:
            log.error("Lens API HTTP error: %s %s", resp.status_code, resp.text[:300])
            print(f"❌ Lens API error {resp.status_code}: {resp.text[:300]}")
            break

        data = resp.json()
        hits = data.get("data", [])
        scroll_id = data.get("scroll_id")

        for hit in hits:
            rec = _record_from_hit(hit)
            if rec:
                results.append(rec)

        # No more results
        if not hits or len(hits) < _PAGE_SIZE:
            break

        if not scroll_id:
            break

    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_all_patents(query_terms: list[str]) -> list[PatentRecord]:
    """
    Fetch patents from Lens.org for each term in *query_terms*.

    Drop-in replacement for search_service.fetch_all_patents.
    The caller should deduplicate results with dedup_service.deduplicate.
    """
    log.info("lens fetch_all_patents started", extra={"num_terms": len(query_terms)})
    all_patents: list[PatentRecord] = []

    for term in query_terms:
        log.debug("Lens: fetching term", extra={"term": term})
        payload = _build_query(term)
        results = _fetch_for_payload(payload)
        all_patents.extend(results)
        log.info("Lens: term complete", extra={"term": term, "fetched": len(results)})

    log.info("lens fetch_all_patents complete", extra={"total": len(all_patents)})
    return all_patents


def fetch_all_patents_from_groups(
    groups: list[list[str]],
    combine: str = "AND",
) -> list[PatentRecord]:
    """
    Fetch patents using the full JSON-mode group query (single API call).

    More precise than per-term fetching — uses the full Boolean group logic.
    """
    log.info("Lens: fetching from groups", extra={"num_groups": len(groups)})
    payload = _build_group_query(groups, combine)
    results = _fetch_for_payload(payload)
    log.info("Lens: group fetch complete", extra={"total": len(results)})
    return results


def fetch_result_count(groups: list[list[str]], combine: str = "AND") -> Optional[int]:
    """
    Return total_hits for a group query (used by UI for 'PatentsView hits' metric).
    Fires a size=1 request — no scroll needed.
    """
    payload = _build_group_query(groups, combine)
    payload["size"] = 1
    payload.pop("scroll", None)

    try:
        resp = requests.post(_LENS_URL, headers=_headers(), json=payload, timeout=15)
        if resp.status_code == 200:
            return resp.json().get("total", {}).get("value")
    except Exception as exc:
        log.warning("Lens count fetch failed: %s", exc)
    return None
