"""
services/json_query_service.py

Handles the structured JSON query format that users can paste into the UI
instead of relying on Mistral query expansion.

Expected JSON schema
--------------------
{
  "groups": [
    { "terms": ["autonomous vehicle", "self-driving vehicle"] },
    { "terms": ["lane detection", "lane recognition"] },
    { "terms": ["infrared sensor", "IR detector"] }
  ],
  "combine_groups_with": "AND"   // "AND" | "OR"  (default: "AND")
}

Public API
----------
validate_json_query(raw_str)          -> JsonQuerySchema  (raises ValueError on bad input)
build_boolean_string(schema)          -> str   e.g. "(A OR B) AND (C OR D)"
build_patentsview_query(schema)       -> dict  PatentsView API-ready query dict
build_uspto_url(boolean_str)          -> str   USPTO Full Text clickable link
fetch_result_count(schema)            -> int | None  single-page API call for total count
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote_plus
from typing import Optional

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from utils.logger import get_logger  # noqa: E402

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

@dataclass
class TermGroup:
    """One group of synonym / alternative terms (joined with OR)."""
    terms: list[str]


@dataclass
class JsonQuerySchema:
    """Validated, normalised representation of a pasted JSON query."""
    groups: list[TermGroup]
    combine_with: str = "AND"   # "AND" | "OR"

    @property
    def all_terms(self) -> list[str]:
        """Flat list of every term across all groups."""
        result: list[str] = []
        for g in self.groups:
            result.extend(g.terms)
        return result


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_json_query(raw_str: str) -> JsonQuerySchema:
    """
    Parse and validate the user-pasted JSON query string.

    Raises
    ------
    ValueError
        If the JSON is malformed or does not satisfy the required structure.
    """
    try:
        data = json.loads(raw_str)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON syntax — {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("JSON must be an object (not an array or scalar).")

    raw_groups = data.get("groups")
    if not raw_groups or not isinstance(raw_groups, list):
        raise ValueError('JSON must contain a non-empty "groups" array.')

    groups: list[TermGroup] = []
    for idx, g in enumerate(raw_groups):
        if not isinstance(g, dict):
            raise ValueError(f'groups[{idx}] must be an object, got {type(g).__name__}.')
        raw_terms = g.get("terms")
        if not isinstance(raw_terms, list) or not raw_terms:
            raise ValueError(f'groups[{idx}] must have a non-empty "terms" array.')
        cleaned = [str(t).strip() for t in raw_terms if str(t).strip()]
        if not cleaned:
            raise ValueError(f'groups[{idx}] has no non-empty term strings.')
        groups.append(TermGroup(terms=cleaned))

    if not groups:
        raise ValueError("At least one group with at least one term is required.")

    raw_combine = str(data.get("combine_groups_with", "AND")).upper()
    if raw_combine not in {"AND", "OR"}:
        raise ValueError('"combine_groups_with" must be "AND" or "OR".')

    return JsonQuerySchema(groups=groups, combine_with=raw_combine)


# ---------------------------------------------------------------------------
# Boolean string builder
# ---------------------------------------------------------------------------

def build_boolean_string(schema: JsonQuerySchema) -> str:
    """
    Build a human-readable (and USPTO-compatible) Boolean query string.

    Example
    -------
    Input groups: [["A", "B"], ["C", "D"]], combine_with="AND"
    Output: "(A OR B) AND (C OR D)"
    """
    parts: list[str] = []
    for g in schema.groups:
        if len(g.terms) == 1:
            # No parentheses needed for a single term
            parts.append(g.terms[0])
        else:
            inner = " OR ".join(g.terms)
            parts.append(f"({inner})")

    joiner = f" {schema.combine_with} "
    return joiner.join(parts)


# ---------------------------------------------------------------------------
# PatentsView query builder
# ---------------------------------------------------------------------------

def build_patentsview_query(schema: JsonQuerySchema) -> dict:
    """
    Build a PatentsView API-ready ``q`` dict that mirrors the Boolean logic.

    Each group becomes an ``_or`` clause across title & abstract.
    Groups are combined with ``_and`` or ``_or`` depending on combine_with.

    For a single group the outer wrapper is omitted.
    """
    def _group_clause(group: TermGroup) -> dict:
        """One group → flat _or of _text_any clauses across title & abstract."""
        # Flatten: each term contributes two _text_any leaves (title + abstract)
        # so the whole group is a single _or without nested _or wrappers.
        leaves: list[dict] = []
        for t in group.terms:
            leaves.append({"_text_any": {"patent_title": t}})
            leaves.append({"_text_any": {"patent_abstract": t}})
        if len(leaves) == 1:
            return leaves[0]
        return {"_or": leaves}

    group_clauses = [_group_clause(g) for g in schema.groups]

    if len(group_clauses) == 1:
        return group_clauses[0]

    outer_key = "_and" if schema.combine_with == "AND" else "_or"
    return {outer_key: group_clauses}


# ---------------------------------------------------------------------------
# USPTO URL builder
# ---------------------------------------------------------------------------

_USPTO_FULLT_BASE = (
    "https://patft.uspto.gov/netacgi/nph-Parser"
    "?Sect1=PTO2&Sect2=HITOFF"
    "&u=%2Fnetahtml%2FPTO%2Fsearch-bool.html"
    "&r=0&p=1&f=S&l=50"
    "&Query="
)


def build_uspto_url(boolean_str: str) -> str:
    """
    Return a USPTO Patent Full Text Database URL pre-loaded with *boolean_str*.

    The returned link opens straight into the search results page.
    """
    return _USPTO_FULLT_BASE + quote_plus(boolean_str)


# ---------------------------------------------------------------------------
# PatentsView result count
# ---------------------------------------------------------------------------

def fetch_result_count(schema: JsonQuerySchema) -> Optional[int]:
    """
    Fire a single size=1 request to PatentsView and return ``total_patent_count``.

    Returns ``None`` on any network / API error (non-fatal — UI shows "unknown").
    """
    url = "https://search.patentsview.org/api/v1/patent/"
    headers: dict = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    if settings.patentsview_api_key:
        headers["X-Api-Key"] = settings.patentsview_api_key

    query_dict = build_patentsview_query(schema)
    params = {
        "q": query_dict,
        "f": ["patent_id"],          # minimal field set → fast response
        "o": {"size": 1},
    }

    try:
        log.info("Fetching result count for JSON query")
        resp = requests.post(url, headers=headers, json=params, timeout=15)
        if resp.status_code != 200:
            log.warning(
                "Count request returned non-200",
                extra={"status": resp.status_code},
            )
            return None
        data = resp.json()
        count = data.get("total_patent_count")
        if count is not None:
            log.info("PatentsView total count: %s", count)
            return int(count)
        return None
    except Exception as exc:
        log.warning("Failed to fetch result count: %s", exc)
        return None
