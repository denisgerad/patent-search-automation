"""
services/epo_search_service.py

EPO Open Patent Services (OPS) search backend.

Uses EPOClient (models/epo_client.py) which mirrors the two-step pattern
from test_epo.py:
  Step 1: search CQL → get patent IDs
  Step 2: fetch biblio per ID → get title + abstract

Token is cached in the singleton _epo_client so re-auth does not happen
on every query call.

Public API
----------
epo_fetch_by_keywords(keyword_terms) -> list[PatentRecord]
fetch_all_patents(query_terms)        -> list[PatentRecord]
"""

from __future__ import annotations

from models.epo_client import EPOClient
from models.schemas import PatentRecord
from utils.logger import get_logger

log = get_logger(__name__)

# Module-level singleton — one token shared across all calls in a session
_epo_client = EPOClient()


def _build_cql(keyword_terms: list[str]) -> str:
    """
    Build CQL from short keyword terms only.

    Correct input : ["infrared", "lane detection", "autonomous vehicle"]
    Wrong input   : ["Method for lane detection comprising infrared steps"]

    CQL reference:
      ti=term        → title contains term
      ab=term        → abstract contains term
      "two words"    → phrase search (wrap multi-word in quotes)
      AND            → all conditions must match
    """
    parts = []
    for term in keyword_terms:
        quoted = f'"{term}"' if " " in term else term
        parts.append(f"(ti={quoted} OR ab={quoted})")
    return " AND ".join(parts)


def epo_fetch_by_keywords(keyword_terms: list[str]) -> list[PatentRecord]:
    """
    Full two-step EPO fetch matching the test_epo.py pattern.

    Step 1: search CQL → get patent IDs
    Step 2: fetch biblio per ID → get title + abstract
    Step 3: convert to PatentRecord objects

    keyword_terms must be short technical terms from token_extractor,
    NOT full Mistral-expanded natural language phrases.
    """
    if not keyword_terms:
        log.warning("epo_fetch_by_keywords called with empty terms")
        return []

    # Step 1 — build CQL and search (AND query for precision)
    cql = _build_cql(keyword_terms)
    log.info(f"EPO CQL: {cql}")
    patent_ids = _epo_client.search(cql, max_results=25)

    if not patent_ids:
        # Retry with relaxed CQL — OR instead of AND
        log.warning("No results with AND query, retrying with OR")
        relaxed_parts = []
        for term in keyword_terms:
            quoted = f'"{term}"' if " " in term else term
            relaxed_parts.append(f"(ti={quoted} OR ab={quoted})")
        relaxed_cql = " OR ".join(relaxed_parts)
        log.info(f"EPO relaxed CQL: {relaxed_cql}")
        patent_ids = _epo_client.search(relaxed_cql, max_results=25)

    if not patent_ids:
        log.error(f"EPO returned 0 patents for terms: {keyword_terms}")
        return []

    # Step 2 — fetch full biblio per ID
    records: list[PatentRecord] = []
    for pid in patent_ids:
        biblio = _epo_client.fetch_biblio(pid)
        if not biblio:
            continue
        if biblio["abstract"] == "NO ABSTRACT FOUND":
            log.debug(f"Skipping {pid} — no abstract (unusable for embedding)")
            continue
        try:
            records.append(PatentRecord(
                patent_id       = biblio["patent_id"],
                patent_title    = biblio["title"],
                patent_abstract = biblio["abstract"],
                patent_type     = "",      # EPO doesn't return type in biblio
                patent_date     = None,
            ))
        except Exception as e:
            log.warning(f"PatentRecord build failed for {pid}: {e}")

    log.info(f"EPO fetch complete: {len(records)} usable records "
             f"(with abstracts) from {len(patent_ids)} IDs")
    return records


def fetch_all_patents(query_terms: list[str]) -> list[PatentRecord]:
    """
    Compatibility wrapper — delegates to epo_fetch_by_keywords.
    query_terms are treated as short keyword terms (same as keyword_terms).
    """
    return epo_fetch_by_keywords(query_terms)
