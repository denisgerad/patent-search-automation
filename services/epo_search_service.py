"""
services/epo_search_service.py

EPO CQL builder — updated to use tokens.epo_search_order.

Key change:
  _build_cql() now receives terms pre-ordered by discriminating power
  (most specific multi-word phrase first, generic single terms last).
  This means EPO's AND chain filters aggressively on the most specific
  term first, producing a much smaller and more relevant result set
  before the less discriminating terms are applied.

  Example for "autonomous vehicle lane detection using infrared":
    OLD: (ti=infrared OR ab=infrared) AND (ti="autonomous vehicle" OR ...) AND ...
    NEW: (ti="autonomous vehicle" OR ab="autonomous vehicle")
         AND (ti="lane detection" OR ab="lane detection")
         AND (ti=infrared OR ab=infrared)

  The new order means EPO first finds patents about autonomous vehicles
  (already a narrow domain), then filters to lane detection ones, then
  to infrared ones — instead of starting with every infrared patent
  across medical, display, security, and automotive domains.

Tiered search strategy:
  Tier 1 — FULL AND: all terms in epo_search_order. Highest precision.
  Tier 2 — DROP LAST: remove the least discriminating term (last in order).
            Used if Tier 1 returns zero results.
  Tier 3 — TOP 2 ONLY: use only the two most discriminating terms.
            Final fallback before OR relaxation.
  Tier 4 — OR relaxation: only if all AND tiers returned nothing.
"""

from __future__ import annotations

from models.epo_client import EPOClient
from models.schemas import PatentRecord
from utils.logger import get_logger

log = get_logger(__name__)

_epo_client = EPOClient()


def _build_cql(keyword_terms: list[str]) -> str:
    """
    Build EPO CQL from terms already ordered by discriminating power.
    Multi-word terms are phrase-quoted automatically.

    Example input (most → least discriminating):
      ["autonomous vehicle", "lane detection", "infrared"]
    Output:
      (ti="autonomous vehicle" OR ab="autonomous vehicle")
      AND (ti="lane detection" OR ab="lane detection")
      AND (ti=infrared OR ab=infrared)
    """
    parts = []
    for term in keyword_terms:
        quoted = f'"{term}"' if " " in term else term
        parts.append(f"(ti={quoted} OR ab={quoted})")
    return " AND ".join(parts)


def epo_fetch_by_keywords(keyword_terms: list[str]) -> list[PatentRecord]:
    """
    Search EPO with a tiered AND strategy using pre-ordered keyword_terms.

    keyword_terms should be tokens.epo_search_order (most discriminating first).
    Falls back through progressively relaxed tiers if no results found.
    """
    if not keyword_terms:
        log.warning("epo_fetch_by_keywords called with empty terms")
        return []

    # --- Tier 1: full AND on all terms (highest precision) ----------------
    cql = _build_cql(keyword_terms)
    log.info("EPO Tier 1 CQL: %s", cql)
    patent_ids = _epo_client.search(cql, max_results=50)

    # --- Tier 2: drop least discriminating term (last in list) ------------
    if not patent_ids and len(keyword_terms) > 2:
        tier2_terms = keyword_terms[:-1]
        cql2 = _build_cql(tier2_terms)
        log.info("EPO Tier 2 CQL (dropped '%s'): %s", keyword_terms[-1], cql2)
        patent_ids = _epo_client.search(cql2, max_results=50)

    # --- Tier 3: top 2 most discriminating terms only ---------------------
    if not patent_ids and len(keyword_terms) > 1:
        tier3_terms = keyword_terms[:2]
        cql3 = _build_cql(tier3_terms)
        log.info("EPO Tier 3 CQL (top-2 only): %s", cql3)
        patent_ids = _epo_client.search(cql3, max_results=50)

    # --- Tier 4: OR relaxation (last resort) ------------------------------
    if not patent_ids:
        parts = []
        for term in keyword_terms:
            quoted = f'"{term}"' if " " in term else term
            parts.append(f"(ti={quoted} OR ab={quoted})")
        cql_or = " OR ".join(parts)
        log.warning("EPO Tier 4 CQL (OR relaxation): %s", cql_or)
        patent_ids = _epo_client.search(cql_or, max_results=50)

    if not patent_ids:
        log.error("EPO returned 0 patents across all tiers for terms: %s", keyword_terms)
        return []

    # Fetch biblio for each patent ID
    records: list[PatentRecord] = []
    for pid in patent_ids:
        biblio = _epo_client.fetch_biblio(pid)
        if not biblio:
            continue
        if biblio["abstract"] == "NO ABSTRACT FOUND":
            log.debug("Skipping %s — no abstract", pid)
            continue
        try:
            records.append(PatentRecord(
                patent_id       = biblio["patent_id"],
                patent_title    = biblio["title"],
                patent_abstract = biblio["abstract"],
                patent_type     = "",
                patent_date     = None,
            ))
        except Exception as e:
            log.warning("PatentRecord build failed for %s: %s", pid, e)

    log.info(
        "EPO fetch complete: %d usable records from %d IDs",
        len(records), len(patent_ids),
    )
    return records


def fetch_all_patents(query_terms: list[str]) -> list[PatentRecord]:
    """Compatibility wrapper."""
    return epo_fetch_by_keywords(query_terms)
