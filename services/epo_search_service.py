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
from services.token_extractor import ExtractedTokens
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


def _build_cql_tiered(tokens: ExtractedTokens) -> list[str]:
    """
    Build CQL queries from most → least specific.
    Returns a list of CQL strings to try in order until MIN_RESULTS are found.

    Tier 1 (narrow):  primary_anchor AND domain_concept
    Tier 2 (medium):  primary_anchor only
    Tier 3 (broad):   primary_anchor synonyms OR domain synonyms
    """
    anchor   = tokens.primary_anchor
    anchor_q = f'"{anchor}"' if " " in anchor else anchor

    # Collect domain-level terms (autonomous vehicle etc.)
    domain_terms: list[str] = []
    for concept, data in tokens.concept_groups.items():
        if data.get("type") == "domain":
            domain_terms.extend(data["matched"])
            domain_terms.extend(data["patent_synonyms"][:2])

    cql_tiers: list[str] = []

    # Tier 1: anchor AND domain
    if domain_terms:
        domain_parts = " OR ".join(
            f'"{t}"' if " " in t else t for t in domain_terms[:3]
        )
        cql_tiers.append(
            f"(ti={anchor_q} OR ab={anchor_q}) "
            f"AND (ti=({domain_parts}) OR ab=({domain_parts}))"
        )

    # Tier 2: anchor only
    cql_tiers.append(f"ti={anchor_q} OR ab={anchor_q}")

    # Tier 3: anchor synonyms
    anchor_concept = tokens.concept_groups.get(tokens.primary_concept, {})
    synonyms = anchor_concept.get("patent_synonyms", [])[:3]
    if synonyms:
        syn_parts = " OR ".join(
            f'"{s}"' if " " in s else s for s in synonyms
        )
        cql_tiers.append(f"ti=({syn_parts}) OR ab=({syn_parts})")

    return cql_tiers


def epo_fetch_by_keywords(tokens: ExtractedTokens) -> list[PatentRecord]:
    """
    Tiered CQL search — tries narrow first, broadens until MIN_RESULTS met.
    Receives full ExtractedTokens; uses primary_anchor and concept_groups
    to build discriminating CQL rather than a flat keyword list.
    """
    if not tokens.primary_anchor:
        log.warning("epo_fetch_by_keywords: primary_anchor is empty — no search performed")
        return []

    MIN_RESULTS = 10
    cql_tiers   = _build_cql_tiered(tokens)
    all_ids: list[str] = []

    for i, cql in enumerate(cql_tiers):
        log.info("EPO CQL tier %d: %s", i + 1, cql)
        ids = _epo_client.search(cql, max_results=25)
        log.info("Tier %d returned %d IDs", i + 1, len(ids))
        all_ids.extend(ids)
        # Deduplicate while preserving order
        all_ids = list(dict.fromkeys(all_ids))
        if len(all_ids) >= MIN_RESULTS:
            log.info("MIN_RESULTS met at tier %d, stopping", i + 1)
            break

    if not all_ids:
        log.error(
            "EPO returned 0 patents across all CQL tiers. "
            "Primary anchor: '%s' — consider adding synonyms to taxonomy.",
            tokens.primary_anchor,
        )
        return []

    # Fetch biblio for each ID
    records: list[PatentRecord] = []
    for pid in all_ids:
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
        len(records), len(all_ids),
    )
    return records


def fetch_all_patents(query_terms: list[str]) -> list[PatentRecord]:
    """Compatibility wrapper — builds a minimal ExtractedTokens from a keyword list."""
    from services.token_extractor import ExtractedTokens as _ET
    if not query_terms:
        return []
    # Use the longest term as the primary anchor (best proxy for specificity)
    sorted_terms = sorted(query_terms, key=len, reverse=True)
    stub = _ET(
        primary_anchor  = sorted_terms[0],
        primary_concept = "unknown",
        critical_tokens = sorted_terms,
        epo_search_order= sorted_terms,
    )
    return epo_fetch_by_keywords(stub)
