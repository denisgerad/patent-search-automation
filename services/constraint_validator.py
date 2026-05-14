"""
services/constraint_validator.py

Filters expanded search queries to prevent LLM-induced domain drift.

Updated for primary-token fix:
  Validation is now TIERED:
    1. PRIMARY check  — query must contain tokens.primary_token or a direct
                        synonym of it.  This is the hard gate.
    2. SUPPORTING check — if primary passes, query should also contain at
                          least one supporting_token or patent_synonym.
                          Queries that pass (1) but fail (2) are kept but
                          logged as "weak" so the pipeline can deprioritise them.

  A query that loses the primary_token entirely is always dropped.
  The original query is always kept as a safety net.
"""
import logging

from services.token_extractor import ExtractedTokens

log = logging.getLogger(__name__)


def validate_queries(queries: list[str], tokens: ExtractedTokens) -> list[str]:
    """
    Remove expanded queries that lost the primary anchor.

    Args:
        queries: Candidate query strings (original + expanded).
        tokens:  ExtractedTokens from extract_critical_tokens / Claude pre-call.

    Returns:
        Filtered list.  Always contains at least the original query.
    """
    primary = tokens.primary_token.lower() if tokens.primary_token else None

    # Build synonym set for the primary token (direct patent synonyms only)
    primary_synonyms: set[str] = set()
    if primary:
        for syn in tokens.patent_synonyms:
            # Include a synonym if it contains or is contained by the primary token
            if primary in syn.lower() or syn.lower() in primary:
                primary_synonyms.add(syn.lower())

    # Supporting anchors (soft check)
    supporting_anchors: set[str] = {
        t.lower() for t in tokens.supporting_tokens + tokens.patent_synonyms
    }

    validated: list[str] = []
    weak: list[str] = []

    for q in queries:
        q_lower = q.lower()

        # Hard gate: primary token must be present
        if primary:
            primary_present = (
                primary in q_lower
                or any(syn in q_lower for syn in primary_synonyms)
            )
            if not primary_present:
                log.warning(
                    "Query dropped (missing primary '%s'): %r",
                    primary, q,
                )
                continue

        # Soft check: at least one supporting anchor
        if supporting_anchors and not any(a in q_lower for a in supporting_anchors):
            log.info("Query kept but weak (missing all supporting anchors): %r", q)
            weak.append(q)
        else:
            validated.append(q)

    # Combine: strong queries first, weak queries appended
    all_valid = validated + weak

    if not all_valid:
        log.warning(
            "All expansions failed primary constraint — falling back to original."
        )
        all_valid = [tokens.original_query]

    log.info(
        "Constraint validation: %d in → %d strong + %d weak = %d out "
        "(primary='%s')",
        len(queries), len(validated), len(weak), len(all_valid),
        primary or "none",
    )
    return all_valid
