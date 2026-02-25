"""
services/constraint_validator.py

Filters expanded search queries to prevent LLM-induced domain drift.

After Mistral expands the user query it sometimes drifts toward generic
technical phrases that lose the required domain anchors.  This module
checks each candidate query against the mandatory tokens extracted by
:mod:`services.token_extractor` and silently drops any query that has
lost all domain anchors.
"""
import logging

from services.token_extractor import ExtractedTokens

log = logging.getLogger(__name__)


def validate_queries(queries: list[str], tokens: ExtractedTokens) -> list[str]:
    """
    Remove expanded queries that lost all critical domain anchors.

    A query is *valid* when it contains at least one term from either
    ``tokens.critical_tokens`` **or** ``tokens.patent_synonyms``.

    Args:
        queries: Candidate query strings (original + Mistral expansions).
        tokens:  :class:`~services.token_extractor.ExtractedTokens` returned
                 by :func:`~services.token_extractor.extract_critical_tokens`.

    Returns:
        Filtered list of queries.  Guaranteed to contain at least the
        original query — even if every expansion failed validation.
    """
    all_anchors = {
        t.lower()
        for t in tokens.critical_tokens + tokens.patent_synonyms
    }

    validated: list[str] = []
    for q in queries:
        q_lower = q.lower()
        if any(anchor in q_lower for anchor in all_anchors):
            validated.append(q)
        else:
            log.warning(
                "Query failed constraint validation — dropped: %r  (anchors: %s)",
                q,
                list(all_anchors),
            )

    # Safety net: always keep the original query
    if not validated:
        log.warning(
            "All expansions failed constraint validation — falling back to original query."
        )
        validated = [tokens.original_query]

    log.info(
        "Constraint validation: %d in → %d out", len(queries), len(validated)
    )
    return validated
