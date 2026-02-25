"""
services/query_builder.py

Recall-focused query builder for PatentsView search.

Responsibility split (fix6 + Change 2 + Change 3):
  Search layer  → HIGH RECALL   Queries use _text_any so the API returns
                                 every patent that mentions the domain tokens.
                                 Precision constraints (AND / _text_phrase)
                                 are never applied at search time.
  Ranking layer → HIGH PRECISION ranking_service applies cosine threshold,
                                 BM25, and hybrid scoring to surface the
                                 most relevant results from the recall pool.

Three tiers differ in *what* is searched, not in strictness:

  Tier 1 BROAD   — Token-aware _and of _text_any per critical token.
                   Requires all domain anchors to co-occur in the abstract
                   (without phrase matching) — good recall + light structure.
  Tier 2 MEDIUM  — _text_any per patent synonym string.
  Tier 3 NARROW  — _text_any per Mistral-expanded phrase.

All tiers run; results are merged and handed to dedup + ranking.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.token_extractor import ExtractedTokens  # noqa: E402
from services.search_service import build_token_aware_query  # noqa: E402


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class QueryTier:
    """A named group of pre-built PatentsView query dicts at one specificity."""

    queries: list[dict]
    """List of PatentsView query dicts ready to pass to _fetch_all_for_query."""

    strategy: str
    """'broad' | 'medium' | 'narrow'"""

    description: str
    """Human-readable label for logging."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_tiered_queries(
    original: str,
    expanded: list[str],
    tokens: ExtractedTokens,
) -> list[QueryTier]:
    """
    Build three tiers of PatentsView query dicts with different specificity.

    Args:
        original: The raw user query string.
        expanded: Validated Mistral-expanded query strings (including original).
        tokens:   Critical tokens / synonyms from the deterministic extractor.

    Returns:
        A list of three :class:`QueryTier` objects ordered broad → narrow.
    """
    return [
        QueryTier(
            queries=_build_broad_queries(tokens),
            strategy="broad",
            description="Token-aware _and of _text_any per critical token",
        ),
        QueryTier(
            queries=_build_medium_queries(tokens),
            strategy="medium",
            description="_text_any per patent synonym",
        ),
        QueryTier(
            queries=_build_narrow_queries(expanded, tokens),
            strategy="narrow",
            description="_text_any per Mistral-expanded phrase",
        ),
    ]


# ---------------------------------------------------------------------------
# Tier builders
# ---------------------------------------------------------------------------

def _build_broad_queries(tokens: ExtractedTokens) -> list[dict]:
    """
    Single token-aware query: ``_and`` of ``_text_any`` per critical token.

    All domain anchors must co-occur somewhere in the patent abstract, but
    each token is matched as a bag-of-words (``_text_any``), not a phrase.
    This gives meaningful structure without sacrificing recall.

    Falls back to a plain OR query on the original user query when no
    critical tokens were extracted.
    """
    if tokens.critical_tokens:
        return [build_token_aware_query(tokens.critical_tokens)]
    # Fallback: no taxonomy match — broad OR on the raw query.
    return [{
        "_or": [
            {"_text_any": {"patent_title": tokens.original_query}},
            {"_text_any": {"patent_abstract": tokens.original_query}},
        ]
    }]


def _build_medium_queries(tokens: ExtractedTokens) -> list[dict]:
    """
    One _text_any query per Mistral-expanded string.

    Uses the original query strings directly — any word in the phrase can
    match anywhere in title or abstract.  High recall; precision is left
    entirely to the ranking layer.
    """
    queries: list[dict] = []
    for phrase in tokens.patent_synonyms[:4]:
        queries.append({
            "_or": [
                {"_text_any": {"patent_title": phrase}},
                {"_text_any": {"patent_abstract": phrase}},
            ]
        })
    return queries[:4]


def _build_narrow_queries(
    expanded: list[str],
    tokens: ExtractedTokens,
) -> list[dict]:
    """
    One _text_any query per Mistral-expanded string.

    Searches each expanded phrase as a bag-of-words across title and abstract.
    No phrase-match constraints — recall only.  Precision is handled downstream
    by the cosine + BM25 hybrid ranker.

    Only the first 3 expanded strings are used to keep API calls bounded.
    """
    queries: list[dict] = []
    for phrase in expanded[:3]:
        queries.append({
            "_or": [
                {"_text_any": {"patent_title": phrase}},
                {"_text_any": {"patent_abstract": phrase}},
            ]
        })
    return queries
