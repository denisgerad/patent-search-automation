"""
services/query_expansion.py

Expands a user query into alternative technical search terms using Mistral.
The prompt template is loaded from prompts/query_expansion.txt.
To change what Mistral is asked to do, edit that file — do not modify this service.

Constrained expansion (fix5):
  1. :mod:`services.token_extractor` deterministically identifies critical
     domain anchors from the query before calling Mistral.
  2. Those anchors are injected into the prompt as hard constraints so Mistral
     cannot drift outside the required technology domain.
  3. The caller receives both the expanded query list AND the
     :class:`~services.token_extractor.ExtractedTokens` so the pipeline can
     run :mod:`services.constraint_validator` as a post-filter.
"""
import json
import logging
import sys
from pathlib import Path

# Ensure the project root is importable regardless of the launch directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.mistral_client import MistralClient                # noqa: E402
from services.token_extractor import (                         # noqa: E402
    ExtractedTokens,
    extract_critical_tokens,
)
from utils.prompt_loader import load_prompt                    # noqa: E402

logger = logging.getLogger(__name__)

# Load the constrained prompt template once at module import time.
# Template file: prompts/query_expansion.txt
# Placeholders: {original_query}, {critical_tokens}, {patent_synonyms}, {n_expansions}
PROMPT_TEMPLATE: str = load_prompt("query_expansion.txt")


def expand_query(
    query: str,
    client: MistralClient,
    n_expansions: int = 5,
) -> tuple[list[str], ExtractedTokens]:
    """
    Generate alternative search terms for *query* using Mistral with
    domain-anchored constraints.

    Args:
        query:        The original user query string.
        client:       An initialised MistralClient instance.
        n_expansions: Number of alternative queries to request from Mistral.

    Returns:
        A tuple ``(expanded_queries, tokens)`` where:
        * ``expanded_queries`` — list starting with the original query,
          followed by up to *n_expansions* AI-generated alternatives.
          Falls back to ``[query]`` on parse failure.
        * ``tokens`` — :class:`~services.token_extractor.ExtractedTokens`
          containing the mandatory anchors; pass to
          :func:`~services.constraint_validator.validate_queries`.
    """
    tokens = extract_critical_tokens(query)
    logger.info(
        "Extracted tokens — critical=%s  concepts=%s",
        tokens.critical_tokens,
        tokens.domain_concepts,
    )

    prompt = PROMPT_TEMPLATE.format(
        original_query=query,
        critical_tokens=", ".join(tokens.critical_tokens) or query,
        patent_synonyms=", ".join(tokens.patent_synonyms) or "N/A",
        n_expansions=n_expansions,
    )

    logger.debug("Query expansion prompt sent to Mistral (query=%s)", query)
    raw = client.generate(prompt)
    logger.debug("Raw Mistral response: %s", raw)

    try:
        expanded = json.loads(raw)
        if not isinstance(expanded, list):
            raise ValueError("Expected JSON array from Mistral")
        term_list = [t for t in expanded if isinstance(t, str) and t.strip()]
        logger.info(
            "Query expanded: %d terms generated for '%s'", len(term_list), query
        )
    except (json.JSONDecodeError, ValueError):
        import re

        # Strip markdown code fences and retry
        cleaned = re.sub(r"```[\s\S]*?```", "", raw).strip()
        arr_start = cleaned.find("[")
        arr_end = cleaned.rfind("]")
        if arr_start != -1 and arr_end > arr_start:
            try:
                expanded = json.loads(cleaned[arr_start : arr_end + 1])
                term_list = [t for t in expanded if isinstance(t, str) and t.strip()]
                logger.info(
                    "Query expanded (after cleanup): %d terms for '%s'",
                    len(term_list),
                    query,
                )
            except (json.JSONDecodeError, TypeError):
                logger.warning(
                    "Query expansion parse failed — falling back to original query."
                )
                term_list = []
        else:
            logger.warning(
                "Query expansion parse failed — falling back to original query."
            )
            term_list = []

    all_queries = [query] + term_list
    return all_queries, tokens


def expand_query_with_metadata(
    query: str,
    client: MistralClient,
) -> tuple[list[str], dict]:
    """
    Thin backward-compatibility wrapper used by the Streamlit UI.

    Returns:
        ``(expanded_queries, metadata_dict)`` where *metadata_dict* contains
        the keys ``critical_tokens``, ``domain_concepts``, and
        ``patent_synonyms`` extracted from
        :class:`~services.token_extractor.ExtractedTokens`.
    """
    expanded, tokens = expand_query(query, client)
    metadata: dict = {
        "critical_tokens": tokens.critical_tokens,
        "domain_concepts": tokens.domain_concepts,
        "patent_synonyms": tokens.patent_synonyms,
    }
    return expanded, metadata
