"""
services/query_expansion.py

Expands a user query into alternative technical search terms using Mistral.
The prompt template is loaded from prompts/query_expansion.txt.
To change what Mistral is asked to do, edit that file — do not modify this service.
"""
import json
import logging
import sys
from pathlib import Path

# Ensure the project root is importable regardless of the launch directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.mistral_client import MistralClient      # noqa: E402
from utils.prompt_loader import load_prompt          # noqa: E402

logger = logging.getLogger(__name__)

# Load the prompt template once at module import time.
# Template file: prompts/query_expansion.txt
# Uses Python str.format() placeholders — currently only {query}.
PROMPT_TEMPLATE: str = load_prompt("query_expansion.txt")


def expand_query(query: str, client: MistralClient) -> list[str]:
    """
    Generate alternative search terms for *query* using Mistral.

    Args:
        query:  The original user query string.
        client: An initialised MistralClient instance.

    Returns:
        A list starting with the original query followed by up to 5
        AI-generated alternatives.  Falls back to [query] on parse failure.
    """
    prompt = PROMPT_TEMPLATE.format(query=query)
    logger.debug("Query expansion prompt sent to Mistral (query=%s)", query)

    raw = client.generate(prompt)
    logger.debug("Raw Mistral response: %s", raw)

    try:
        terms = json.loads(raw)
        if not isinstance(terms, list):
            raise ValueError("Expected a JSON array, got: %s" % type(terms))
        expanded = [query] + [t for t in terms if isinstance(t, str)]
        logger.info("Query expanded: %d terms generated for '%s'", len(expanded) - 1, query)
        return expanded
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning(
            "Query expansion parse failed (%s). Falling back to original query.", exc
        )
        return [query]
