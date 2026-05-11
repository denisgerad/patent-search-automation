"""
services/relevance_filter.py

Lightweight binary relevance pre-filter that uses Claude to remove patents
that are clearly outside the query's technical domain before the expensive
full comparison stage.

Why this exists:
  Even after fixing BGE prefix alignment and applying dynamic thresholds,
  high-dimensional embedding spaces can pass off-domain patents with scores
  above the threshold (e.g. "federated learning" leaking into "infrared
  sensors" queries due to shared general vocabulary).  A cheap Claude
  binary classification on the top-N candidates catches remaining drift
  without running the full comparison.
"""
import json
import logging

from models.claude_client import ClaudeClient
from models.schemas import RankedPatent

logger = logging.getLogger(__name__)

_FILTER_PROMPT = """\
You are a patent relevance classifier.

Query domain: {domain_concepts}
Original query: {original_query}

For each patent below, respond with a JSON array of objects:
{{"patent_id": "...", "relevant": true/false, "reason": "one sentence"}}

Only mark relevant=true if the patent is genuinely in the same technical
domain as the query. Be strict — surface keyword overlap is not enough.

Patents:
{patent_list}

Return ONLY the JSON array, no other text."""


def filter_irrelevant_patents(
    query: str,
    domain_concepts: list[str],
    ranked: list[RankedPatent],
    client: ClaudeClient,
    max_to_filter: int = 20,
) -> list[RankedPatent]:
    """
    Use Claude as a lightweight binary relevance filter before the full
    comparison stage.  Runs on the top-*max_to_filter* candidates only
    to control API cost.

    On any Claude error the original list is returned unchanged so the
    pipeline never fails due to the filter.

    Args:
        query:           Original user query string.
        domain_concepts: High-level domain labels from the token extractor.
        ranked:          Sorted list of RankedPatent objects from ranking_service.
        client:          Initialised ClaudeClient instance.
        max_to_filter:   Maximum number of candidates to send to Claude.

    Returns:
        Filtered list of RankedPatent objects (order preserved).
    """
    if not ranked:
        return ranked

    candidates = ranked[:max_to_filter]

    patent_list = "\n".join(
        f"- ID: {r.patent.patent_id} | Title: {r.patent.patent_title}"
        for r in candidates
    )

    prompt = _FILTER_PROMPT.format(
        original_query=query,
        domain_concepts=", ".join(domain_concepts) if domain_concepts else "not specified",
        patent_list=patent_list,
    )

    try:
        raw = client.complete(
            system="You are a strict patent relevance classifier. Return only valid JSON.",
            user=prompt,
            max_tokens=1000,
        )
        decisions = json.loads(raw)
        relevant_ids = {
            d["patent_id"] for d in decisions if d.get("relevant")
        }
        filtered = [r for r in candidates if r.patent.patent_id in relevant_ids]
        # Append any candidates beyond max_to_filter unchanged
        filtered.extend(ranked[max_to_filter:])
        logger.info(
            "Relevance filter: %d → %d patents (from top %d)",
            len(candidates), len(filtered), max_to_filter,
        )
        return filtered

    except Exception as exc:
        logger.warning("Relevance filter failed (%s) — returning unfiltered results.", exc)
        return ranked
