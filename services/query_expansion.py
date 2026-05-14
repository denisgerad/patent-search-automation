"""
services/query_expansion.py

Pipeline:
  Step A — Deterministic token extraction (token_extractor.py)
            Sets primary_token when single taxonomy concept matches.
            Always sets epo_search_order via _specificity_sort() as a baseline.

  Step B — Claude pre-call: identify_primary_token
            Called when primary_token is empty (multi-concept or unknown domain).
            Returns primary_token, supporting_tokens, AND epo_search_order
            (Claude's context-aware ordering overwrites the deterministic baseline).

  Step C — Claude expansion call
            Uses tiered prompt: PRIMARY ANCHOR (hard) + SUPPORTING TOKENS (soft).
            Every expansion must contain primary_token or its direct synonym.

  primary_token    → used by expansion prompt and constraint validator
  epo_search_order → used by EPO CQL builder (most discriminating term first)
  These two serve different purposes and are kept separate.
"""
import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.claude_client import ClaudeClient
from services.token_extractor import ExtractedTokens, extract_critical_tokens, _specificity_sort
from utils.prompt_loader import load_prompt

logger = logging.getLogger(__name__)

EXPANSION_TEMPLATE: str = load_prompt("query_expansion.txt")
PRIMARY_TOKEN_TEMPLATE: str = load_prompt("identify_primary_token.txt")

_SYSTEM_JSON = (
    "You are a patent search expert. Return only valid JSON. "
    "No markdown, no explanation, no preamble."
)


# ---------------------------------------------------------------------------
# Step B — Claude pre-call
# ---------------------------------------------------------------------------

def _identify_primary_token(
    query: str,
    client: ClaudeClient,
) -> tuple[str, list[str], list[str]]:
    """
    Ask Claude to identify the inventive concept AND the EPO search order.

    Returns:
        (primary_token, supporting_tokens, epo_search_order)
        All three default to empty on parse failure.
    """
    prompt = PRIMARY_TOKEN_TEMPLATE.format(query=query)
    logger.debug("Primary token pre-call: %s", query)

    try:
        raw = client.complete(system=_SYSTEM_JSON, user=prompt, max_tokens=300)
        logger.debug("Pre-call raw: %s", raw)

        cleaned = re.sub(r"```[\s\S]*?```", "", raw).strip()
        obj_start = cleaned.find("{")
        obj_end   = cleaned.rfind("}") + 1
        if obj_start == -1 or obj_end == 0:
            raise ValueError("No JSON object in response")

        parsed = json.loads(cleaned[obj_start:obj_end])
        primary   = str(parsed.get("primary_token", "")).strip().lower()
        supporting = [
            str(t).strip().lower()
            for t in parsed.get("supporting_tokens", [])
            if str(t).strip()
        ]
        epo_order = [
            str(t).strip().lower()
            for t in parsed.get("epo_search_order", [])
            if str(t).strip()
        ]
        reasoning = parsed.get("reasoning", "")

        logger.info(
            "Pre-call result — primary='%s'  epo_order=%s  reason: %s",
            primary, epo_order, reasoning,
        )
        return primary, supporting, epo_order

    except (json.JSONDecodeError, ValueError, KeyError) as exc:
        logger.warning("Pre-call parse failed (%s) — using fallback.", exc)
        return "", [], []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def expand_query(
    query: str,
    client: ClaudeClient,
    n_expansions: int = 5,
) -> tuple[list[str], ExtractedTokens, str]:
    """
    Steps A → B → C.  Returns (expanded_queries, tokens, raw_expansion).
    tokens.epo_search_order is always populated for the EPO CQL builder.
    """
    # ── Step A ───────────────────────────────────────────────────────────
    tokens = extract_critical_tokens(query)
    logger.info(
        "Step A — primary='%s'  supporting=%s  epo_order=%s",
        tokens.primary_token, tokens.supporting_tokens, tokens.epo_search_order,
    )

    # ── Step B ───────────────────────────────────────────────────────────
    if not tokens.primary_token:
        logger.info("primary_token empty — running Claude pre-call")
        claude_primary, claude_supporting, claude_epo_order = \
            _identify_primary_token(query, client)

        if claude_primary:
            tokens.primary_token = claude_primary

            # Merge supporting tokens then clean:
            # - deduplicate substrings ('infrared' dropped when 'infrared sensors' present)
            # - remove the primary token itself from the supporting list
            # - remove bare generic words that add no EPO precision
            merged = list(dict.fromkeys(
                claude_supporting + tokens.supporting_tokens
            ))
            tokens.supporting_tokens = _clean_term_list(merged, exclude=claude_primary)

            # epo_search_order: Claude's ordering cleaned and capped at 4.
            # Fallback to specificity sort if Claude didn't return one.
            if claude_epo_order:
                all_known = list(dict.fromkeys(
                    claude_epo_order + tokens.supporting_tokens
                ))
                tokens.epo_search_order = _clean_term_list(all_known)[:4]
            else:
                tokens.epo_search_order = _specificity_sort(
                    _clean_term_list(
                        list(dict.fromkeys(
                            [tokens.primary_token] + tokens.supporting_tokens
                        ))
                    )
                )[:4]

            # Keep critical_tokens in sync
            tokens.critical_tokens = list(dict.fromkeys(
                [tokens.primary_token] + tokens.supporting_tokens
            ))
            logger.info(
                "Step B complete — primary='%s'  epo_order=%s",
                tokens.primary_token, tokens.epo_search_order,
            )
        else:
            # Claude pre-call failed: use specificity sort as fallback
            fallback = (tokens.supporting_tokens or [None])[0]
            if not fallback:
                fallback = next(
                    (w for w in query.lower().split() if len(w) > 5),
                    query.split()[0],
                )
            tokens.primary_token = fallback
            tokens.epo_search_order = _specificity_sort(
                _clean_term_list(tokens.critical_tokens)
            )[:4]
            logger.warning(
                "Pre-call failed — fallback primary='%s'  epo_order=%s",
                tokens.primary_token, tokens.epo_search_order,
            )

    # ── Step C ───────────────────────────────────────────────────────────
    supporting_str = ", ".join(tokens.supporting_tokens) if tokens.supporting_tokens else "N/A"
    synonyms_str   = ", ".join(tokens.patent_synonyms)   if tokens.patent_synonyms   else "N/A"

    prompt = EXPANSION_TEMPLATE.format(
        original_query=query,
        primary_token=tokens.primary_token,
        supporting_tokens=supporting_str,
        patent_synonyms=synonyms_str,
        n_expansions=n_expansions,
    )

    raw = client.complete(system=_SYSTEM_JSON, user=prompt, max_tokens=512)
    logger.debug("Expansion raw: %s", raw)

    term_list = _parse_json_array(raw, query)
    logger.info(
        "Step C — %d expansions for '%s'  (primary='%s'  epo_order=%s)",
        len(term_list), query, tokens.primary_token, tokens.epo_search_order,
    )

    return [query] + term_list, tokens, raw


def expand_query_with_metadata(
    query: str,
    client: ClaudeClient,
) -> tuple[list[str], dict]:
    """Backward-compatibility wrapper for the Streamlit UI."""
    expanded, tokens, raw = expand_query(query, client)
    metadata: dict = {
        "primary_token":      tokens.primary_token,
        "supporting_tokens":  tokens.supporting_tokens,
        "epo_search_order":   tokens.epo_search_order,
        "critical_tokens":    tokens.critical_tokens,
        "domain_concepts":    tokens.domain_concepts,
        "patent_synonyms":    tokens.patent_synonyms,
        "claude_raw_response": raw,
    }
    return expanded, metadata


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_json_array(raw: str, original_query: str) -> list[str]:
    try:
        expanded = json.loads(raw)
        if isinstance(expanded, list):
            return [t for t in expanded if isinstance(t, str) and t.strip()]
        raise ValueError("Expected JSON array")
    except (json.JSONDecodeError, ValueError):
        cleaned = re.sub(r"```[\s\S]*?```", "", raw).strip()
        arr_start = cleaned.find("[")
        arr_end   = cleaned.rfind("]")
        if arr_start != -1 and arr_end > arr_start:
            try:
                expanded = json.loads(cleaned[arr_start:arr_end + 1])
                return [t for t in expanded if isinstance(t, str) and t.strip()]
            except (json.JSONDecodeError, TypeError):
                pass
        logger.warning("Expansion parse failed for '%s'", original_query)
        return []


# Single-word terms that add no discriminating power to EPO CQL.
# These words are fine inside multi-word phrases but harmful as standalone terms.
_GENERIC_STANDALONE: set[str] = {
    "system", "method", "device", "apparatus", "sensor", "sensors",
    "vehicle", "vehicles", "using", "based", "data", "network",
    "model", "module", "unit", "process", "technology", "application",
    "detection", "recognition", "navigation", "control", "management",
}


def _clean_term_list(terms: list[str], exclude: str = "") -> list[str]:
    """
    Clean a list of token terms before use in EPO CQL or display:

    1. Substring deduplication: if term A is contained within term B,
       drop A (keep the longer, more specific phrase).
       e.g. 'infrared' dropped when 'infrared sensors' is present.

    2. Generic standalone filter: single-word terms in _GENERIC_STANDALONE
       are removed because they match too broadly on EPO.
       e.g. 'sensors', 'system', 'vehicle' dropped.

    3. Remove *exclude* (typically the primary_token) from the list
       so it doesn't duplicate in supporting_tokens.

    4. Deduplicate while preserving order.
    """
    if not terms:
        return []

    lower_exclude = exclude.lower().strip()

    # Step 1: substring deduplication
    deduped: list[str] = []
    for t in terms:
        dominated = any(
            t != other and t.lower() in other.lower()
            for other in terms
        )
        if not dominated:
            deduped.append(t)

    # Step 2: remove generic standalones (single-word only — keep phrases)
    filtered: list[str] = []
    for t in deduped:
        is_single_word = len(t.split()) == 1
        if is_single_word and t.lower() in _GENERIC_STANDALONE:
            continue
        filtered.append(t)

    # Step 3: remove primary token duplicate + dedup
    seen: set[str] = set()
    result: list[str] = []
    for t in filtered:
        key = t.lower().strip()
        if key == lower_exclude:
            continue
        if key not in seen:
            seen.add(key)
            result.append(t)

    return result
