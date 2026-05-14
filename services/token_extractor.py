"""
services/token_extractor.py

Deterministic, rule-based token extraction for patent query anchoring.

Key design:
  primary_token     — the inventive concept; constrains every Claude expansion
  supporting_tokens — secondary anchors; soft constraint on expansions
  epo_search_order  — terms ordered by discriminating power for EPO CQL
                      (most discriminating first = smallest EPO result set)

  primary_token is for EXPANSION QUALITY.
  epo_search_order is for EPO SEARCH PRECISION.
  These are different things and must not be conflated.

  When taxonomy matches multiple concepts (common case), primary_token is
  left empty so the Claude pre-call can identify it from the full query
  context.  epo_search_order is always set: either from Claude pre-call
  or by the deterministic specificity scorer below.
"""
import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Suppression list — words that are never the inventive concept
# ---------------------------------------------------------------------------
_SUPPRESSED: set[str] = {
    "system", "method", "device", "apparatus", "mechanism", "assembly",
    "module", "component", "unit", "structure", "arrangement", "means",
    "process", "procedure", "technique", "approach", "solution",
    "optical", "digital", "electronic", "electric", "electrical",
    "advanced", "improved", "enhanced", "efficient", "effective",
    "automatic", "automated", "intelligent", "smart", "dynamic",
    "integrated", "embedded", "novel", "high", "low",
    "maintaining", "providing", "improving", "increasing",
    "reducing", "detecting", "measuring", "controlling", "monitoring",
    "processing", "generating", "using", "based", "having", "comprising",
    "vehicle", "screen", "display", "network", "circuit", "signal",
    "data", "information", "output", "input", "image", "video",
    "sensor", "camera", "property", "performance", "quality",
}

# Generic single-word sensor/tech terms — discriminating power is LOW
# because they appear in many unrelated patent domains
_LOW_DISCRIMINATING: set[str] = {
    "infrared", "radar", "lidar", "ultrasonic", "optical", "laser",
    "sensor", "camera", "detector", "imaging", "vision",
}

# ---------------------------------------------------------------------------
# Domain taxonomy
# ---------------------------------------------------------------------------
DOMAIN_TAXONOMY: dict[str, dict] = {
    "sensor": {
        "terms": [
            "infrared", "lidar", "radar", "ultrasonic",
            "ir sensor", "thermal sensor", "depth sensor",
        ],
        "patent_synonyms": ["sensing device", "detector", "transducer"],
    },
    "autonomous_vehicle": {
        "terms": [
            "autonomous vehicle", "self-driving", "ego vehicle",
            "driverless", "adas", "advanced driver assistance",
            "autonomous driving",
        ],
        "patent_synonyms": ["autonomous driving system", "vehicle control system"],
    },
    "detection": {
        "terms": [
            "lane detection", "pedestrian detection", "object detection",
            "crosswalk detection", "crossing detection",
            "pedestrian crossing",
        ],
        "patent_synonyms": [
            "recognition system", "identification method", "classification system",
        ],
    },
    "imaging": {
        "terms": [
            "camera-based", "vision", "image processing",
            "computer vision",
        ],
        "patent_synonyms": ["imaging system", "visual sensor", "optical detector"],
    },
}


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------
@dataclass
class ExtractedTokens:
    """Structured result from :func:`extract_critical_tokens`."""

    primary_token: str = ""
    """Inventive concept — constrains every Claude expansion query.
    Set by taxonomy (single match) or Claude pre-call (multi-match / unknown domain).
    """

    supporting_tokens: list[str] = field(default_factory=list)
    """Secondary anchors — soft constraint on expansion."""

    epo_search_order: list[str] = field(default_factory=list)
    """Terms ordered by discriminating power for EPO CQL.
    Most discriminating (multi-word, domain-specific) first.
    Set by Claude pre-call when available, otherwise by _specificity_sort().
    This is what _build_cql() should iterate over — NOT critical_tokens.
    """

    critical_tokens: list[str] = field(default_factory=list)
    """Union of primary + supporting — kept for backward compat with
    constraint_validator, ranking_service, and coverage scoring.
    """

    domain_concepts: list[str] = field(default_factory=list)
    patent_synonyms: list[str] = field(default_factory=list)
    original_query: str = ""
    concept_groups: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_critical_tokens(query: str) -> ExtractedTokens:
    """
    Deterministically extract domain-anchoring tokens from *query*.

    epo_search_order is always populated here using _specificity_sort().
    The Claude pre-call in query_expansion.py may overwrite it with a
    better-ordered list from the full query context.
    """
    query_lower = query.lower()
    critical_tokens: list[str] = []
    domain_concepts: list[str] = []
    patent_synonyms: list[str] = []
    concept_groups: dict = {}
    matched_concepts: list[str] = []

    for concept, data in DOMAIN_TAXONOMY.items():
        matched_terms = [t for t in data["terms"] if t in query_lower]
        if matched_terms:
            critical_tokens.extend(matched_terms)
            domain_concepts.append(concept)
            patent_synonyms.extend(data["patent_synonyms"])
            concept_groups[concept] = {
                "terms": data["terms"],
                "patent_synonyms": data["patent_synonyms"],
            }
            matched_concepts.append(concept)

    primary_token = ""
    supporting_tokens: list[str] = []

    if critical_tokens:
        deduped = list(dict.fromkeys(critical_tokens))
        if len(matched_concepts) == 1:
            primary_token = deduped[0]
            supporting_tokens = deduped[1:]
        else:
            # Multiple concepts: leave primary_token empty for Claude pre-call
            supporting_tokens = deduped
    else:
        ranked = _rarity_scored_extraction(query)
        if ranked:
            primary_token = ranked[0]
            supporting_tokens = ranked[1:]
        critical_tokens = ranked

    all_critical = list(dict.fromkeys(
        ([primary_token] if primary_token else []) + supporting_tokens
    ))

    # epo_search_order: sort all matched terms by discriminating power.
    # Claude pre-call will overwrite this with a context-aware ordering.
    epo_order = _specificity_sort(all_critical)

    return ExtractedTokens(
        primary_token=primary_token,
        supporting_tokens=list(dict.fromkeys(supporting_tokens)),
        epo_search_order=epo_order,
        critical_tokens=all_critical,
        domain_concepts=list(dict.fromkeys(domain_concepts)),
        patent_synonyms=list(dict.fromkeys(patent_synonyms)),
        original_query=query,
        concept_groups=concept_groups,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _specificity_sort(terms: list[str]) -> list[str]:
    """
    Sort *terms* by discriminating power for EPO CQL, most discriminating first.

    Scoring:
      +10 per word in the term       (multi-word phrases are more specific)
      +8  if not in _LOW_DISCRIMINATING
      +len(term)                     (longer = rarer = more specific)
    """
    def _score(term: str) -> int:
        words = term.split()
        s = 10 * len(words) + len(term)
        if term.lower() not in _LOW_DISCRIMINATING:
            s += 8
        return s

    return sorted(terms, key=_score, reverse=True)


def _rarity_scored_extraction(query: str) -> list[str]:
    """
    Score each word in *query* by lexical rarity and return sorted candidates.
    Used only when taxonomy has no match (completely unknown domain).
    """
    _RARE_SUFFIXES = (
        "ization", "isation", "ectomy", "ometry",
        "ology", "otropy", "fluence", "escence", "ography",
        "olysis", "ogenesis", "ification", "otropic", "philic",
    )

    words = re.findall(r"\b[\w-]+\b", query.lower())
    scored: list[tuple[float, str]] = []

    for w in words:
        if len(w) < 5:
            continue
        base = w.replace("-", "")
        if base in _SUPPRESSED or w in _SUPPRESSED:
            continue

        score: float = len(w) + 8
        if any(w.endswith(sfx) for sfx in _RARE_SUFFIXES):
            score += 5
        if "-" in w:
            score += 3

        scored.append((score, w))

    seen: set[str] = set()
    ranked: list[str] = []
    for _, w in sorted(scored, reverse=True):
        if w not in seen:
            seen.add(w)
            ranked.append(w)

    return ranked[:5]
