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
# Discriminating power score per concept type.
# Lower score = more specific = better primary anchor.
# ---------------------------------------------------------------------------
CONCEPT_DISCRIMINATING_POWER: dict[str, int] = {
    "task":           1,   # most specific — "lane detection", "collision avoidance"
    "detection":      1,
    "application":    2,   # "lane marking", "pedestrian crossing"
    "domain":         3,   # "autonomous vehicle", "self-driving"
    "sensor":         4,   # "infrared", "lidar" — broad, many patents
    "imaging":        4,
    "generic_method": 5,   # "recognition", "detection" alone — too broad
}

# ---------------------------------------------------------------------------
# Domain taxonomy
# ---------------------------------------------------------------------------
DOMAIN_TAXONOMY: dict[str, dict] = {
    # ── Task concepts (most discriminating) ──────────────────────────
    "lane_detection": {
        "type": "task",
        "terms": [
            "lane detection", "lane recognition", "lane tracking",
            "lane boundary detection", "lane marking recognition",
            "road marking detection", "lane keeping",
        ],
        "patent_synonyms": [
            "lane boundary recognition", "road lane identification",
            "traffic lane detection", "lane departure detection",
        ],
    },
    "fatigue_detection": {
        "type": "task",
        "terms": [
            "fatigue detection", "drowsiness detection", "driver fatigue",
            "driver alertness", "driver monitoring",
        ],
        "patent_synonyms": [
            "driver state monitoring", "operator alertness system",
            "drowsiness monitoring system",
        ],
    },
    "pedestrian_detection": {
        "type": "task",
        "terms": [
            "pedestrian detection", "pedestrian crossing detection",
            "crosswalk detection", "pedestrian recognition",
        ],
        "patent_synonyms": [
            "pedestrian identification system", "crosswalk recognition",
        ],
    },
    "collision_avoidance": {
        "type": "task",
        "terms": [
            "collision avoidance", "collision detection", "obstacle avoidance",
            "forward collision warning",
        ],
        "patent_synonyms": [
            "collision prevention system", "obstacle detection system",
        ],
    },

    # ── Domain concepts ───────────────────────────────────────────────
    "autonomous_vehicle": {
        "type": "domain",
        "terms": [
            "autonomous vehicle", "self-driving", "driverless",
            "autonomous car", "autonomous driving", "adas",
            "advanced driver assistance", "ego vehicle",
        ],
        "patent_synonyms": [
            "autonomous driving system", "vehicle control system",
            "ego vehicle", "automated vehicle",
        ],
    },

    # ── Sensor concepts (least discriminating — appear everywhere) ────
    "infrared_sensor": {
        "type": "sensor",
        "terms": [
            "infrared", "infrared sensor", "ir sensor", "thermal sensor",
            "thermal camera", "thermal imaging", "infrared camera",
            "thermal detector",
        ],
        "patent_synonyms": [
            "infrared detector", "thermal sensing device",
            "IR imaging system", "thermal imager",
        ],
    },
    "camera_sensor": {
        "type": "sensor",
        "terms": [
            "camera", "camera-based", "camera sensor", "vision sensor",
            "optical sensor", "imaging sensor",
        ],
        "patent_synonyms": [
            "optical detector", "visual sensor", "imaging device",
        ],
    },
    "lidar_sensor": {
        "type": "sensor",
        "terms": ["lidar", "lidar sensor", "laser scanner", "laser sensor"],
        "patent_synonyms": ["laser detection system", "lidar array"],
    },
    "radar_sensor": {
        "type": "sensor",
        "terms": ["radar", "radar sensor", "radar-based"],
        "patent_synonyms": ["radio detection system", "radar array"],
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
    primary_anchor: str = ""
    """Most discriminating taxonomy-matched term — used as the CQL primary anchor."""
    primary_concept: str = ""
    """Taxonomy concept name of the primary anchor (e.g. 'lane_detection')."""
    taxonomy_miss: bool = False
    """True when no taxonomy concept matched — fallback extraction was used."""


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
    matched: list[tuple] = []   # (concept_name, concept_data, matched_terms)

    for concept, data in DOMAIN_TAXONOMY.items():
        matched_terms = [t for t in data["terms"] if t in query_lower]
        if matched_terms:
            matched.append((concept, data, matched_terms))

    if not matched:
        return _fallback_extraction(query)

    # Sort matched concepts by discriminating power (lower score = more specific)
    matched.sort(
        key=lambda x: CONCEPT_DISCRIMINATING_POWER.get(x[1]["type"], 99)
    )

    # Primary anchor = most discriminating matched concept
    primary_concept_name, primary_data, primary_terms = matched[0]

    # Build flat lists for downstream use
    critical_tokens: list[str] = []
    patent_synonyms: list[str] = []
    domain_concepts: list[str] = []
    concept_groups: dict = {}

    for concept, data, terms in matched:
        critical_tokens.extend(terms)
        patent_synonyms.extend(data["patent_synonyms"])
        domain_concepts.append(concept)
        concept_groups[concept] = {
            "type":            data["type"],
            "terms":           data["terms"],
            "patent_synonyms": data["patent_synonyms"],
            "matched":         terms,
        }

    # Backward-compat: primary_token / supporting_tokens used by expansion prompt
    primary_token = ""
    supporting_tokens: list[str] = []
    if len(matched) == 1:
        primary_token = primary_terms[0]
        supporting_tokens = list(dict.fromkeys(critical_tokens))[1:]
    else:
        # Multiple concepts: leave primary_token empty for Claude pre-call
        supporting_tokens = list(dict.fromkeys(critical_tokens))

    all_critical = list(dict.fromkeys(
        ([primary_token] if primary_token else []) + supporting_tokens
    ))

    epo_order = _specificity_sort(all_critical)

    return ExtractedTokens(
        primary_token    = primary_token,
        supporting_tokens= list(dict.fromkeys(supporting_tokens)),
        epo_search_order = epo_order,
        critical_tokens  = list(dict.fromkeys(critical_tokens)),
        domain_concepts  = domain_concepts,
        patent_synonyms  = list(dict.fromkeys(patent_synonyms)),
        original_query   = query,
        concept_groups   = concept_groups,
        primary_anchor   = primary_terms[0],        # most discriminating taxonomy term
        primary_concept  = primary_concept_name,    # taxonomy concept name
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fallback_extraction(query: str) -> ExtractedTokens:
    """
    No taxonomy match — extract noun phrases heuristically.
    Sets taxonomy_miss=True so the pipeline can warn the user to extend the taxonomy.
    """
    import logging as _logging
    _log = _logging.getLogger(__name__)

    tokens = _rarity_scored_extraction(query)

    _log.warning(
        "No taxonomy match for query — fallback extraction used. "
        "Add missing concepts to DOMAIN_TAXONOMY in token_extractor.py",
    )

    primary_token = tokens[0] if tokens else ""
    supporting    = tokens[1:] if len(tokens) > 1 else []

    return ExtractedTokens(
        primary_token     = primary_token,
        supporting_tokens = supporting,
        epo_search_order  = _specificity_sort(tokens),
        critical_tokens   = tokens,
        domain_concepts   = [],
        patent_synonyms   = [],
        original_query    = query,
        concept_groups    = {},
        primary_anchor    = primary_token,
        primary_concept   = "unknown",
        taxonomy_miss     = True,
    )


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
