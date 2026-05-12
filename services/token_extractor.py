"""
services/token_extractor.py

Deterministic, rule-based token extraction for patent query anchoring.
No LLM involved — uses a domain taxonomy to identify which tokens in the
user query are non-negotiable domain anchors that must survive expansion.
"""
import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Domain taxonomy
# Maps surface forms to canonical patent concepts.
# Grow this dict to cover new technology domains over time.
# ---------------------------------------------------------------------------
DOMAIN_TAXONOMY: dict[str, dict] = {
    "sensor": {
        "terms": [
            "infrared", "lidar", "radar", "camera", "ultrasonic",
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
            "camera", "camera-based", "vision", "image processing",
            "computer vision", "optical",
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

    critical_tokens: list[str] = field(default_factory=list)
    """Tokens that MUST appear (or their synonyms) in every search query."""

    domain_concepts: list[str] = field(default_factory=list)
    """High-level domain labels matched from the taxonomy (e.g. 'detection')."""

    patent_synonyms: list[str] = field(default_factory=list)
    """Alternative patent vocabulary for the matched concepts."""

    original_query: str = ""
    """The raw user query that was analysed."""

    concept_groups: dict = field(default_factory=dict)
    """Per-concept grouping used for coverage scoring.
    Shape: {"sensor": {"terms": [...], "patent_synonyms": [...]}, ...}
    """


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_critical_tokens(query: str) -> ExtractedTokens:
    """
    Deterministically extract domain-anchoring tokens from *query*.

    Sweeps the DOMAIN_TAXONOMY against the lowercased query text.  Any term
    that matches becomes a *critical_token*.  Its parent concept is added to
    *domain_concepts* and its patent synonyms to *patent_synonyms*.

    Falls back to :func:`_fallback_noun_extraction` when no taxonomy entry
    matches (e.g. a brand-new domain not yet represented in the taxonomy).

    Returns:
        :class:`ExtractedTokens` — all fields deduplicated, taxonomy order
        preserved otherwise.
    """
    query_lower = query.lower()
    critical_tokens: list[str] = []
    domain_concepts: list[str] = []
    patent_synonyms: list[str] = []
    concept_groups: dict = {}

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

    # Fallback: extract noun phrases via simple heuristic when taxonomy misses
    if not critical_tokens:
        critical_tokens = _fallback_noun_extraction(query)

    return ExtractedTokens(
        critical_tokens=list(dict.fromkeys(critical_tokens)),
        domain_concepts=list(dict.fromkeys(domain_concepts)),
        patent_synonyms=list(dict.fromkeys(patent_synonyms)),
        original_query=query,
        concept_groups=concept_groups,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fallback_noun_extraction(query: str) -> list[str]:
    """
    Simple fallback when the taxonomy has no match.

    Extracts hyphenated technical compounds (e.g. *camera-based*) and words
    longer than 6 characters (likely technical, not stop words).
    """
    hyphenated = re.findall(r"\b\w+-\w+\b", query.lower())
    long_words = [w for w in query.lower().split() if len(w) > 6]
    return list(dict.fromkeys(hyphenated + long_words))
