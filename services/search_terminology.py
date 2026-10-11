"""Explicit terminology variants for patent-search concepts."""

import re


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


TERMINOLOGY_VARIANTS = {
    "dynamic fraud scoring": [
        "transaction risk rating",
        "risk based transaction scoring",
        "fraud risk score",
        "fraud probability",
        "probability of fraud",
        "fraud risk state",
        "fraudulent transaction probability",
        "fraud probability assessment",
    ],
    "context aware fraud detection": [
        "transaction context analysis",
        "context dependent fraud detection",
        "contextual indicators",
        "contextual transaction analysis",
        "evolving risk states",
    ],
    "adaptive detection threshold": [
        "adaptive decision threshold",
        "risk based threshold",
        "transaction risk threshold",
        "dynamic fraud threshold",
        "variable decision threshold",
        "threshold adjusted based on user behavior",
    ],
    "behavioral pattern analysis": [
        "customer behavioral profile",
        "transaction behavior patterns",
        "observed transaction behavior",
        "user actions over time",
        "account usage over time",
        "behavioral baselines",
        "historical behavioral patterns",
    ],
    "transaction information analysis": [
        "transaction attributes",
        "payment transaction data",
        "transaction events",
        "transaction event data",
        "transaction features",
        "current transaction information",
    ],
    "device characteristics fingerprinting": [
        "device attributes",
        "device fingerprint",
        "device profiling",
        "device id",
        "device identifier",
        "device ip address",
        "device ip location",
        "ip address",
        "device characteristics",
    ],
    "historical user behavior modeling": [
        "historical transaction behavior",
        "user activity history",
        "historical user actions",
        "prior transaction history",
        "historical transaction data",
        "user behavior over time",
        "historical behavior baseline",
        "historical behavioral representations",
        "previously created vector",
        "stored behavior data",
    ],
    "multi factor fraud signal fusion": [
        "combined fraud indicators",
        "multi factor risk scoring",
        "multiple transaction features",
        "combined transaction and device data",
        "correlated transaction attributes",
        "feature vectors",
        "n-dimensional vector",
    ],
    "user specific baseline profiling": [
        "individual behavioral baseline",
        "personalized risk profile",
        "user specific behavior baseline",
        "established behavioral baseline",
        "individual user transaction history",
    ],
    "anomaly detection in financial transactions": [
        "transaction anomalies",
        "financial transaction outliers",
        "fraud probability",
        "anomalous transaction behavior",
    ],
}


def add_terminology_variants(name: str, terms: list[str]) -> list[str]:
    """Preserve original terms and append recognized concept variants."""
    result = []
    seen = set()

    variants = TERMINOLOGY_VARIANTS.get(_normalize_name(name), [])

    for term in [*terms, *variants]:
        cleaned = str(term).strip()
        key = cleaned.casefold()

        if cleaned and key not in seen:
            result.append(cleaned)
            seen.add(key)

    return result
