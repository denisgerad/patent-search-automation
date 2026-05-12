"""
retrieval/similarity.py

Vectorised cosine similarity between a query embedding and a matrix of
document embeddings, plus dynamic threshold computation.

IMPORTANT: Both inputs must already be L2-normalised before calling these
functions. EmbeddingModel.embed_documents() and embed_query() always return
normalised vectors, so no extra normalisation step is required here.

With unit vectors, cosine similarity reduces to a dot product, which numpy
computes as a single BLAS call — no looping, no per-pair computation.
"""
import logging

import numpy as np

logger = logging.getLogger(__name__)


def cosine_similarity_matrix(
    query_vec: np.ndarray,
    doc_vecs: np.ndarray,
) -> np.ndarray:
    """
    Compute cosine similarity between *query_vec* and every row of *doc_vecs*.

    Both arguments must be L2-normalised (norm == 1.0 for every row/vector).
    Under that assumption, cosine similarity equals the dot product.

    Args:
        query_vec: 1-D array of shape (dim,) — the embedded query.
        doc_vecs:  2-D array of shape (n_docs, dim) — patent embeddings.

    Returns:
        1-D float array of shape (n_docs,) with one similarity score per patent.
        Scores are in [-1, 1]; higher is more similar.
    """
    # Ensure query is 1-D
    q = query_vec.ravel()
    # (n_docs, dim) @ (dim,) → (n_docs,)
    return doc_vecs @ q


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """
    Return the indices of the top-*k* highest scores (descending).

    Args:
        scores: 1-D array of similarity scores.
        k:      Number of top results to return.

    Returns:
        Integer array of length min(k, len(scores)).
    """
    k = min(k, len(scores))
    # argpartition is O(n) — faster than full sort for large arrays
    top = np.argpartition(scores, -k)[-k:]
    # Sort the small top-k slice for deterministic ordering
    return top[np.argsort(scores[top])[::-1]]


def compute_dynamic_threshold(
    scores: np.ndarray,
    strategy: str = "mean_plus_std",
) -> float:
    """
    Compute a data-driven threshold rather than using a fixed value.

    Strategies:
    - mean_plus_std : mean + 0.5*std  (keeps top ~30% typically)
    - top_k_pct     : score at 80th percentile
    - elbow         : finds the sharpest drop in sorted scores

    Args:
        scores:   1-D array of similarity / hybrid scores.
        strategy: One of "mean_plus_std", "top_k_pct", "elbow".

    Returns:
        Float threshold value.  Returns 0.0 for empty input.
    """
    if len(scores) == 0:
        return 0.0

    if strategy == "mean_plus_std":
        threshold = float(np.mean(scores) + 0.5 * np.std(scores))

    elif strategy == "top_k_pct":
        threshold = float(np.percentile(scores, 80))

    elif strategy == "elbow":
        sorted_scores = np.sort(scores)[::-1]
        if len(sorted_scores) < 2:
            return float(sorted_scores[0])
        diffs = np.diff(sorted_scores)
        elbow_idx = int(np.argmin(diffs))  # sharpest drop
        threshold = float(sorted_scores[elbow_idx])

    else:
        threshold = 0.45  # fallback

    logger.info(
        "Dynamic threshold computed: strategy=%s threshold=%.4f mean=%.4f std=%.4f max=%.4f",
        strategy,
        round(threshold, 4),
        round(float(np.mean(scores)), 4),
        round(float(np.std(scores)), 4),
        round(float(np.max(scores)), 4),
    )
    return threshold


def filter_by_threshold(
    scores: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    Return indices of entries in *scores* that are >= *threshold*, sorted descending.

    Args:
        scores:    1-D score array.
        threshold: Minimum score to keep.

    Returns:
        Integer index array sorted by score descending.
    """
    above = np.where(scores >= threshold)[0]
    return above[np.argsort(scores[above])[::-1]]


def compute_token_coverage(patent, tokens) -> dict:
    """
    Compute how many critical token groups are represented in the patent.

    Tokens from the same domain concept count as one group — "infrared" and
    "IR sensor" both satisfy the sensor concept, not two wins.

    Args:
        patent: PatentRecord — patent to evaluate.
        tokens: ExtractedTokens — must have a populated concept_groups dict.

    Returns:
        dict with keys: coverage (float 0–1), matched_concepts (int),
        total_concepts (int), concept_hits (dict[str, bool]).
    """
    text = (
        f"{patent.patent_title or ''} {patent.patent_abstract or ''}"
    ).lower()

    concept_hits: dict[str, bool] = {}
    for concept, data in tokens.concept_groups.items():
        terms = [t.lower() for t in data.get("terms", [])]
        synonyms = [s.lower() for s in data.get("patent_synonyms", [])]
        concept_hits[concept] = any(t in text for t in terms + synonyms)

    total_concepts = len(concept_hits)
    matched_concepts = sum(concept_hits.values())
    coverage = matched_concepts / total_concepts if total_concepts else 0.0

    return {
        "coverage": coverage,
        "matched_concepts": matched_concepts,
        "total_concepts": total_concepts,
        "concept_hits": concept_hits,
    }


def apply_coverage_penalty(
    hybrid_score: float,
    coverage: float,
    penalty_curve: str = "quadratic",
) -> float:
    """
    Apply a smooth penalty based on concept coverage.

    Coverage 1.0 (all concepts matched) → no penalty
    Coverage 0.67 (2/3 matched)         → moderate penalty
    Coverage 0.33 (1/3 matched)         → heavy penalty
    Coverage 0.0  (nothing matched)     → maximum penalty

    penalty_curve options:
      linear    : multiplier = coverage
      quadratic : multiplier = coverage ** 2  (recommended — accelerating penalty)
      step      : hard cutoffs at fixed thresholds
    """
    if penalty_curve == "linear":
        multiplier = coverage
    elif penalty_curve == "quadratic":
        multiplier = coverage ** 2
    elif penalty_curve == "step":
        if coverage == 1.0:
            multiplier = 1.0
        elif coverage >= 0.67:
            multiplier = 0.6
        elif coverage >= 0.33:
            multiplier = 0.25
        else:
            multiplier = 0.05
    else:
        multiplier = coverage  # fallback to linear

    return hybrid_score * multiplier

