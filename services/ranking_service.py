"""
services/ranking_service.py

Combines cosine-similarity scores and BM25 scores into a single hybrid
ranking for each patent, then returns a sorted list of RankedPatent objects.

Algorithm
---------
1. Normalise both score arrays to [0, 1] with min-max normalisation.
   (A small epsilon prevents division-by-zero when all scores are equal.)
2. Compute hybrid = cosine_weight * cosine_norm + bm25_weight * bm25_norm.
   Weights are pulled from app.config.settings.
3. Sort descending and return the top-k results as RankedPatent objects.
"""
import logging

import numpy as np

from app.config import settings
from models.schemas import PatentRecord, RankedPatent

logger = logging.getLogger(__name__)

_EPSILON = 1e-9   # prevents division-by-zero in normalisation


def normalize(scores: np.ndarray) -> np.ndarray:
    """
    Min-max normalise *scores* to [0, 1].

    Args:
        scores: 1-D float array of arbitrary range.

    Returns:
        1-D float array with values in [0, 1].
        If all scores are identical the array is mapped to all-zeros.
    """
    mn, mx = float(scores.min()), float(scores.max())
    return (scores - mn) / (mx - mn + _EPSILON)


def hybrid_rank(
    cosine_scores: np.ndarray,
    bm25_scores: np.ndarray,
) -> np.ndarray:
    """
    Produce hybrid scores from two normalised score arrays.

    Weights are read from ``settings.cosine_weight`` and
    ``settings.bm25_weight`` (must sum to 1.0, but this is not enforced).

    Args:
        cosine_scores: Raw cosine similarity scores, shape (n,).
        bm25_scores:   Raw BM25 scores, shape (n,).

    Returns:
        Hybrid score array, shape (n,), in [0, 1].
    """
    c = normalize(cosine_scores)
    b = normalize(bm25_scores)
    return settings.cosine_weight * c + settings.bm25_weight * b


def rank(
    query: str,
    patents: list[PatentRecord],
    doc_vecs: np.ndarray,
    query_vec: np.ndarray,
    top_k: int | None = None,
) -> list[RankedPatent]:
    """
    Rank *patents* using hybrid cosine + BM25 scoring.

    Args:
        query:     Original (or expanded) query string — used for BM25.
        patents:   List of PatentRecord objects aligned to *doc_vecs*.
        doc_vecs:  L2-normalised embedding matrix, shape (n, dim).
        query_vec: L2-normalised query embedding, shape (dim,).
        top_k:     Return the top-k results only.  Defaults to
                   ``settings.top_k_results`` when None.

    Returns:
        List of RankedPatent objects sorted by hybrid_score descending.
    """
    from retrieval.similarity import cosine_similarity_matrix
    from retrieval.bm25_index import build_index
    from retrieval.bm25_ranker import get_scores

    k = top_k if top_k is not None else settings.top_k_results

    if not patents or doc_vecs.ndim < 2:
        logger.warning("rank() called with no patents; returning empty list.")
        return []

    # --- Cosine similarity ---
    cosine = cosine_similarity_matrix(query_vec, doc_vecs)

    # --- BM25 ---
    bm25_index = build_index(patents)
    bm25 = get_scores(bm25_index, query)

    # --- Hybrid ---
    hybrid = hybrid_rank(cosine, np.array(bm25, dtype=float))

    # --- Sort and truncate ---
    sorted_idx = np.argsort(hybrid)[::-1][:k]

    ranked: list[RankedPatent] = []
    for i in sorted_idx:
        ranked.append(
            RankedPatent(
                patent=patents[i],
                cosine_score=float(cosine[i]),
                bm25_score=float(bm25[i]),
                hybrid_score=float(hybrid[i]),
            )
        )

    logger.info(
        "Ranked %d patents → top-%d returned (best hybrid=%.4f)",
        len(patents), len(ranked),
        ranked[0].hybrid_score if ranked else 0.0,
    )
    return ranked
