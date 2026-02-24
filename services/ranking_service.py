"""
services/ranking_service.py

Combines cosine-similarity scores and BM25 scores into a single hybrid
ranking for each patent, then returns a sorted list of RankedPatent objects.

Algorithm (clean / mathematically sound)
-----------------------------------------
1. Cosine similarity is already in [0, 1] for L2-normalised vectors — use raw.
   Do NOT re-normalise cosine relative to the batch; that would distort the
   absolute similarity signal (e.g. a batch where all patents score 0.45–0.50
   would be stretched to 0–1, making weak matches look strong).
2. BM25 is unbounded — normalise to [0, 1] with per-batch min-max scaling.
3. Hybrid = cosine_weight * cosine + bm25_weight * bm25_norm.
   Default weights: 0.7 cosine + 0.3 BM25 (semantic > keyword for patents).
   Weights live in app.config.settings and can be tuned via .env.
4. Sort descending, return the top-k results as RankedPatent objects.
   cosine_score and bm25_score on each RankedPatent are both in [0, 1].
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
    Compute hybrid scores: raw cosine + normalised BM25.

    Cosine is used as-is (already in [0, 1] for unit-normalised vectors).
    BM25 is min-max normalised to [0, 1] per batch before combining.

    Weights are read from ``settings.cosine_weight`` (default 0.7) and
    ``settings.bm25_weight`` (default 0.3).

    Args:
        cosine_scores: Raw cosine similarity scores, shape (n,).  Must be in [0, 1].
        bm25_scores:   Raw BM25 scores, shape (n,).  Unbounded — normalised here.

    Returns:
        Hybrid score array, shape (n,), in [0, 1].
    """
    # Cosine: use raw — do NOT normalise relative to batch.
    c = cosine_scores
    # BM25: unbounded → normalise to [0, 1].
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

    # --- Cosine similarity (raw, already in [0, 1]) ---
    cosine = cosine_similarity_matrix(query_vec, doc_vecs)

    # --- BM25 (raw, unbounded) ---
    bm25_raw = np.array(get_scores(build_index(patents), query), dtype=float)

    # --- BM25 normalised to [0, 1] for both hybrid formula and display ---
    bm25_norm = normalize(bm25_raw)

    # --- Hybrid: raw cosine + normalised BM25 ---
    hybrid = hybrid_rank(cosine, bm25_raw)   # hybrid_rank normalises BM25 internally

    # --- Sort and truncate ---
    sorted_idx = np.argsort(hybrid)[::-1][:k]

    ranked: list[RankedPatent] = []
    for i in sorted_idx:
        ranked.append(
            RankedPatent(
                patent=patents[i],
                cosine_score=float(cosine[i]),      # raw cosine in [0, 1]
                bm25_score=float(bm25_norm[i]),     # normalised BM25 in [0, 1]
                hybrid_score=float(hybrid[i]),
            )
        )

    logger.info(
        "Ranked %d patents → top-%d returned (best hybrid=%.4f)",
        len(patents), len(ranked),
        ranked[0].hybrid_score if ranked else 0.0,
    )
    return ranked
