"""
retrieval/similarity.py

Vectorised cosine similarity between a query embedding and a matrix of
document embeddings.

IMPORTANT: Both inputs must already be L2-normalised before calling these
functions. EmbeddingModel.embed() always returns normalised vectors, so no
extra normalisation step is required here.

With unit vectors, cosine similarity reduces to a dot product, which numpy
computes as a single BLAS call — no looping, no per-pair computation.
"""
import numpy as np


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
