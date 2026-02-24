"""
retrieval/bm25_ranker.py

Responsible for QUERYING an existing BM25 index.

Separation of concerns
----------------------
bm25_index.py owns:  index construction, serialisation, deserialisation.
This file owns:      running queries against an already-built index.

This module never builds or loads an index — it only receives one as an
argument. That makes every function here trivially unit-testable without
touching the filesystem or requiring a real patent corpus.
"""

import logging
import sys
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi  # pip install rank-bm25

# Ensure the project root is importable regardless of the launch directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logger = logging.getLogger(__name__)


def get_scores(index: BM25Okapi, query: str) -> np.ndarray:
    """
    Return a BM25 score for every document in *index* against *query*.

    Tokenisation mirrors the approach used in bm25_index._patent_to_tokens():
    lowercase whitespace split.  If you upgrade to a richer tokeniser in
    bm25_index.py, update the single line marked below to match.

    Args:
        index: A BM25Okapi index built by bm25_index.build_index().
        query: Free-text query string (e.g. the user's original search term
               or one of the expanded terms from query_expansion.py).

    Returns:
        numpy array of float scores, shape (n_docs,), aligned 1-to-1 with
        the patent list that was passed to build_index().
    """
    # ↓ Keep this tokenisation in sync with bm25_index._patent_to_tokens()
    tokens = query.lower().split()

    if not tokens:
        logger.warning("Empty query passed to get_scores(); returning zero scores.")
        return np.zeros(len(index.doc_freqs))

    scores = index.get_scores(tokens)
    logger.debug(
        "BM25 scores computed: query='%s', top score=%.4f", query, float(scores.max())
    )
    return np.array(scores, dtype=float)


def get_top_n(
    index: BM25Okapi,
    query: str,
    patents: list,
    n: int = 20,
) -> list[tuple]:
    """
    Return the top-*n* patents sorted by BM25 score descending.

    Args:
        index:   A BM25Okapi index built by bm25_index.build_index().
        query:   Free-text query string.
        patents: The original patent list passed to build_index() — used only
                 to pair each score with its record.
        n:       Maximum number of results to return.

    Returns:
        List of (patent, score) tuples, highest score first.
    """
    scores = get_scores(index, query)
    # argsort ascending → reverse for descending
    ranked_indices = np.argsort(scores)[::-1][:n]
    results = [(patents[i], float(scores[i])) for i in ranked_indices]
    logger.info(
        "BM25 top-%d retrieved for query='%s' (best=%.4f, worst=%.4f)",
        len(results), query,
        results[0][1] if results else 0.0,
        results[-1][1] if results else 0.0,
    )
    return results
