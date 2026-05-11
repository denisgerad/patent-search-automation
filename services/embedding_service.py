"""
services/embedding_service.py

Produces and manages embeddings for a list of PatentRecord objects.

Responsibilities:
  1. Concatenate patent_title + patent_abstract into a single string per patent.
  2. Batch-embed via EmbeddingModel (sentence-transformers).
  3. Optionally persist to / load from a VectorStore.

This service is the single place where patent text is turned into vectors.
No similarity logic lives here — see retrieval/similarity.py.
"""
import logging
from pathlib import Path

import numpy as np

from models.embedding_model import EmbeddingModel
from models.schemas import PatentRecord
from retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


def _patent_to_text(patent: PatentRecord) -> str:
    """
    Concatenate title + abstract for embedding.
    Title is repeated twice to give it higher weight in the embedding;
    abstract contains the technical substance so both are required.
    """
    title = (patent.patent_title or "").strip()
    abstract = (patent.patent_abstract or "").strip()

    if not title and not abstract:
        logger.warning(
            "Patent %s has no text content — embedding will be noise",
            patent.patent_id,
        )
        return f"patent {patent.patent_id}"  # minimal placeholder

    if not abstract:
        logger.debug("Patent %s has no abstract — using doubled title", patent.patent_id)
        return f"{title}. {title}"  # double title as weight fallback

    # Repeat title to boost its weight in the resulting vector
    return f"{title}. {title}. {abstract}"


def embed_patents(
    patents: list[PatentRecord],
    model: EmbeddingModel,
    store: VectorStore | None = None,
) -> np.ndarray:
    """
    Embed *patents* and return a (n, dim) normalised float32 array.

    If *store* is supplied, embeddings are also added to it (idempotent).

    Args:
        patents: List of PatentRecord objects to embed.
        model:   An initialised EmbeddingModel instance.
        store:   Optional VectorStore to accumulate vectors across calls.

    Returns:
        Float32 ndarray of shape (len(patents), embedding_dim), L2-normalised.
    """
    if not patents:
        logger.warning("embed_patents called with empty list; returning empty array.")
        return np.empty((0,), dtype=np.float32)

    texts = [_patent_to_text(p) for p in patents]
    logger.info("Embedding %d patent(s).", len(texts))

    vecs = model.embed_documents(texts)

    if store is not None:
        ids = [p.patent_id for p in patents]
        store.add(ids, vecs)

    return vecs


def embed_query(
    query: str,
    model: EmbeddingModel,
    expanded: list[str] | None = None,
) -> np.ndarray:
    """
    Embed a query string and return a 1-D normalised float32 array.

    When *expanded* is supplied the original query is enriched with the
    expanded terms before embedding.  The concatenated text is more
    keyword-dense than the raw natural-language query alone, which
    improves cosine similarity against patent abstracts (which are
    written in technical, keyword-heavy prose).

    The last entry of *expanded* is placed first after the original query
    because — with the structural prompt — it is the KEYWORD-ONLY form
    (3-5 critical technical keywords, no filler), making it the
    most valuable signal for embedding alignment.  All other expansions
    follow to add breadth.

    Args:
        query:    Free-text query string (the original user input).
        model:    An initialised EmbeddingModel instance.
        expanded: Optional list returned by
                  :func:`~services.query_expansion.expand_query`
                  (includes the original at index 0).  Only the
                  Mistral-generated alternatives (index 1 onward) are
                  appended; the original is never duplicated.

    Returns:
        1-D float32 array of shape (embedding_dim,), L2-normalised.
    """
    if expanded and len(expanded) > 1:
        alternatives = expanded[1:]  # skip index-0 (original query)
        # Put the keyword-only form (last) immediately after the original
        # for maximum embedding density, then append the rest.
        ordered = [alternatives[-1]] + alternatives[:-1]
        enriched = query + " " + " ".join(ordered)
        logger.debug(
            "embed_query: enriched text built from original + %d expansions",
            len(alternatives),
        )
    else:
        enriched = query

    return model.embed_query(enriched)
