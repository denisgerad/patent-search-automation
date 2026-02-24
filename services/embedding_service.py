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
    """Combine title and abstract into a single embeddable string."""
    title = patent.patent_title or ""
    abstract = patent.patent_abstract or ""
    return f"{title} {abstract}".strip()


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

    vecs = model.embed(texts)

    if store is not None:
        ids = [p.patent_id for p in patents]
        store.add(ids, vecs)

    return vecs


def embed_query(query: str, model: EmbeddingModel) -> np.ndarray:
    """
    Embed a single query string and return a 1-D normalised float32 array.

    Args:
        query: Free-text query string.
        model: An initialised EmbeddingModel instance.

    Returns:
        1-D float32 array of shape (embedding_dim,).
    """
    vecs = model.embed([query])
    return vecs[0]
