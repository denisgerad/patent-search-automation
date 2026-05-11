"""
models/embedding_model.py

Thin wrapper around sentence-transformers for local BGE-large embeddings.
Responsibility: load the model once, embed a list of texts, return normalized
numpy arrays. No retrieval or ranking logic lives here.
"""
from sentence_transformers import SentenceTransformer
import numpy as np

from app.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingModel:
    """Produces L2-normalised embeddings via a local SentenceTransformer model."""

    # BGE-large requires this prefix on QUERIES only, not documents.
    # Without it, query and document embeddings are computed in different
    # vector sub-spaces and cosine scores become unreliable.
    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str | None = None) -> None:
        name = model_name or settings.embedding_model
        logger.info("Loading embedding model", extra={"model": name})
        self.model = SentenceTransformer(name)
        logger.info("Embedding model loaded", extra={"model": name})

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a search query with the BGE asymmetric retrieval prefix.

        Returns:
            1-D float32 array of shape (embedding_dim,), L2-normalised.
        """
        prefixed = self.QUERY_PREFIX + query
        vecs: np.ndarray = self.model.encode(
            [prefixed],
            normalize_embeddings=True,
            batch_size=1,
        )
        logger.info("Embedded query (with BGE prefix), dim=%d", vecs.shape[1])
        return vecs[0]

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """
        Embed document texts (patents) WITHOUT the query prefix.

        Returns:
            Float32 ndarray of shape (len(texts), embedding_dim), L2-normalised.
        """
        if not texts:
            return np.empty((0,), dtype=np.float32)
        vecs: np.ndarray = self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
        )
        logger.info("Embedded documents", extra={"count": len(texts), "dim": vecs.shape[1]})
        return vecs

    def embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Backward-compatible alias for embed_documents."""
        return self.embed_documents(texts, batch_size=batch_size)
