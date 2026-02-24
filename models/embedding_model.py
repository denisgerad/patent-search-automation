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

    def __init__(self, model_name: str | None = None) -> None:
        name = model_name or settings.embedding_model
        logger.info("Loading embedding model", extra={"model": name})
        self.model = SentenceTransformer(name)
        logger.info("Embedding model loaded", extra={"model": name})

    def embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """
        Embed a list of strings and return a normalised numpy array.

        Args:
            texts:      Strings to embed (typically patent title + abstract).
            batch_size: Internal batch size passed to sentence-transformers.

        Returns:
            Float32 ndarray of shape (len(texts), embedding_dim).
            Each row is L2-normalised (norm ≈ 1.0), ready for dot-product
            cosine similarity.
        """
        if not texts:
            return np.empty((0,), dtype=np.float32)

        vecs: np.ndarray = self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=False,
        )
        logger.info("Embedded texts", extra={"count": len(texts), "dim": vecs.shape[1]})
        return vecs
