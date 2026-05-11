"""
retrieval/vector_store.py

A lightweight in-memory vector store backed by numpy for patent embeddings.

Design decisions:
  • Keep it simple: store (patent_ids, embeddings) as aligned arrays.  No
    external index needed for the corpus sizes expected (≤ 20 000 patents).
  • faiss-cpu is listed in requirements for future scale-out; swap the
    similarity backend in similarity.py when you hit that threshold.
  • Persistence uses numpy .npy / .npz files for zero-dependency serialisation.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_STORE_DIR = Path("data") / "vectors"


class VectorStore:
    """
    Stores and retrieves L2-normalised patent embeddings.

    Usage:
        store = VectorStore()
        store.add(patent_ids, embeddings)       # add records
        vecs = store.get_embeddings()            # (n, dim) array
        ids  = store.get_ids()                   # list[str]
        store.save()                             # persist to disk
        store.load()                             # reload from disk
    """

    def __init__(self, store_dir: Path | str = _DEFAULT_STORE_DIR) -> None:
        self._dir = Path(store_dir)
        self._ids: list[str] = []
        self._matrix: Optional[np.ndarray] = None     # shape (n, dim)

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------

    def add(self, patent_ids: list[str], embeddings: np.ndarray) -> None:
        """
        Append *embeddings* aligned to *patent_ids* to the store.

        Duplicate patent_ids are silently ignored so that re-running the
        pipeline keeps the store idempotent.

        Args:
            patent_ids: List of patent ID strings, length == len(embeddings).
            embeddings: Float32 ndarray of shape (n, dim), L2-normalised.
        """
        if len(patent_ids) != len(embeddings):
            raise ValueError(
                f"patent_ids length {len(patent_ids)} != "
                f"embeddings rows {len(embeddings)}"
            )

        existing = set(self._ids)
        new_mask = [i for i, pid in enumerate(patent_ids) if pid not in existing]

        if not new_mask:
            logger.debug("VectorStore.add: all %d patents already present.", len(patent_ids))
            return

        new_ids = [patent_ids[i] for i in new_mask]
        new_vecs = embeddings[new_mask]

        self._ids.extend(new_ids)
        if self._matrix is None:
            self._matrix = new_vecs.astype(np.float32)
        else:
            self._matrix = np.vstack([self._matrix, new_vecs.astype(np.float32)])

        logger.info(
            "VectorStore: added %d vectors (total %d).",
            len(new_ids), len(self._ids),
        )

    def clear(self) -> None:
        """Remove all stored embeddings."""
        self._ids = []
        self._matrix = None

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_ids(self) -> list[str]:
        """Return the list of stored patent IDs (insertion order)."""
        return list(self._ids)

    def get_embeddings(self) -> np.ndarray:
        """
        Return the embedding matrix of shape (n, dim).
        Returns an empty (0,) array when the store is empty.
        """
        if self._matrix is None:
            return np.empty((0,), dtype=np.float32)
        return self._matrix

    def __len__(self) -> int:
        return len(self._ids)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, name: str = "patents") -> None:
        """
        Save embeddings to *store_dir*/<name>.npz (ids + matrix).

        Args:
            name: Base filename (no extension) — default "patents".
        """
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._dir / f"{name}.npz"
        matrix = self._matrix if self._matrix is not None else np.empty((0, 0), dtype=np.float32)
        np.savez_compressed(path, ids=np.array(self._ids), matrix=matrix)
        logger.info("VectorStore saved: %s (%d vectors)", path, len(self._ids))

    def load(self, name: str = "patents") -> bool:
        """
        Load embeddings from *store_dir*/<name>.npz.

        Returns:
            True if the file was found and loaded, False otherwise.
        """
        path = self._dir / f"{name}.npz"
        if not path.exists():
            logger.debug("VectorStore.load: file not found at %s", path)
            return False
        data = np.load(path, allow_pickle=False)
        self._ids = list(data["ids"])
        self._matrix = data["matrix"] if data["matrix"].ndim == 2 else None
        logger.info("VectorStore loaded: %s (%d vectors)", path, len(self._ids))
        return True

    # ------------------------------------------------------------------
    # Cache-aware helpers (Fix B from fix_embeddings1.txt)
    # The vector store is an EMBEDDING CACHE only — never the search corpus.
    # The patent corpus always comes from search_service live results.
    # ------------------------------------------------------------------

    def store_embeddings(
        self,
        patents: list,
        embeddings: np.ndarray,
    ) -> None:
        """Cache *embeddings* keyed by patent_id. Alias for add()."""
        ids = [p.patent_id for p in patents]
        self.add(ids, embeddings)

    def get_cached_embeddings(
        self,
        patents: list,
    ) -> tuple[list, list, np.ndarray]:
        """
        Split *patents* into those with cached embeddings and those without.

        Returns:
            (cached_patents, uncached_patents, cached_vecs_matrix)
            cached_vecs_matrix has shape (len(cached_patents), dim) or (0,).
        """
        id_to_idx = {pid: i for i, pid in enumerate(self._ids)}

        cached_patents, uncached_patents, cached_vecs = [], [], []
        for p in patents:
            idx = id_to_idx.get(p.patent_id)
            if idx is not None and self._matrix is not None:
                cached_patents.append(p)
                cached_vecs.append(self._matrix[idx])
            else:
                uncached_patents.append(p)

        if cached_vecs:
            matrix = np.array(cached_vecs, dtype=np.float32)
        else:
            matrix = np.empty((0,), dtype=np.float32)

        return cached_patents, uncached_patents, matrix
