"""
retrieval/bm25_index.py

Responsible for BUILDING a BM25 index from a list of patents and optionally
persisting it to disk so it can be reloaded without rebuilding.

Separation of concerns
----------------------
This file owns:  index construction, serialisation, deserialisation.
bm25_ranker.py owns: querying an already-built index.

Keeping build and query separate means:
  - The index can be built once and reused across many queries.
  - bm25_ranker.py can be tested without touching the filesystem.
  - Rebuilding (e.g. after ingesting new patents) is a single, isolated step.
"""

import logging
import pickle
import sys
from pathlib import Path
from typing import Union

# Ensure the project root is importable regardless of the launch directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rank_bm25 import BM25Okapi  # pip install rank-bm25

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text preparation
# ---------------------------------------------------------------------------

def _patent_to_tokens(patent: Union[dict, object]) -> list[str]:
    """
    Concatenate title + abstract for a patent and return a token list.

    Supports both plain dicts (from search_service) and PatentRecord objects.
    Uses simple whitespace tokenisation — upgrade to a patent-specific
    tokeniser here when needed; nothing else needs to change.
    """
    if isinstance(patent, dict):
        title    = patent.get("patent_title") or ""
        abstract = patent.get("patent_abstract") or ""
        pid      = patent.get("patent_id", "")
    else:
        title    = getattr(patent, "patent_title", "") or ""
        abstract = getattr(patent, "patent_abstract", "") or ""
        pid      = getattr(patent, "patent_id", "") or ""

    text = f"{title} {abstract}".lower().strip()
    if not text:
        # Prevent empty token list crash in BM25Okapi
        text = f"patent {pid}"
    return text.split()  # ← swap this line for a richer tokeniser if needed


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_index(patents: list) -> BM25Okapi:
    """
    Build a BM25Okapi index from a list of patent records.

    Args:
        patents: List of patent dicts or PatentRecord objects.  Must be
                 non-empty (BM25Okapi raises on an empty corpus).

    Returns:
        A fitted BM25Okapi index aligned 1-to-1 with *patents*.
        Pass this object to bm25_ranker.get_scores() to run queries.
    """
    if not patents:
        raise ValueError("Cannot build a BM25 index from an empty patent list.")

    corpus = [_patent_to_tokens(p) for p in patents]

    # Warn about any patents that produced empty token lists (e.g. no title/abstract).
    empty_count = sum(1 for tokens in corpus if not tokens)
    if empty_count:
        logger.warning(
            "%d patent(s) produced empty token lists and will not contribute "
            "to BM25 scoring.", empty_count
        )

    index = BM25Okapi(corpus)
    logger.info("BM25 index built: %d document(s) in corpus.", len(corpus))
    return index


# ---------------------------------------------------------------------------
# Persist / Load
# ---------------------------------------------------------------------------

def save_index(index: BM25Okapi, path: Union[str, Path]) -> None:
    """
    Serialise *index* to *path* using pickle.

    The saved file can be reloaded with load_index() to skip rebuilding on
    subsequent runs — especially useful when the patent corpus is large.

    Args:
        index: A BM25Okapi index produced by build_index().
        path:  Destination file path (e.g. "data/processed/bm25.pkl").
               Parent directories are created automatically.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(index, fh, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("BM25 index saved: %s", path)


def load_index(path: Union[str, Path]) -> BM25Okapi:
    """
    Deserialise a BM25Okapi index previously saved with save_index().

    Args:
        path: Path to the pickle file.

    Returns:
        The restored BM25Okapi index, ready for querying.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"BM25 index file not found: {path}\n"
            "Run build_index() and save_index() first."
        )
    with path.open("rb") as fh:
        index = pickle.load(fh)
    logger.info("BM25 index loaded: %s", path)
    return index
