"""
services/dedup_service.py

Deduplicates patents by patent_id before they reach the embedding stage.

Why this matters:
  Query expansion generates multiple search terms, each of which hits PatentsView
  independently. The same patent can appear in several result pages. Embedding
  duplicates wastes GPU/CPU compute and distorts ranking scores. Deduplicating
  here — immediately after search — ensures every downstream stage works on a
  unique set.

Supports:
  - Plain dicts returned directly by search_service.py
  - Pydantic PatentRecord objects (accessed via attribute, not dict key)
  - Any mixed list of the two (e.g. during a gradual schema migration)
"""

import logging
import sys
from pathlib import Path
from typing import Union

# Ensure the project root is importable regardless of the launch directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logger = logging.getLogger(__name__)


def _get_patent_id(patent: Union[dict, object]) -> str | None:
    """
    Extract patent_id from either a dict or a PatentRecord-like object.
    Returns None if the field is missing or blank, so the record is kept
    rather than silently dropped.
    """
    if isinstance(patent, dict):
        return patent.get("patent_id") or None
    return getattr(patent, "patent_id", None) or None


def deduplicate(patents: list) -> list:
    """
    Remove duplicate patents, keeping the first occurrence of each patent_id.

    Deduplication is performed on patent_id, which is the stable, unique
    identifier shared across PatentsView result pages and any future USPTO
    data sources.

    Records with a missing or empty patent_id are kept as-is (they are not
    dropped) so that upstream data quality issues remain visible rather than
    being silently discarded.

    Args:
        patents: List of patent dicts or PatentRecord objects, as returned by
                 search_service.fetch_patents_by_keywords().

    Returns:
        A new list with duplicates removed, preserving original order.
    """
    seen: set[str] = set()
    unique: list = []
    duplicate_count = 0

    for patent in patents:
        pid = _get_patent_id(patent)

        if pid is None:
            # No ID to deduplicate on — keep and warn so the issue is visible.
            logger.warning(
                "Patent record with missing patent_id encountered; keeping it."
            )
            unique.append(patent)
            continue

        if pid in seen:
            duplicate_count += 1
            logger.debug("Duplicate removed: patent_id=%s", pid)
        else:
            seen.add(pid)
            unique.append(patent)

    logger.info(
        "Deduplication complete: %d input → %d unique (%d duplicate%s removed)",
        len(patents),
        len(unique),
        duplicate_count,
        "s" if duplicate_count != 1 else "",
    )

    return unique
