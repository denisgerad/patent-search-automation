"""
services/dedup_service.py

Deduplicates PatentRecord objects by patent_id before they reach the
embedding stage.

Why this matters:
  Query expansion generates multiple search terms, each of which hits
  PatentsView independently.  The same patent can appear in several result
  sets.  Embedding duplicates wastes compute and distorts ranking scores.
  Deduplicating here — immediately after search — ensures every downstream
  stage works on a unique set.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is importable regardless of the launch directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.schemas import PatentRecord  # noqa: E402
from utils.logger import get_logger  # noqa: E402

log = get_logger(__name__)


def deduplicate(patents: list[PatentRecord]) -> list[PatentRecord]:
    """Remove duplicate PatentRecord objects, keeping the first occurrence.

    Deduplication key is ``patent_id``, which is the stable unique identifier
    from PatentsView.  Because PatentRecord requires ``patent_id`` at
    construction time (it is a non-optional ``str`` field), every record
    passed here is guaranteed to have a valid id.

    Parameters
    ----------
    patents:
        Output of ``fetch_all_patents()`` — one or more pages of results,
        potentially containing the same patent from different query terms.

    Returns
    -------
    list[PatentRecord]
        Deduplicated list in order of first occurrence.
    """
    seen: dict[str, PatentRecord] = {}
    for p in patents:
        if p.patent_id not in seen:
            seen[p.patent_id] = p
        else:
            log.debug("Duplicate removed", extra={"patent_id": p.patent_id})

    unique = list(seen.values())
    duplicate_count = len(patents) - len(unique)
    log.info(
        "Deduplication complete",
        extra={
            "input": len(patents),
            "unique": len(unique),
            "duplicates_removed": duplicate_count,
        },
    )
    return unique
