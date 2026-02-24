"""
storage/patent_repository.py

Persistence layer for PatentRecord objects.

Two complementary stores:
  1. MySQL patent_db.patents  — structured queries, deduplication across runs
  2. JSON cache in data/raw/  — raw API responses keyed by query hash,
                                enabling full re-runs without hitting the API

Column mapping (PatentRecord field → MySQL column):
  patent_id       → patent_number
  patent_title    → title
  patent_abstract → abstract
  patent_type     → metadata  (text field; stores type string)
  patent_date     → publication_date

Public API
----------
save_patents(patents, query)     — upsert into MySQL + write JSON cache
load_from_cache(query)           — return cached PatentRecords (or None)
get_all_patents()                — return every row from the DB
get_patent_by_id(patent_id)      — return a single row (or None)
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Optional

from sqlalchemy import select
from sqlalchemy.dialects.mysql import insert as mysql_insert

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.schemas import PatentRecord  # noqa: E402
from storage.database import create_tables, engine, patents_table  # noqa: E402
from utils.logger import get_logger  # noqa: E402

log = get_logger(__name__)

# Absolute path to the raw JSON cache directory
_RAW_CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
_RAW_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Ensure tables exist when this module is first imported
create_tables()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _query_hash(query: str) -> str:
    """Return a short, filesystem-safe SHA-256 prefix for *query*."""
    return hashlib.sha256(query.strip().lower().encode()).hexdigest()[:16]


def _cache_path(query: str) -> Path:
    return _RAW_CACHE_DIR / f"{_query_hash(query)}.json"


def _record_to_row(p: PatentRecord) -> dict:
    """Map PatentRecord fields to the existing MySQL patents table columns."""
    return {
        "patent_number":    p.patent_id,
        "title":            p.patent_title,
        "abstract":         p.patent_abstract,
        "metadata":         p.patent_type,       # re-use existing text column
        "publication_date": p.patent_date,
    }


def _row_to_record(row) -> PatentRecord:
    """Map MySQL patents table row back to a PatentRecord."""
    return PatentRecord(
        patent_id=row.patent_number,
        patent_title=row.title or "",
        patent_abstract=row.abstract,
        patent_type=row.metadata,
        patent_date=str(row.publication_date) if row.publication_date else None,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_patents(patents: list[PatentRecord], query: str) -> None:
    """Upsert *patents* into MySQL and write a JSON cache file for *query*.

    Uses MySQL's INSERT ... ON DUPLICATE KEY UPDATE so repeat runs never
    create duplicates — patent_id is the primary key.

    The JSON cache file at data/raw/<hash>.json stores the raw list of
    patent dicts keyed by the normalised query string, enabling full re-runs
    without hitting the PatentsView API.
    """
    if not patents:
        log.info("save_patents called with empty list; skipping", extra={"query": query})
        return

    # ── MySQL upsert — ON DUPLICATE KEY UPDATE on patent_number ────────────
    rows = [_record_to_row(p) for p in patents]
    with engine.begin() as conn:
        stmt = mysql_insert(patents_table).values(rows)
        stmt = stmt.on_duplicate_key_update(
            title=stmt.inserted.title,
            abstract=stmt.inserted.abstract,
            metadata=stmt.inserted.metadata,
            publication_date=stmt.inserted.publication_date,
        )
        conn.execute(stmt)
    log.info(
        "Upserted patents into MySQL",
        extra={"count": len(patents), "query": query},
    )

    # ── JSON cache ───────────────────────────────────────────────────────────
    cache_file = _cache_path(query)
    # Store PatentRecord field names in JSON cache (not MySQL column names)
    # so load_from_cache can reconstruct via PatentRecord(**d) cleanly.
    cache_data = {
        "query": query,
        "count": len(patents),
        "patents": [p.model_dump() for p in patents],
    }
    cache_file.write_text(json.dumps(cache_data, indent=2), encoding="utf-8")
    log.info("JSON cache written", extra={"path": str(cache_file), "query": query})


def load_from_cache(query: str) -> Optional[list[PatentRecord]]:
    """Return cached PatentRecords for *query*, or ``None`` if no cache exists.

    Call this before ``fetch_all_patents`` to skip the live API when the
    same query was already run in a previous session.
    """
    cache_file = _cache_path(query)
    if not cache_file.exists():
        log.debug("Cache miss", extra={"query": query})
        return None

    try:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        patents = [PatentRecord(**d) for d in data.get("patents", [])]
        log.info(
            "Cache hit",
            extra={"query": query, "count": len(patents), "path": str(cache_file)},
        )
        return patents
    except Exception as exc:
        log.warning(
            "Cache file unreadable; ignoring",
            extra={"path": str(cache_file), "error": str(exc)},
        )
        return None


def get_all_patents() -> list[PatentRecord]:
    """Return every patent currently stored in the database."""
    with engine.connect() as conn:
        rows = conn.execute(select(patents_table)).fetchall()
    patents = [_row_to_record(r) for r in rows]
    log.info("get_all_patents", extra={"count": len(patents)})
    return patents


def get_patent_by_id(patent_id: str) -> Optional[PatentRecord]:
    """Return a single PatentRecord by primary key, or ``None`` if not found."""
    with engine.connect() as conn:
        row = conn.execute(
            select(patents_table).where(patents_table.c.patent_number == patent_id)
        ).fetchone()
    if row is None:
        log.debug("Patent not found in DB", extra={"patent_id": patent_id})
        return None
    return _row_to_record(row)
