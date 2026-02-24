"""
storage/patent_repository.py

Persistence layer for PatentRecord objects.

Two complementary stores:
  1. MySQL patent_db.patents  — structured queries, deduplication across runs.
                                Column names match PatentRecord fields directly.
  2. JSON cache in data/raw/  — raw API responses keyed by query hash,
                                enabling full re-runs without hitting the API.

Public API
----------
save_patents(patents, query)  — upsert into MySQL + write JSON cache
load_from_cache(query)        — return cached PatentRecords (or None)
get_all_patents()             — return every row from the DB as PatentRecords
get_patent_by_id(patent_id)  — return a single PatentRecord (or None)
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Optional

from sqlalchemy import select
from sqlalchemy.dialects.mysql import insert as mysql_insert

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.schemas import PatentRecord  # noqa: E402
from storage.database import create_tables, engine, patents_table  # noqa: E402
from utils.logger import get_logger  # noqa: E402

log = get_logger(__name__)

_RAW_CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
_RAW_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Ensure table exists when this module is first imported
create_tables()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _query_hash(query: str) -> str:
    return hashlib.sha256(query.strip().lower().encode()).hexdigest()[:16]


def _cache_path(query: str) -> Path:
    return _RAW_CACHE_DIR / f"{_query_hash(query)}.json"


def _row_to_record(row) -> PatentRecord:
    """Convert a DB row directly to PatentRecord — column names match exactly."""
    return PatentRecord(
        patent_id=row.patent_id,
        patent_title=row.patent_title,
        patent_abstract=row.patent_abstract,
        patent_type=row.patent_type,
        patent_date=row.patent_date,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_patents(patents: list[PatentRecord], query: str) -> None:
    """Upsert *patents* into MySQL and write a JSON cache file for *query*.

    Uses ON DUPLICATE KEY UPDATE on patent_id (primary key) so repeat runs
    never create duplicates.  The JSON cache allows offline re-runs.
    """
    if not patents:
        log.info("save_patents: empty list, skipping", extra={"query": query})
        return

    # ── MySQL upsert ─────────────────────────────────────────────────────────
    rows = [p.model_dump(exclude={"created_at"}, exclude_none=False) for p in patents]
    # model_dump includes all PatentRecord fields; column names match exactly.
    rows = [{k: v for k, v in r.items() if k != "created_at"} for r in rows]

    with engine.begin() as conn:
        stmt = mysql_insert(patents_table).values(rows)
        stmt = stmt.on_duplicate_key_update(
            patent_title=stmt.inserted.patent_title,
            patent_abstract=stmt.inserted.patent_abstract,
            patent_type=stmt.inserted.patent_type,
            patent_date=stmt.inserted.patent_date,
        )
        conn.execute(stmt)
    log.info("Upserted patents into MySQL", extra={"count": len(patents), "query": query})

    # ── JSON cache ────────────────────────────────────────────────────────────
    cache_data = {
        "query": query,
        "count": len(patents),
        "patents": [p.model_dump() for p in patents],
    }
    cache_file = _cache_path(query)
    cache_file.write_text(json.dumps(cache_data, indent=2, default=str), encoding="utf-8")
    log.info("JSON cache written", extra={"path": str(cache_file), "query": query})


def load_from_cache(query: str) -> Optional[list[PatentRecord]]:
    """Return cached PatentRecords for *query*, or None on a cache miss."""
    cache_file = _cache_path(query)
    if not cache_file.exists():
        log.debug("Cache miss", extra={"query": query})
        return None
    try:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        patents = [PatentRecord(**d) for d in data.get("patents", [])]
        log.info("Cache hit", extra={"query": query, "count": len(patents)})
        return patents
    except Exception as exc:
        log.warning("Cache unreadable; ignoring", extra={"path": str(cache_file), "error": str(exc)})
        return None


def get_all_patents() -> list[PatentRecord]:
    """Return every patent stored in the database."""
    with engine.connect() as conn:
        rows = conn.execute(select(patents_table)).fetchall()
    patents = [_row_to_record(r) for r in rows]
    log.info("get_all_patents", extra={"count": len(patents)})
    return patents


def get_patent_by_id(patent_id: str) -> Optional[PatentRecord]:
    """Return a single PatentRecord by patent_id, or None if not found."""
    with engine.connect() as conn:
        row = conn.execute(
            select(patents_table).where(patents_table.c.patent_id == patent_id)
        ).fetchone()
    if row is None:
        log.debug("Patent not found", extra={"patent_id": patent_id})
        return None
    return _row_to_record(row)
