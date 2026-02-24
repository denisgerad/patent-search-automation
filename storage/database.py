"""
storage/database.py

SQLAlchemy Core engine and table reflection for the patent-search pipeline.

The patents table already exists in patent_db with the schema defined in
schema.sql.  We reflect it at startup rather than recreating it, which
means the existing data and any existing columns are preserved.

A UNIQUE constraint on patent_number is added automatically if not present
(needed for the ON DUPLICATE KEY UPDATE upsert in patent_repository.py).

Exports
-------
engine          — the shared SQLAlchemy Engine instance
patents_table   — reflected Table object (all columns available)
create_tables() — no-op for already-existing tables; kept for API consistency
"""

from __future__ import annotations

import sys
from pathlib import Path

from sqlalchemy import MetaData, Table, create_engine, inspect, text
from sqlalchemy.engine import Engine

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from utils.logger import get_logger  # noqa: E402

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

def _make_engine() -> Engine:
    """Create and return a SQLAlchemy Engine from settings.db_url."""
    url: str = settings.db_url
    engine = create_engine(url, echo=False, pool_pre_ping=True)
    safe_url = url.split("@")[-1] if "@" in url else url
    log.info("Database engine created", extra={"db": safe_url})
    return engine


engine: Engine = _make_engine()

# ---------------------------------------------------------------------------
# Reflect the existing patents table
# ---------------------------------------------------------------------------

metadata = MetaData()
patents_table = Table("patents", metadata, autoload_with=engine)


def _ensure_unique_constraint() -> None:
    """Add UNIQUE KEY on patent_number if it does not already exist.

    This is required for ON DUPLICATE KEY UPDATE to treat patent_number as
    the business-key deduplication column.
    """
    insp = inspect(engine)
    unique_constraints = insp.get_unique_constraints("patents")
    unique_cols = {col for uc in unique_constraints for col in uc["column_names"]}
    if "patent_number" not in unique_cols:
        with engine.begin() as conn:
            conn.execute(
                text("ALTER TABLE patents ADD UNIQUE KEY uq_patent_number (patent_number)")
            )
        log.info("Added UNIQUE constraint on patents.patent_number")
    else:
        log.debug("UNIQUE constraint on patent_number already exists")


_ensure_unique_constraint()


def create_tables() -> None:
    """No-op stub kept for API consistency (table already exists in MySQL)."""
    log.debug("create_tables() called — patents table already exists, skipping")
