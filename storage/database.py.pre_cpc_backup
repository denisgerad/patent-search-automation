"""
storage/database.py

SQLAlchemy Core engine and table definition for the patent-search pipeline.

Uses the MySQL credentials from settings (loaded from .env) and creates the
patents table with column names that match PatentRecord fields directly —
no column mapping needed anywhere in the codebase.

Exports
-------
engine          — the shared SQLAlchemy Engine instance
patents_table   — the Table metadata object (used in all queries)
create_tables() — creates the patents table if it does not exist
"""

from __future__ import annotations

import sys
from pathlib import Path

from sqlalchemy import (
    Column,
    DateTime,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    func,
)
from sqlalchemy.engine import Engine

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from utils.logger import get_logger  # noqa: E402

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

def _make_engine() -> Engine:
    url: str = settings.db_url
    engine = create_engine(url, echo=False, pool_pre_ping=True)
    safe_url = url.split("@")[-1] if "@" in url else url
    log.info("Database engine created", extra={"db": safe_url})
    return engine


engine: Engine = _make_engine()

# ---------------------------------------------------------------------------
# Schema — column names match PatentRecord fields exactly
# ---------------------------------------------------------------------------

metadata = MetaData()

patents_table = Table(
    "patents",
    metadata,
    Column("patent_id",       String(64),   primary_key=True),
    Column("patent_title",    Text,         nullable=False),
    Column("patent_abstract", Text,         nullable=True),
    Column("patent_type",     String(64),   nullable=True),
    Column("patent_date",     String(16),   nullable=True),  # e.g. '2023-05-09'
    Column(
        "created_at",
        DateTime,
        server_default=func.now(),
        nullable=False,
    ),
    mysql_engine="InnoDB",
    mysql_charset="utf8mb4",
)


def create_tables() -> None:
    """Create the patents table if it does not already exist (idempotent)."""
    metadata.create_all(engine)
    log.info("Tables created / verified", extra={"tables": list(metadata.tables.keys())})
