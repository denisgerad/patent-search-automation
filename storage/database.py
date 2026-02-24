"""
storage/database.py

SQLAlchemy Core engine and table definitions for the patent-search pipeline.

Uses SQLAlchemy Core (no ORM) for simplicity, as specified in the architecture.
The engine is created once from settings.db_url (defaults to
sqlite:///data/patents.db) and shared across all repository calls.

Exports
-------
engine          — the shared SQLAlchemy Engine instance
patents_table   — the Table metadata object (used in all queries)
create_tables() — call once at startup (or in tests) to create the schema
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

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from utils.logger import get_logger  # noqa: E402

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Engine — created once from settings.db_url
# ---------------------------------------------------------------------------

def _make_engine() -> Engine:
    """Create and return a SQLAlchemy Engine from settings.db_url."""
    url: str = settings.db_url
    # For SQLite, ensure the parent directory exists
    if url.startswith("sqlite:///"):
        db_path = Path(url.replace("sqlite:///", ""))
        db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(url, echo=False)
    log.info("Database engine created", extra={"db_url": url})
    return engine


engine: Engine = _make_engine()

# ---------------------------------------------------------------------------
# Schema — SQLAlchemy Core table definitions
# ---------------------------------------------------------------------------

metadata = MetaData()

patents_table = Table(
    "patents",
    metadata,
    Column("patent_id",       String(32),  primary_key=True),
    Column("patent_title",    Text,        nullable=False),
    Column("patent_abstract", Text,        nullable=True),
    Column("patent_type",     String(64),  nullable=True),
    Column("patent_date",     String(16),  nullable=True),  # ISO date e.g. 2023-05-09
    Column(
        "created_at",
        DateTime,
        server_default=func.now(),
        nullable=False,
    ),
)


def create_tables() -> None:
    """Create all tables in the database (idempotent — safe to call repeatedly)."""
    metadata.create_all(engine)
    log.info("Tables created / verified", extra={"tables": list(metadata.tables.keys())})
