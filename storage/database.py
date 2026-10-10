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
    Boolean,
    Column,
    DateTime,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    create_engine,
    func,
)
from sqlalchemy.dialects.mysql import LONGTEXT
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

classification_entries_table = Table(
    "classification_entries",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("system", String(8), nullable=False),
    Column("classification_code", String(32), nullable=False),
    Column("title", Text, nullable=False),
    Column("level", String(24), nullable=True),
    Column("parent_code", String(32), nullable=True),
    Column("official_definition", LONGTEXT, nullable=True),
    Column("scope_notes", LONGTEXT, nullable=True),
    Column("references_json", LONGTEXT, nullable=True),
    Column("glossary_json", LONGTEXT, nullable=True),
    Column("special_rules", LONGTEXT, nullable=True),
    Column("definition_available", Boolean, nullable=False, default=False),
    Column("source_file", String(255), nullable=False),
    Column("source_version", String(32), nullable=False),
    Column("source_publication_date", String(16), nullable=True),
    Column("source_publication_type", String(32), nullable=True),
    Column("source_url", Text, nullable=True),
    Column(
        "imported_at",
        DateTime,
        nullable=False,
        server_default=func.now(),
    ),
    UniqueConstraint(
        "system",
        "classification_code",
        "source_version",
        name="uq_classification_system_code_version",
    ),
    Index("ix_classification_system_code", "system", "classification_code"),
    mysql_engine="InnoDB",
    mysql_charset="utf8mb4",
)

classification_scheme_entries_table = Table(
    "classification_scheme_entries",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("system", String(8), nullable=False),
    Column("classification_code", String(32), nullable=False),
    Column("title", Text, nullable=False),
    Column("level", Integer, nullable=False),
    Column("parent_code", String(32), nullable=True),
    Column("definition_available", Boolean, nullable=False, default=False),
    Column("source_file", String(255), nullable=False),
    Column("source_version", String(32), nullable=False),
    Column("source_publication_date", String(16), nullable=True),
    Column("source_publication_type", String(32), nullable=True),
    Column("date_revised", String(16), nullable=True),
    Column("status", String(40), nullable=True),
    Column(
        "imported_at",
        DateTime,
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    ),
    UniqueConstraint(
        "system",
        "classification_code",
        "source_version",
        name="uq_scheme_system_code_version",
    ),
    Index(
        "ix_scheme_system_code",
        "system",
        "classification_code",
    ),
    mysql_engine="InnoDB",
    mysql_charset="utf8mb4",
)


def create_tables() -> None:
    """Create the patents table if it does not already exist (idempotent)."""
    metadata.create_all(engine)
    log.info("Tables created / verified", extra={"tables": list(metadata.tables.keys())})
