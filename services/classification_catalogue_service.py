
"""Read-only access to the imported CPC classification catalogue."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

from sqlalchemy import and_, func, or_, select

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from storage.database import engine, classification_entries_table


DEFAULT_SYSTEM = "CPC"


def normalize_code(code: str) -> str:
    """Normalize a CPC code, e.g. 'A22B 3/00' -> 'A22B3/00'."""
    return re.sub(r"\s+", "", str(code or "").strip()).upper()


def _latest_version_query(system: str = DEFAULT_SYSTEM):
    """Return a scalar query for the latest imported source version."""
    table = classification_entries_table
    return (
        select(func.max(table.c.source_version))
        .where(table.c.system == system.upper())
        .scalar_subquery()
    )


def _base_conditions(system: str = DEFAULT_SYSTEM):
    """Restrict catalogue queries to the latest imported version."""
    table = classification_entries_table
    return and_(
        table.c.system == system.upper(),
        table.c.source_version == _latest_version_query(system),
    )


def _row_to_dict(row: Any) -> dict[str, Any]:
    """Convert a SQLAlchemy row into a normal dictionary."""
    result = dict(row._mapping)

    for field in ("references_json", "glossary_json"):
        raw_value = result.get(field)
        if not raw_value:
            result[field.removesuffix("_json")] = []
            continue

        try:
            result[field.removesuffix("_json")] = json.loads(raw_value)
        except (TypeError, json.JSONDecodeError):
            result[field.removesuffix("_json")] = raw_value

    return result


def get_cpc_catalogue_entry(code: str) -> dict[str, Any] | None:
    """Retrieve one CPC entry by its exact classification code."""
    normalized = normalize_code(code)
    if not normalized:
        return None

    table = classification_entries_table
    statement = (
        select(table)
        .where(
            _base_conditions(),
            func.upper(func.replace(table.c.classification_code, " ", ""))
            == normalized,
        )
        .limit(1)
    )

    with engine.connect() as connection:
        row = connection.execute(statement).first()

    return _row_to_dict(row) if row else None


def search_cpc_catalogue(
    query: str,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Search CPC entries by classification code or title."""
    search_text = str(query or "").strip()
    if not search_text:
        return []

    limit = max(1, min(int(limit), 100))
    table = classification_entries_table
    normalized = normalize_code(search_text)

    conditions = [
        _base_conditions(),
        or_(
            func.upper(table.c.classification_code).like(
                f"%{normalized}%"
            ),
            func.lower(table.c.title).like(
                f"%{search_text.lower()}%"
            ),
        ),
    ]

    statement = (
        select(table)
        .where(*conditions)
        .order_by(
            table.c.classification_code.asc()
        )
        .limit(limit)
    )

    with engine.connect() as connection:
        rows = connection.execute(statement).fetchall()

    return [_row_to_dict(row) for row in rows]


def get_cpc_catalogue_children(
    parent_code: str,
) -> list[dict[str, Any]]:
    """Return entries whose parent_code matches the supplied CPC code."""
    normalized = normalize_code(parent_code)
    if not normalized:
        return []

    table = classification_entries_table
    statement = (
        select(table)
        .where(
            _base_conditions(),
            func.upper(func.replace(table.c.parent_code, " ", ""))
            == normalized,
        )
        .order_by(table.c.classification_code.asc())
    )

    with engine.connect() as connection:
        rows = connection.execute(statement).fetchall()

    return [_row_to_dict(row) for row in rows]


def get_cpc_catalogue_metadata(
    code: str,
) -> dict[str, Any] | None:
    """Return catalogue metadata and official definition for one CPC code."""
    entry = get_cpc_catalogue_entry(code)
    if entry is None:
        return None

    return {
        "system": entry["system"],
        "classification_code": entry["classification_code"],
        "title": entry["title"],
        "level": entry["level"],
        "parent_code": entry["parent_code"],
        "official_definition": entry["official_definition"],
        "scope_notes": entry["scope_notes"],
        "references": entry["references"],
        "glossary": entry["glossary"],
        "special_rules": entry["special_rules"],
        "definition_available": entry["definition_available"],
        "source_file": entry["source_file"],
        "source_version": entry["source_version"],
        "source_publication_date": entry["source_publication_date"],
        "source_publication_type": entry["source_publication_type"],
        "source_url": entry["source_url"],
        "imported_at": entry["imported_at"],
    }
