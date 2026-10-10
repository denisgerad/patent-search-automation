

"""Read-only access to the CPC scheme and definition catalogues."""

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

from storage.database import (
    engine,
    classification_entries_table,
    classification_scheme_entries_table,
)

DEFAULT_SYSTEM = "CPC"


def normalize_code(code: str) -> str:
    """Normalize a CPC code, e.g. 'A22B 3/00' -> 'A22B3/00'."""
    return re.sub(r"\s+", "", str(code or "").strip()).upper()


def _latest_version_query(table, system: str = DEFAULT_SYSTEM):
    """Return the latest source version for a catalogue table."""
    return (
        select(func.max(table.c.source_version))
        .where(table.c.system == system.upper())
        .scalar_subquery()
    )


def _latest_conditions(table, system: str = DEFAULT_SYSTEM):
    """Restrict queries to the latest version of the selected catalogue."""
    return and_(
        table.c.system == system.upper(),
        table.c.source_version == _latest_version_query(table, system),
    )


def _row_to_dict(row: Any) -> dict[str, Any]:
    """Convert a definition-catalogue row into a normal dictionary."""
    result = dict(row._mapping)

    for field in ("references_json", "glossary_json"):
        raw_value = result.get(field)
        output_field = field.removesuffix("_json")

        if not raw_value:
            result[output_field] = []
            continue

        try:
            result[output_field] = json.loads(raw_value)
        except (TypeError, json.JSONDecodeError):
            result[output_field] = raw_value

    return result


def _scheme_row_to_dict(row: Any) -> dict[str, Any]:
    """Convert a scheme row into a dictionary."""
    return dict(row._mapping)


def _get_scheme_entry(normalized: str) -> dict[str, Any] | None:
    """Retrieve an exact entry from the latest CPC scheme version."""
    table = classification_scheme_entries_table

    statement = (
        select(table)
        .where(
            _latest_conditions(table),
            func.upper(table.c.classification_code) == normalized,
        )
        .limit(1)
    )

    with engine.connect() as connection:
        row = connection.execute(statement).first()

    return _scheme_row_to_dict(row) if row else None


def _get_definition_entry(normalized: str) -> dict[str, Any] | None:
    """Retrieve an exact entry from the latest definition version."""
    table = classification_entries_table

    statement = (
        select(table)
        .where(
            _latest_conditions(table),
            func.upper(
                func.replace(table.c.classification_code, " ", "")
            ) == normalized,
        )
        .limit(1)
    )

    with engine.connect() as connection:
        row = connection.execute(statement).first()

    return _row_to_dict(row) if row else None


def _merge_entries(
    scheme_entry: dict[str, Any] | None,
    definition_entry: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Use scheme fields as the classification source and attach definitions."""
    if scheme_entry is None and definition_entry is None:
        return None

    # Preserve the definition-only fallback for codes not present in the scheme.
    if scheme_entry is None:
        return definition_entry

    definition_entry = definition_entry or {}

    merged = dict(definition_entry)
    merged.update({
        "system": scheme_entry["system"],
        "classification_code": scheme_entry["classification_code"],
        "title": scheme_entry["title"],
        "level": scheme_entry["level"],
        "parent_code": scheme_entry["parent_code"],
        "definition_available": bool(
            scheme_entry["definition_available"]
        ),
        "scheme_source_file": scheme_entry["source_file"],
        "scheme_source_version": scheme_entry["source_version"],
        "scheme_publication_date": scheme_entry["source_publication_date"],
        "scheme_publication_type": scheme_entry["source_publication_type"],
        "date_revised": scheme_entry["date_revised"],
        "status": scheme_entry["status"],
    })

    # Keep definition provenance separate from scheme provenance.
    merged.setdefault("official_definition", None)
    merged.setdefault("scope_notes", None)
    merged.setdefault("references", [])
    merged.setdefault("glossary", [])
    merged.setdefault("special_rules", None)
    merged.setdefault("source_url", None)
    merged.setdefault("imported_at", None)

    return merged


def get_cpc_catalogue_entry(
    code: str,
) -> dict[str, Any] | None:
    """Retrieve a CPC classification, even if it has no separate definition."""
    normalized = normalize_code(code)
    if not normalized:
        return None

    scheme_entry = _get_scheme_entry(normalized)
    definition_entry = _get_definition_entry(normalized)

    return _merge_entries(scheme_entry, definition_entry)


def search_cpc_catalogue(
    query: str,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Search CPC scheme codes and titles, enriched with definitions."""
    search_text = str(query or "").strip()
    if not search_text:
        return []

    limit = max(1, min(int(limit), 100))
    normalized = normalize_code(search_text)
    table = classification_scheme_entries_table

    statement = (
        select(table.c.classification_code)
        .where(
            _latest_conditions(table),
            or_(
                func.upper(table.c.classification_code).like(
                    f"%{normalized}%"
                ),
                func.lower(table.c.title).like(
                    f"%{search_text.lower()}%"
                ),
            ),
        )
        .order_by(table.c.classification_code.asc())
        .limit(limit)
    )

    with engine.connect() as connection:
        codes = connection.execute(statement).scalars().all()

    results = []
    for code in codes:
        entry = get_cpc_catalogue_entry(code)
        if entry is not None:
            results.append(entry)

    return results


def get_cpc_catalogue_children(
    parent_code: str,
) -> list[dict[str, Any]]:
    """Return scheme entries whose parent matches the supplied CPC code."""
    normalized = normalize_code(parent_code)
    if not normalized:
        return []

    table = classification_scheme_entries_table

    statement = (
        select(table.c.classification_code)
        .where(
            _latest_conditions(table),
            func.upper(
                func.replace(table.c.parent_code, " ", "")
            ) == normalized,
        )
        .order_by(table.c.classification_code.asc())
    )

    with engine.connect() as connection:
        codes = connection.execute(statement).scalars().all()

    results = []
    for code in codes:
        entry = get_cpc_catalogue_entry(code)
        if entry is not None:
            results.append(entry)

    return results


def get_cpc_catalogue_metadata(
    code: str,
) -> dict[str, Any] | None:
    """Return classification metadata and any available definition."""
    entry = get_cpc_catalogue_entry(code)
    if entry is None:
        return None

    return {
        "system": entry["system"],
        "classification_code": entry["classification_code"],
        "title": entry["title"],
        "level": entry["level"],
        "parent_code": entry["parent_code"],
        "official_definition": entry.get("official_definition"),
        "scope_notes": entry.get("scope_notes"),
        "references": entry.get("references", []),
        "glossary": entry.get("glossary", []),
        "special_rules": entry.get("special_rules"),
        "definition_available": entry["definition_available"],
        "source_file": entry.get("source_file"),
        "source_version": entry.get("source_version"),
        "source_publication_date": entry.get("source_publication_date"),
        "source_publication_type": entry.get("source_publication_type"),
        "source_url": entry.get("source_url"),
        "imported_at": entry.get("imported_at"),
        "scheme_source_file": entry.get("scheme_source_file"),
        "scheme_source_version": entry.get("scheme_source_version"),
        "scheme_publication_date": entry.get("scheme_publication_date"),
        "scheme_publication_type": entry.get("scheme_publication_type"),
        "date_revised": entry.get("date_revised"),
        "status": entry.get("status"),
    }
