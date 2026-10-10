
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

from sqlalchemy import func
from sqlalchemy.dialects.mysql import insert as mysql_insert

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from storage.database import (
    engine,
    metadata,
    classification_scheme_entries_table as scheme_table,
)  # noqa: E402

ARCHIVE = Path("data/classification/CPCSchemeXML202608.zip")
SOURCE_VERSION = "202608"
BATCH_SIZE = 1000


def local_name(tag):
    return tag.rsplit("}", 1)[-1]


def extract_title(item):
    for child in item:
        if local_name(child.tag) != "class-title":
            continue

        parts = []
        for node in child:
            if local_name(node.tag) == "title-part":
                text = " ".join("".join(node.itertext()).split())
                if text:
                    parts.append(text)

        if parts:
            return " ".join(parts)

        return " ".join("".join(child.itertext()).split())

    return ""


def iter_classification_items(root, member_name, publication_date,
                              publication_type):
    """Yield classification entries with their nearest CPC parent."""

    def walk(element, nearest_parent_code):
        current_parent = nearest_parent_code

        if local_name(element.tag) == "classification-item":
            code = ""

            for child in element:
                if local_name(child.tag) == "classification-symbol":
                    code = "".join(child.itertext()).strip().upper()
                    break

            code = re.sub(r"\s+", "", code)
            title = extract_title(element)

            if code and title:
                row = {
                    "system": "CPC",
                    "classification_code": code,
                    "title": title,
                    "level": int(element.attrib.get("level", 0)),
                    "parent_code": nearest_parent_code,
                    "definition_available": int(
                        element.attrib.get(
                            "definition-exists", ""
                        ).lower() == "true"
                    ),
                    "source_file": member_name,
                    "source_version": SOURCE_VERSION,
                    "source_publication_date": publication_date or None,
                    "source_publication_type": publication_type or None,
                    "date_revised": element.attrib.get("date-revised"),
                    "status": element.attrib.get("status"),
                }

                yield row
                current_parent = code

        for child in element:
            yield from walk(child, current_parent)

    yield from walk(root, None)


def import_scheme():
    if not ARCHIVE.exists():
        raise FileNotFoundError(f"Archive not found: {ARCHIVE}")

    metadata.create_all(engine, tables=[scheme_table])

    batch = []
    imported = 0
    seen = set()
    target_code = "G06F40/30"
    target_row = None

    with ZipFile(ARCHIVE) as archive, engine.begin() as connection:
        xml_files = [
            name for name in archive.namelist()
            if name.lower().endswith(".xml")
            and not name.startswith("__MACOSX/")
        ]

        print(f"Archive: {ARCHIVE}")
        print(f"XML files: {len(xml_files)}")
        print(f"Source version: {SOURCE_VERSION}")

        for member in xml_files:
            with archive.open(member) as stream:
                root = ET.parse(stream).getroot()

            publication_date = root.attrib.get("publication-date", "")
            publication_type = root.attrib.get("publication-type", "")

            for row in iter_classification_items(
                root, member, publication_date, publication_type
            ):
                key = (row["system"], row["classification_code"])
                if key in seen:
                    continue
                seen.add(key)

                if row["classification_code"] == target_code:
                    target_row = row

                batch.append(row)

                if len(batch) >= BATCH_SIZE:
                    upsert_batch(connection, batch)
                    imported += len(batch)
                    batch.clear()

        if batch:
            upsert_batch(connection, batch)
            imported += len(batch)

    print(f"\nRows processed: {imported}")
    print(f"Unique codes in archive: {len(seen)}")

    if target_row:
        print("\nVerified target:")
        for key, value in target_row.items():
            print(f"{key}: {value}")
    else:
        print(f"\nWARNING: {target_code} was not found.")


def upsert_batch(connection, rows):
    statement = mysql_insert(scheme_table).values(rows)

    update_columns = {
        column.name: getattr(statement.inserted, column.name)
        for column in scheme_table.columns
        if column.name not in {
            "id",
            "system",
            "classification_code",
            "source_version",
            "imported_at",
        }
    }
    update_columns["imported_at"] = func.now()

    statement = statement.on_duplicate_key_update(**update_columns)
    connection.execute(statement)


if __name__ == "__main__":
    import_scheme()
