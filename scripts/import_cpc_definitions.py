"""Streaming importer for official CPC definition XML ZIP archives."""
from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

from sqlalchemy import func
from sqlalchemy.dialects.mysql import insert as mysql_insert

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from storage.database import engine, create_tables, classification_entries_table  # noqa: E402

OFFICIAL_CPC_URL = "https://www.cooperativepatentclassification.org/"
DEFAULT_BATCH_SIZE = 500


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def first_descendant(element: ET.Element, wanted: str) -> ET.Element | None:
    for node in element.iter():
        if local_name(node.tag) == wanted:
            return node
    return None


def clean_text(element: ET.Element | None) -> str | None:
    if element is None:
        return None
    pieces = []
    for node in element.iter():
        if node.text:
            value = " ".join(node.text.split())
            if value:
                pieces.append(value)
    text = " ".join(pieces).strip()
    return text or None


def clean_title(element: ET.Element | None) -> str | None:
    """Extract a CPC title while excluding nested classification references."""
    if element is None:
        return None

    pieces = []

    def visit(node: ET.Element) -> None:
        if local_name(node.tag) == "class-ref":
            return

        if node.text:
            value = " ".join(node.text.split())
            if value:
                pieces.append(value)

        for child in list(node):
            visit(child)
            if child.tail:
                value = " ".join(child.tail.split())
                if value:
                    pieces.append(value)

    visit(element)
    text = " ".join(pieces).strip()
    return text or None


def body_text(element: ET.Element | None) -> str | None:
    if element is None:
        return None
    bodies = [n for n in element.iter() if local_name(n.tag) == "section-body"]
    if bodies:
        text = " ".join(filter(None, (clean_text(n) for n in bodies)))
        return text or None
    return clean_text(element)


def section_json(element: ET.Element | None) -> str | None:
    if element is None:
        return None
    data = []
    for child in list(element):
        title = clean_text(first_descendant(child, "section-title"))
        body = clean_text(first_descendant(child, "section-body")) or clean_text(child)
        if title or body:
            data.append({"section": title or local_name(child.tag), "text": body or ""})
    if not data:
        text = clean_text(element)
        return json.dumps([{"section": local_name(element.tag), "text": text}],
                          ensure_ascii=False) if text else None
    return json.dumps(data, ensure_ascii=False)


def infer_level_parent(code: str) -> tuple[str | None, str | None]:
    code = code.strip().replace(" ", "")
    if re.fullmatch(r"[A-HY]\d{2}[A-Z]", code):
        return "subclass", None
    match = re.fullmatch(r"([A-HY]\d{2}[A-Z])(\d+)/(\d+)", code)
    if match:
        subclass, group_num, subgroup_num = match.groups()
        if set(subgroup_num) == {"0"}:
            return "main_group", subclass
        return "subgroup", f"{subclass}{group_num}/00"
    if re.fullmatch(r"[A-HY]\d{2}", code):
        return "class", None
    if re.fullmatch(r"[A-HY]", code):
        return "section", None
    return None, None


def parse_item(item: ET.Element, source_file: str, version: str,
               pub_date: str | None, pub_type: str | None) -> dict[str, Any] | None:
    code = clean_text(first_descendant(item, "classification-symbol"))
    title = clean_title(first_descendant(item, "definition-title"))
    if not code or not title:
        return None
    code = code.replace(" ", "")
    definition = body_text(first_descendant(item, "definition-statement"))
    level, parent = infer_level_parent(code)
    return {
        "system": "CPC",
        "classification_code": code,
        "title": title,
        "level": level,
        "parent_code": parent,
        "official_definition": definition,
        "scope_notes": body_text(first_descendant(item, "scope-notes")),
        "references_json": section_json(first_descendant(item, "references")),
        "glossary_json": section_json(first_descendant(item, "glossary-of-terms")),
        "special_rules": body_text(first_descendant(item, "special-rules")),
        "definition_available": bool(definition),
        "source_file": source_file[:255],
        "source_version": version[:32],
        "source_publication_date": pub_date,
        "source_publication_type": pub_type,
        "source_url": OFFICIAL_CPC_URL,
    }


def iter_records(archive: zipfile.ZipFile, version: str):
    entries = [
        e for e in archive.infolist()
        if not e.is_dir() and e.filename.lower().endswith(".xml")
        and Path(e.filename).name.lower().startswith("cpc-definition-")
    ]
    if not entries:
        raise RuntimeError("No cpc-definition-*.xml files found in this ZIP.")
    for entry in entries:
        pub_date = pub_type = None
        with archive.open(entry, "r") as stream:
            try:
                for event, elem in ET.iterparse(stream, events=("start", "end")):
                    tag = local_name(elem.tag)
                    if event == "start" and tag == "definitions":
                        pub_date = elem.attrib.get("publication-date")
                        pub_type = elem.attrib.get("publication-type")
                    elif event == "end" and tag == "definition-item":
                        record = parse_item(elem, Path(entry.filename).name,
                                            version, pub_date, pub_type)
                        if record:
                            yield record
                        elem.clear()
            except ET.ParseError as exc:
                raise RuntimeError(f"Invalid XML in {entry.filename}: {exc}") from exc


def upsert_batch(connection, records: list[dict[str, Any]]) -> None:
    table = classification_entries_table
    stmt = mysql_insert(table).values(records)

    update_cols = [
        col.name for col in table.columns
        if col.name not in {
            "id",
            "system",
            "classification_code",
            "source_version",
            "imported_at",
        }
    ]

    updates = {
        name: getattr(stmt.inserted, name)
        for name in update_cols
    }

    # imported_at has a database default and is not supplied in records.
    # Refresh it explicitly when an existing record is re-imported.
    updates["imported_at"] = func.now()

    connection.execute(stmt.on_duplicate_key_update(**updates))


def import_zip(zip_path: Path, batch_size: int = DEFAULT_BATCH_SIZE) -> dict[str, int]:
    if not zip_path.is_file():
        raise FileNotFoundError(f"ZIP not found: {zip_path}")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    match = re.search(r"(\d{6})", zip_path.stem)
    version = match.group(1) if match else "unknown"

    create_tables()
    counts = {"processed": 0, "batches": 0, "xml_files": 0}
    batch: list[dict[str, Any]] = []

    with zipfile.ZipFile(zip_path, "r") as archive:
        counts["xml_files"] = sum(
            1 for e in archive.infolist()
            if not e.is_dir() and e.filename.lower().endswith(".xml")
            and Path(e.filename).name.lower().startswith("cpc-definition-")
        )
        with engine.begin() as connection:
            for record in iter_records(archive, version):
                batch.append(record)
                if len(batch) >= batch_size:
                    upsert_batch(connection, batch)
                    counts["processed"] += len(batch)
                    counts["batches"] += 1
                    print(f"Processed {counts['processed']:,} entries...")
                    batch.clear()
            if batch:
                upsert_batch(connection, batch)
                counts["processed"] += len(batch)
                counts["batches"] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", required=True, help="Path to CPC definition ZIP")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()
    zip_path = Path(args.zip)
    result = import_zip(zip_path, args.batch_size)
    version_match = re.search(r"(\d{6})", zip_path.stem)
    version = version_match.group(1) if version_match else "unknown"
    print("\\nCPC import complete")
    print(f"  Source version: {version}")
    print(f"  XML files: {result['xml_files']:,}")
    print(f"  Entries upserted: {result['processed']:,}")
    print(f"  Batches: {result['batches']:,}")
    print("  Note: upserted includes inserted and updated rows.")


if __name__ == "__main__":
    main()
