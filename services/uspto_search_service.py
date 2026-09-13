
"""USPTO Open Data Portal patent search service.

Uses the current USPTO ODP Patent Applications Search API.

This service intentionally does not reuse the legacy PatentsView query
syntax because ODP has a different request/response contract.
"""

from __future__ import annotations

from xml.etree import ElementTree as ET

import requests

from app.config import settings
from models.schemas import PatentRecord
from utils.logger import get_logger

log = get_logger(__name__)

USPTO_SEARCH_URL = (
    "https://api.uspto.gov/api/v1/patent/applications/search"
)


def _build_headers() -> dict[str, str]:
    """Build headers for the USPTO ODP API."""

    # Keep using the existing project setting for now.
    # The value is our working USPTO ODP API key.
    api_key = getattr(settings, "patentsview_api_key", None)

    if not api_key:
        raise RuntimeError(
            "USPTO ODP API key is not configured. "
            "Set patentsview_api_key in .env."
        )

    return {
        "X-API-KEY": api_key,
        "Accept": "application/json",
    }


def _get_xml(url: str) -> str:
    """Retrieve a USPTO APPXML publication document."""

    api_key = getattr(settings, "patentsview_api_key", None)

    if not api_key:
        raise RuntimeError(
            "USPTO ODP API key is not configured."
        )

    response = requests.get(
        url,
        headers={
            "X-API-KEY": api_key,
            "Accept": "application/xml",
        },
        timeout=30,
    )

    response.raise_for_status()

    return response.text


def _extract_xml_text(
    root: ET.Element,
    tag_name: str,
) -> str:
    """Extract text from the first matching XML element."""

    for element in root.iter():
        local_name = element.tag.split("}")[-1]

        if local_name.lower() == tag_name.lower():
            text = "".join(element.itertext())
            return " ".join(text.split()).strip()

    return ""


def _get_abstract(raw: dict) -> str:
    """Retrieve and extract the abstract from the USPTO publication XML."""

    metadata = raw.get("applicationMetaData") or {}

    # Some ODP records may contain the abstract directly.
    abstract = metadata.get("abstractText")

    if isinstance(abstract, str):
        return " ".join(abstract.split()).strip()

    if isinstance(abstract, list):
        parts: list[str] = []

        for item in abstract:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = (
                    item.get("text")
                    or item.get("abstractText")
                    or ""
                )
                if text:
                    parts.append(str(text))

        result = " ".join(" ".join(parts).split()).strip()

        if result:
            return result

    # Search results normally do not contain the abstract.
    # Retrieve the associated PGPub publication XML instead.
    application_number = (
        raw.get("applicationNumberText") or ""
    ).strip()

    if not application_number:
        log.warning(
            "USPTO abstract: application number missing"
        )
        return ""

    try:
        documents_url = (
            "https://api.uspto.gov/api/v1/patent/applications/"
            f"{application_number}/associated-documents"
        )

        response = requests.get(
            documents_url,
            headers=_build_headers(),
            timeout=30,
        )
        response.raise_for_status()

        data = response.json()

        records = data.get("patentFileWrapperDataBag") or []

        if not records:
            log.warning(
                "USPTO abstract: no associated documents for %s",
                application_number,
            )
            return ""

        document_metadata = (
            records[0].get("pgpubDocumentMetaData") or {}
        )

        xml_url = document_metadata.get("fileLocationURI")

        if not xml_url:
            log.warning(
                "USPTO abstract: PGPub XML URL missing for %s",
                application_number,
            )
            return ""

        xml_text = _get_xml(xml_url)

        root = ET.fromstring(xml_text)

        abstract_text = _extract_xml_text(
            root,
            "abstract",
        )

        log.info(
            "USPTO abstract extracted: %d characters",
            len(abstract_text),
        )

        return abstract_text

    except Exception as exc:
        log.warning(
            "USPTO abstract retrieval failed for %s: %s",
            application_number,
            exc,
        )
        return ""


def _get_continuity_family_id(raw: dict) -> str | None:
    """Determine the root application number for a USPTO continuity family."""

    application_number = (
        str(raw.get("applicationNumberText") or "").strip()
    )

    if not application_number:
        return None

    parent_bag = raw.get("parentContinuityBag") or []

    if not isinstance(parent_bag, list) or not parent_bag:
        return application_number

    parent_candidates: list[tuple[str, str]] = []

    for relationship in parent_bag:
        if not isinstance(relationship, dict):
            continue

        parent_number = str(
            relationship.get("parentApplicationNumberText") or ""
        ).strip()

        parent_filing_date = str(
            relationship.get("parentApplicationFilingDate") or ""
        ).strip()

        if parent_number:
            parent_candidates.append(
                (parent_filing_date, parent_number)
            )

    if not parent_candidates:
        return application_number

    # The earliest parent in the continuity chain is the family root.
    parent_candidates.sort(
        key=lambda item: item[0] or "9999-99-99"
    )

    return parent_candidates[0][1]


def _build_patent_record(raw: dict) -> PatentRecord | None:
    """Convert one USPTO ODP record into PatentRecord."""

    metadata = raw.get("applicationMetaData") or {}

    application_number = (
        raw.get("applicationNumberText")
        or ""
    )

    continuity_family_id = _get_continuity_family_id(raw)

    title = (
        metadata.get("inventionTitle")
        or ""
    )

    publication_number = (
        metadata.get("earliestPublicationNumber")
        or ""
    )

    patent_id = (
        publication_number
        or application_number
    )

    patent_type = (
        metadata.get("applicationTypeLabelName")
        or ""
    )

    patent_date = (
        metadata.get("filingDate")
        or metadata.get("earliestPublicationDate")
        or None
    )

    if not patent_id:
        return None

    if not title:
        return None

    try:
        abstract = _get_abstract(raw)

        log.info(
            "USPTO abstract extracted: %d characters",
            len(abstract),
        )

        return PatentRecord(
            patent_id=str(patent_id),
            patent_title=str(title),
            patent_abstract=abstract or None,
            patent_type=patent_type or None,
            patent_date=str(patent_date) if patent_date else None,
            continuity_family_id=continuity_family_id,
        )

    except Exception as exc:
        log.warning(
            "Could not build PatentRecord for %s: %s",
            patent_id,
            exc,
        )
        return None


def _build_odp_query(query: str) -> str:
    """Convert an application search phrase into an ODP title query."""

    query = " ".join(query.split()).strip()

    if not query:
        return query

    return f'applicationMetaData.inventionTitle:"{query}"'


def search_patents(
    query: str,
    limit: int = 25,
    offset: int = 0,
) -> list[PatentRecord]:
    """Search USPTO ODP and return PatentRecord objects.

    The ODP search endpoint uses Lucene-style query syntax.
    """

    headers = _build_headers()
    odp_query = _build_odp_query(query)

    params = {
        "q": odp_query,
        "offset": offset,
        "limit": min(limit, 100),
    }

    log.info(
        "USPTO ODP search: %s -> %s",
        query,
        odp_query,
    )

    try:
        response = requests.get(
            USPTO_SEARCH_URL,
            headers=headers,
            params=params,
            timeout=30,
        )

    except requests.exceptions.RequestException as exc:
        log.error(
            "USPTO ODP network error: %s",
            exc,
        )
        return []

    if response.status_code != 200:
        log.error(
            "USPTO ODP search failed: %s — %s",
            response.status_code,
            response.text[:500],
        )
        return []

    try:
        data = response.json()

    except ValueError:
        log.error(
            "USPTO ODP returned invalid JSON"
        )
        return []

    log.info(
        "USPTO ODP response keys: %s",
        list(data.keys())
        if isinstance(data, dict)
        else type(data).__name__,
    )

    raw_records = data.get(
        "patentFileWrapperDataBag",
        [],
    )

    log.info("USPTO raw records: %d", len(raw_records))

    records: list[PatentRecord] = []

    for raw in raw_records:

        log.info("USPTO processing one raw record")

        if not isinstance(raw, dict):
            continue

        log.info("USPTO building PatentRecord")
        patent = _build_patent_record(raw)

        if patent is not None:
            records.append(patent)

    log.info(
        "USPTO ODP search complete: %d records",
        len(records),
    )

    return records