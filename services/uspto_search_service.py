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


def _extract_xml_claims(root: ET.Element) -> str | None:
    """Extract numbered claims from a USPTO publication XML document."""
    claim_texts: list[str] = []

    for element in root.iter():
        local_name = element.tag.split("}")[-1].lower()

        if local_name != "claim":
            continue

        text = " ".join("".join(element.itertext()).split()).strip()
        if text:
            claim_texts.append(text)

    return "\n".join(claim_texts) or None


def _get_publication_text(raw: dict) -> tuple[str, str | None]:
    """Retrieve abstract and claims from the associated publication XML."""
    application_number = str(
        raw.get("applicationNumberText") or ""
    ).strip()

    if not application_number:
        return "", None

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

        for record in records:
            document_metadata = (
                record.get("pgpubDocumentMetaData") or {}
            )
            xml_url = document_metadata.get("fileLocationURI")

            if not xml_url:
                continue

            xml_text = _get_xml(xml_url)
            root = ET.fromstring(xml_text)

            abstract_text = _extract_xml_text(root, "abstract")
            claims_text = _extract_xml_claims(root)

            if abstract_text or claims_text:
                return abstract_text, claims_text

        return "", None

    except Exception as exc:
        log.warning(
            "USPTO publication text retrieval failed for %s: %s",
            application_number,
            exc,
        )
        return "", None


def _get_metadata_abstract(raw: dict) -> str:
    """Return the abstract supplied directly by the search response, if any."""

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


def _normalize_cpc_classifications(value: object) -> list[str]:
    """Normalize CPC classifications from the varying ODP response shapes.

    ODP may return the classification bag as strings or as objects containing
    a classification symbol. Converting a dict with ``str(item)`` produces a
    Python dictionary representation, not a usable CPC code; this helper
    extracts only plausible code-bearing fields and removes duplicates.
    """
    if value is None:
        return []

    # Field names seen in classification payloads; support common variants
    # without accidentally treating descriptive metadata as a classification.
    code_keys = (
        "classificationSymbol",
        "cpcClassificationSymbol",
        "classificationCode",
        "cpcClassificationCode",
        "symbol",
        "code",
    )
    nested_keys = (
        "classification",
        "cpcClassification",
        "classificationData",
    )

    if isinstance(value, str):
        code = " ".join(value.split()).strip()
        return [code] if code else []

    if isinstance(value, dict):
        for key in code_keys:
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                code = " ".join(candidate.split()).strip()
                return [code]
        result: list[str] = []
        for key in nested_keys:
            if key in value:
                result.extend(_normalize_cpc_classifications(value[key]))
        return list(dict.fromkeys(result))

    if isinstance(value, (list, tuple, set)):
        result: list[str] = []
        for item in value:
            result.extend(_normalize_cpc_classifications(item))
        return list(dict.fromkeys(result))

    return []


def _extract_applicant_assignee(metadata: dict) -> str | None:
    """Extract applicant names from USPTO application metadata."""
    names: list[str] = []

    applicant_bag = metadata.get("applicantBag") or []

    if isinstance(applicant_bag, dict):
        applicant_bag = [applicant_bag]

    if isinstance(applicant_bag, list):
        for applicant in applicant_bag:
            if not isinstance(applicant, dict):
                continue

            name = str(
                applicant.get("applicantNameText") or ""
            ).strip()

            if name and name not in names:
                names.append(name)

    if not names:
        first_applicant = str(
            metadata.get("firstApplicantName") or ""
        ).strip()

        if first_applicant:
            names.append(first_applicant)

    return "; ".join(names) if names else None


def _build_patent_record(raw: dict) -> PatentRecord | None:
    """Convert one USPTO ODP record into PatentRecord."""

    metadata = raw.get("applicationMetaData") or {}

    uspc_class = (
        str(metadata.get("class") or "").strip()
        or None
    )

    uspc_subclass = (
        str(metadata.get("subclass") or "").strip()
        or None
    )

    cpc_classifications = _normalize_cpc_classifications(
        metadata.get("cpcClassificationBag")
    )

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

    applicant_assignee = _extract_applicant_assignee(metadata)

    if not patent_id:
        return None

    if not title:
        return None

    try:
        # Preserve the abstract already supplied by the search response.
        abstract = _get_metadata_abstract(raw)
        claims_text = None

        # One publication-XML retrieval supplies the claims and, when the
        # search response had no abstract, the abstract as well.
        if raw.get("applicationNumberText"):
            xml_abstract, claims_text = _get_publication_text(raw)

            if not abstract:
                abstract = xml_abstract

        log.info(
            "USPTO publication text: abstract=%d chars, claims=%d chars",
            len(abstract),
            len(claims_text or ""),
        )

        return PatentRecord(
            patent_id=str(patent_id),
            patent_title=str(title),
            patent_abstract=abstract or None,
            patent_claims=claims_text or None,
            patent_type=patent_type or None,
            patent_date=str(patent_date) if patent_date else None,
            applicant_assignee=applicant_assignee,
            continuity_family_id=continuity_family_id,
            uspc_class=uspc_class,
            uspc_subclass=uspc_subclass,
            cpc_classifications=cpc_classifications,
        )

    except Exception as exc:
        log.warning(
            "Could not build PatentRecord for %s: %s",
            patent_id,
            exc,
        )
        return None


def _build_odp_query(
    query: str,
    cpc_classifications: list[str] | None = None,
    uspc_classes: list[str] | None = None,
    classification_operator: str = "AND",
) -> str:
    """Build a title search constrained by classification criteria.

    Original title search is always ANDed with the classification
    expression. Within each classification family, values use OR.
    The supplied operator combines the CPC and USPC groups.
    """
    query = " ".join(query.split()).strip()
    if not query:
        return ""

    title_clause = f'applicationMetaData.inventionTitle:"{query}"'
    groups = []

    cpc_terms = [
        f"applicationMetaData.cpcClassificationBag:{normalized}"
        for item in (cpc_classifications or [])
        if (normalized := "".join(str(item).split()))
    ]
    if cpc_terms:
        groups.append(
            "(" + " OR ".join(cpc_terms) + ")"
        )

    uspc_terms = [
        f"applicationMetaData.class:{normalized}"
        for item in (uspc_classes or [])
        if (normalized := str(item).strip())
    ]
    if uspc_terms:
        groups.append(
            "(" + " OR ".join(uspc_terms) + ")"
        )

    if not groups:
        return title_clause

    operator = str(classification_operator or "AND").strip().upper()
    if operator not in {"AND", "OR"}:
        raise ValueError(
            "Classification operator must be AND or OR"
        )

    classification_expression = f" {operator} ".join(groups)

    return f"{title_clause} AND ({classification_expression})"


def search_patents(
    query: str,
    limit: int = 25,
    offset: int = 0,
    *,
    cpc_classifications: list[str] | None = None,
    uspc_classes: list[str] | None = None,
    classification_operator: str = "AND",
) -> list[PatentRecord]:
    """Search USPTO ODP and return PatentRecord objects.

    The ODP search endpoint uses Lucene-style query syntax.
    """

    headers = _build_headers()
    odp_query = _build_odp_query(
        query,
        cpc_classifications=cpc_classifications,
        uspc_classes=uspc_classes,
        classification_operator=classification_operator,
    )

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


def search_patents_by_strategy(
    query: str,
    limit: int = 25,
    offset: int = 0,
) -> list[PatentRecord]:
    """Search USPTO ODP using a provider-ready Boolean strategy expression.

    Retries HTTP 413 responses with smaller page sizes, preserving the
    query and offset. This helps avoid the USPTO's response-size limit.
    """
    headers = _build_headers()
    headers["Content-Type"] = "application/json"

    query = " ".join(str(query).split()).strip()

    if not query:
        return []

    requested_limit = max(1, min(limit, 100))
    retry_limits = [
        size for size in (25, 10, 5, 1)
        if size < requested_limit
    ]
    attempt_limits = [requested_limit, *retry_limits]

    # Remove duplicate sizes while preserving order.
    attempt_limits = list(dict.fromkeys(attempt_limits))

    response = None
    actual_limit = requested_limit

    for actual_limit in attempt_limits:
        payload = {
            "q": query,
            "pagination": {
                "offset": offset,
                "limit": actual_limit,
            },
        }

        response = requests.post(
            USPTO_SEARCH_URL,
            headers=headers,
            json=payload,
            timeout=30,
        )

        if response.status_code != 413:
            break

        log.warning(
            "USPTO response too large (413); retrying with limit=%d "
            "at offset=%d",
            actual_limit,
            offset,
        )

    if response is None:
        raise RuntimeError("USPTO strategy search was not attempted.")

    if response.status_code == 404:
        log.warning(
            "USPTO strategy search returned no matching records "
            "for query: %s",
            query,
        )
        return []

    if response.status_code == 413:
        raise RuntimeError(
            "USPTO strategy search exceeded the 6 MB response limit "
            "even at page size 1. The query or returned record fields "
            "may need further investigation."
        )

    if response.status_code != 200:
        raise RuntimeError(
            f"USPTO strategy search failed "
            f"({response.status_code}): {response.text[:500]}"
        )

    data = response.json()
    raw_records = data.get("patentFileWrapperDataBag", [])

    # Temporary diagnostic: inspect USPTO applicant/assignee metadata.
    print("\n=== USPTO RESPONSE DIAGNOSTIC ===")
    print("Top-level keys:", list(data.keys()))
    print("Number of records:", len(raw_records))

    if raw_records:
        record = raw_records[0]
        print("Record keys:", list(record.keys()))

        metadata = record.get("applicationMetaData") or {}
        print("Metadata keys:", list(metadata.keys()))

        for key, value in metadata.items():
            if any(term in key.lower() for term in (
                "applicant", "assignee", "inventor", "party", "organization"
            )):
                print(f"{key}: {value!r}")

    print("=== END DIAGNOSTIC ===\n")

    log.info(
        "USPTO strategy search returned %d raw records "
        "(requested limit=%d, actual limit=%d, offset=%d)",
        len(raw_records),
        requested_limit,
        actual_limit,
        offset,
    )

    records: list[PatentRecord] = []

    for raw in raw_records:
        if not isinstance(raw, dict):
            continue

        patent = _build_patent_record(raw)

        if patent is not None:
            records.append(patent)

    return records


def search_patents_with_classification(
    query: str,
    limit: int = 25,
    offset: int = 0,
    *,
    cpc_classifications: list[str] | None = None,
    uspc_classes: list[str] | None = None,
    classification_operator: str = "AND",
) -> list[PatentRecord]:
    """Search USPTO ODP using a structured classification filter.

    The title query is sent through q. Classification codes are sent
    through the endpoint's filters parameter.
    """
    headers = _build_headers()
    query = " ".join(query.split()).strip()

    if not query:
        return []

    cpc_values = list(dict.fromkeys(
        "".join(str(code).split())
        for code in (cpc_classifications or [])
        if str(code).strip()
    ))
    uspc_values = list(dict.fromkeys(
        str(code).strip()
        for code in (uspc_classes or [])
        if str(code).strip()
    ))

    if cpc_values and uspc_values:
        raise ValueError(
            "The ODP filters parameter accepts one classification field "
            "per request; CPC and USPC groups must be combined separately."
        )

    if cpc_values:
        filter_field = "applicationMetaData.cpcClassificationBag"
        filter_values = cpc_values
    elif uspc_values:
        filter_field = "applicationMetaData.class"
        filter_values = uspc_values
    else:
        filter_field = None
        filter_values = []

    params = {
        "q": f'applicationMetaData.inventionTitle:"{query}"',
        "offset": offset,
        "limit": min(limit, 100),
    }

    if filter_field:
        params["filters"] = f"{filter_field} {','.join(filter_values)}"

    log.info("USPTO ODP classified search parameters: %s", params)

    try:
        response = requests.get(
            USPTO_SEARCH_URL,
            headers=headers,
            params=params,
            timeout=30,
        )
    except requests.exceptions.RequestException as exc:
        log.error("USPTO ODP classified search network error: %s", exc)
        return []

    if response.status_code not in (200,):
        log.error(
            "USPTO ODP classified search failed: %s — %s",
            response.status_code,
            response.text[:500],
        )
        return []

    try:
        data = response.json()
    except ValueError:
        log.error("USPTO ODP classified search returned invalid JSON")
        return []

    raw_records = data.get("patentFileWrapperDataBag", [])
    log.info(
        "USPTO ODP classified search returned %d raw records",
        len(raw_records),
    )

    records: list[PatentRecord] = []

    for raw in raw_records:
        if not isinstance(raw, dict):
            continue

        patent = _build_patent_record(raw)
        if patent is not None:
            records.append(patent)

    return records