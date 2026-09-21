"""Retrieve authoritative patent classification definitions from USPTO."""

from __future__ import annotations

import re
from functools import lru_cache
from html.parser import HTMLParser

import requests


USPTO_BASE = "https://www.uspto.gov"

_REQUEST_TIMEOUT = 15


class _TextParser(HTMLParser):
    """Small HTML-to-text parser without requiring BeautifulSoup."""

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        text = " ".join(data.split())
        if text:
            self.parts.append(text)

    def text(self) -> str:
        return " ".join(self.parts)


def _fetch_text(url: str) -> str:
    response = requests.get(
        url,
        timeout=_REQUEST_TIMEOUT,
        headers={
            "User-Agent": "PatentSearchResearchTool/1.0",
        },
    )
    response.raise_for_status()

    parser = _TextParser()
    parser.feed(response.text)

    return parser.text()


def _fetch_html(url: str) -> str:
    response = requests.get(
        url,
        timeout=_REQUEST_TIMEOUT,
        headers={
            "User-Agent": "PatentSearchResearchTool/1.0",
        },
    )
    response.raise_for_status()

    return response.text


def _cpc_section(classification: str) -> str:
    """
    Return the CPC section/subclass used by the USPTO scheme page.

    Example:
        G06V 20/588 -> G06V
    """
    match = re.match(
        r"\s*([A-HY]\d{2}[A-Z])",
        classification.upper(),
    )

    if not match:
        raise ValueError(
            f"Unable to determine CPC section from '{classification}'."
        )

    return match.group(1)


def _normalize_cpc(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().upper())


@lru_cache(maxsize=128)
def get_cpc_definition(classification: str) -> str:
    """Retrieve the relevant CPC definition text from USPTO."""

    classification = _normalize_cpc(classification)

    if not classification:
        return ""

    section = _cpc_section(classification)

    url = (
        f"{USPTO_BASE}/web/patents/classification/"
        f"cpc/html/cpc-{section}.html"
    )

    try:
        text = _fetch_html(url)
    except requests.RequestException:
        return ""

    return _extract_classification_definition(
        text,
        classification,
    )


def _normalize_uspc_class(value: str) -> str:
    match = re.search(r"\d+", str(value))

    if not match:
        raise ValueError(
            f"Invalid USPC class '{value}'."
        )

    return match.group(0)


@lru_cache(maxsize=128)
def get_uspc_definition(
    uspc_class: str,
    uspc_subclass: str = "",
) -> str:
    """Retrieve the USPC class definition from USPTO."""

    uspc_class = _normalize_uspc_class(uspc_class)
    uspc_subclass = str(uspc_subclass or "").strip()

    url = (
        f"{USPTO_BASE}/web/patents/classification/"
        f"uspc{uspc_class}/defs{uspc_class}.htm"
    )

    text = _fetch_text(url)

    if uspc_subclass:
        subclass_definition = _extract_uspc_subclass_definition(
            text,
            uspc_subclass,
        )

        if subclass_definition:
            return subclass_definition

    return _extract_uspc_class_definition(
        text,
        uspc_class,
    )


def _extract_classification_definition(
    text: str,
    classification: str,
) -> str:
    """Extract a CPC classification description from a USPTO scheme page."""

    normalized = re.sub(
        r"\s+",
        "",
        classification.strip().upper(),
    )

    if not normalized:
        return ""

    # Find the table containing the exact CPC classification ID.
    pattern = (
        rf'<table[^>]*id="{re.escape(normalized)}"[^>]*>'
        rf'.*?'
        rf'<span[^>]*class="cpc-text"[^>]*>'
        rf'(.*?)'
        rf'</span>'
    )

    match = re.search(
        pattern,
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if not match:
        return ""

    definition = re.sub(
        r"<[^>]+>",
        " ",
        match.group(1),
    )

    definition = definition.replace("&nbsp;", " ")

    return " ".join(definition.split()).strip()


def _extract_uspc_class_definition(
    text: str,
    uspc_class: str,
) -> str:
    """Extract the main USPC class definition."""

    normalized_text = re.sub(
        r"\s+",
        " ",
        text,
    )

    pattern = rf"CLASS\s+{re.escape(uspc_class)}\s*,\s*(.{{0,1800}})"

    match = re.search(
        pattern,
        normalized_text,
        flags=re.IGNORECASE,
    )

    if not match:
        return ""

    return (
        f"Class {uspc_class}: "
        + " ".join(match.group(1).split())
    )[:1500]


def _extract_uspc_subclass_definition(
    text: str,
    subclass: str,
) -> str:
    """
    Find a USPC subclass entry.

    USPC subclass numbering can contain decimals, so matching is
    deliberately conservative.
    """

    subclass = subclass.strip()

    pattern = rf"(?:^|\s){re.escape(subclass)}\s+(.{{0,1000}}?)(?=\s+\d+(?:\.\d+)?\s+|\Z)"

    match = re.search(
        pattern,
        text,
        flags=re.IGNORECASE,
    )

    if not match:
        return ""

    return match.group(0).strip()[:1200]
