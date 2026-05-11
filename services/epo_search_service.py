"""
services/epo_search_service.py

EPO Open Patent Services (OPS) search backend.

Authenticates via OAuth2 client-credentials and queries the
published-data/search endpoint using CQL (Contextual Query Language).

Public API (same signature as search_service / lens_search_service)
----------
fetch_all_patents(query_terms) -> list[PatentRecord]

CQL field codes used:
  ti=<term>  — title
  ab=<term>  — abstract
  ta=<term>  — title OR abstract (any)
  cl=<term>  — claims
"""

from __future__ import annotations

import base64
import logging
import time
from typing import Optional

import requests

from app.config import settings
from models.schemas import PatentRecord
from utils.logger import get_logger

log = get_logger(__name__)

_TOKEN_URL = "https://ops.epo.org/3.2/auth/accesstoken"
_SEARCH_URL = f"{settings.epo_api_url}/published-data/search"
_DETAIL_BASE = f"{settings.epo_api_url}/published-data/publication/epodoc"

# EPO OPS rate limit: 30 requests/minute on free tier
_RESULTS_PER_PAGE = 25   # EPO returns max 100 per request (Range header)
_MAX_RESULTS = 100        # cap per query term to avoid rate limits


class _TokenCache:
    """Simple in-process OAuth2 token cache."""
    token: Optional[str] = None
    expires_at: float = 0.0


_cache = _TokenCache()


def _get_access_token() -> str:
    """Return a valid bearer token, refreshing if expired."""
    now = time.time()
    if _cache.token and now < _cache.expires_at - 30:
        return _cache.token

    if not settings.epo_consumer_key or not settings.epo_consumer_secret:
        raise RuntimeError(
            "EPO credentials not configured. "
            "Set epo_consumer_key and epo_consumer_secret in .env"
        )

    credentials = f"{settings.epo_consumer_key}:{settings.epo_consumer_secret}"
    encoded = base64.b64encode(credentials.encode()).decode()

    resp = requests.post(
        _TOKEN_URL,
        headers={
            "Authorization": f"Basic {encoded}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        data="grant_type=client_credentials",
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    _cache.token = data["access_token"]
    _cache.expires_at = now + int(data.get("expires_in", 1200))
    log.info("EPO OPS token refreshed, expires_in=%s", data.get("expires_in"))
    return _cache.token


def _build_cql(terms: list[str]) -> str:
    """
    Build a CQL query string that searches title AND abstract.

    Each term becomes a `ta=<term>` clause (title-or-abstract).
    Multiple terms are joined with AND for precision, but if only
    one term is given it's used directly.
    """
    clauses = [f'ta="{t}"' for t in terms if t.strip()]
    if not clauses:
        return 'ta="patent"'
    return " AND ".join(clauses)


def _search_epo(cql: str, start: int = 1, count: int = _RESULTS_PER_PAGE) -> dict:
    """Execute one EPO OPS search request. Returns parsed JSON."""
    token = _get_access_token()
    end = start + count - 1
    resp = requests.get(
        _SEARCH_URL,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "X-OPS-Range": f"{start}-{end}",
        },
        params={"q": cql},
        timeout=20,
    )
    if resp.status_code == 404:
        # EPO returns 404 when a query matches nothing
        return {}
    resp.raise_for_status()
    return resp.json()


def _parse_search_refs(data: dict) -> list[tuple[str, str]]:
    """
    Extract (patent_id, epodoc_ref) pairs from an EPO OPS search response.

    patent_id  — concatenated string e.g. ``US20260127435A1``
    epodoc_ref — EPO epodoc path segment e.g. ``US20260127435.A1``
                 (dot before kind code, as required by the detail endpoint)
    """
    refs: list[tuple[str, str]] = []
    try:
        results = (
            data.get("ops:world-patent-data", {})
                .get("ops:biblio-search", {})
                .get("ops:search-result", {})
                .get("ops:publication-reference", [])
        )
        if isinstance(results, dict):
            results = [results]
    except (AttributeError, KeyError):
        return refs

    for ref in results:
        try:
            doc_id  = ref.get("document-id", {})
            country = doc_id.get("country", {}).get("$", "")
            number  = doc_id.get("doc-number", {}).get("$", "")
            kind    = doc_id.get("kind", {}).get("$", "")
            patent_id  = f"{country}{number}{kind}".strip() or number
            epodoc_ref = f"{country}{number}.{kind}" if kind else f"{country}{number}"
            refs.append((patent_id, epodoc_ref))
        except Exception as exc:
            log.debug("Skipping EPO search ref: %s", exc)

    return refs


def _parse_exchange_doc(doc: dict, fallback_id: str) -> PatentRecord | None:
    """
    Parse a single EPO ``exchange-document`` node into a PatentRecord.

    Handles both ``biblio`` and ``abstract`` constituents.
    """
    try:
        biblio = doc.get("bibliographic-data", {})

        # --- Title: prefer English, fall back to first available ---
        titles = biblio.get("invention-title", [])
        if isinstance(titles, dict):
            titles = [titles]
        title = next(
            (t.get("$", "") for t in titles if t.get("@lang") == "en"),
            titles[0].get("$", fallback_id) if titles else fallback_id,
        )

        # --- Abstract: prefer English ---
        abstracts = doc.get("abstract", [])
        if isinstance(abstracts, dict):
            abstracts = [abstracts]
        abstract = ""
        for a in abstracts:
            if a.get("@lang") != "en":
                continue
            paragraphs = a.get("p", [])
            if isinstance(paragraphs, dict):
                paragraphs = [paragraphs]
            if isinstance(paragraphs, list):
                abstract = " ".join(
                    p.get("$", "") if isinstance(p, dict) else str(p)
                    for p in paragraphs
                ).strip()
            elif isinstance(paragraphs, str):
                abstract = paragraphs.strip()
            if abstract:
                break

        # --- Patent ID + date from publication-reference ---
        patent_id = fallback_id
        date_str  = ""
        pub_ref   = biblio.get("publication-reference", {})
        doc_ids   = pub_ref.get("document-id", [])
        if isinstance(doc_ids, dict):
            doc_ids = [doc_ids]
        for did in doc_ids:
            if did.get("@document-id-type") in ("epodoc", "docdb"):
                c = did.get("country", {}).get("$", "")
                n = did.get("doc-number", {}).get("$", "")
                k = did.get("kind", {}).get("$", "")
                d = did.get("date", {}).get("$", "")
                if c and n:
                    patent_id = f"{c}{n}{k}".strip()
                    date_str  = d
                    break

        return PatentRecord(
            patent_id=patent_id,
            patent_title=title or fallback_id,
            patent_abstract=abstract or None,
            patent_type=None,
            patent_date=date_str or None,
        )
    except (KeyError, TypeError, AttributeError) as exc:
        log.warning("Failed to parse EPO exchange-doc for %s: %s", fallback_id, exc)
        return None


def _fetch_details_batch(
    refs: list[tuple[str, str]],
    chunk_size: int = 10,
) -> list[PatentRecord]:
    """
    Fetch ``biblio,abstract`` for a list of (patent_id, epodoc_ref) pairs.

    EPO OPS supports comma-separated multi-document requests up to ~50 refs.
    We chunk into groups of *chunk_size* to stay well within URL-length limits
    and respect the 30 req/min rate-limit on the free developer tier.
    """
    records: list[PatentRecord] = []

    for i in range(0, len(refs), chunk_size):
        chunk     = refs[i : i + chunk_size]
        ref_path  = ",".join(epodoc for _, epodoc in chunk)
        fallbacks = [pid for pid, _ in chunk]

        url   = f"{_DETAIL_BASE}/{ref_path}/biblio,abstract"
        token = _get_access_token()

        try:
            resp = requests.get(
                url,
                headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
                timeout=30,
            )
            if resp.status_code in (400, 404):
                log.warning("EPO detail batch %d–%d returned %d", i + 1, i + len(chunk), resp.status_code)
                records.extend(
                    PatentRecord(patent_id=pid, patent_title=pid, patent_abstract=None,
                                 patent_type=None, patent_date=None)
                    for pid in fallbacks
                )
                continue
            resp.raise_for_status()
            data = resp.json()

            docs = (
                data.get("ops:world-patent-data", {})
                    .get("exchange-documents", {})
                    .get("exchange-document", [])
            )
            if isinstance(docs, dict):
                docs = [docs]

            for doc, pid in zip(docs, fallbacks):
                record = _parse_exchange_doc(doc, pid)
                records.append(record if record else
                               PatentRecord(patent_id=pid, patent_title=pid,
                                            patent_abstract=None, patent_type=None, patent_date=None))

        except Exception as exc:
            log.warning("EPO detail batch failed: %s", exc)
            records.extend(
                PatentRecord(patent_id=pid, patent_title=pid, patent_abstract=None,
                             patent_type=None, patent_date=None)
                for pid in fallbacks
            )

        if i + chunk_size < len(refs):
            time.sleep(1.0)  # stay within 30 req/min free-tier limit

    return records


def _fetch_for_terms(terms: list[str]) -> list[PatentRecord]:
    """
    Fetch up to _MAX_RESULTS patents for a CQL built from *terms*.

    Step 1: paginate the search endpoint to collect (patent_id, epodoc_ref) pairs.
    Step 2: batch-fetch biblio+abstract for all collected refs so that
            every returned PatentRecord has a real title and abstract.
    """
    cql = _build_cql(terms)
    log.info("EPO OPS search CQL: %s", cql)

    all_refs: list[tuple[str, str]] = []
    start = 1
    while len(all_refs) < _MAX_RESULTS:
        try:
            data = _search_epo(cql, start=start, count=_RESULTS_PER_PAGE)
        except requests.HTTPError as exc:
            log.error("EPO OPS HTTP error: %s", exc)
            break

        if not data:
            break

        page_refs = _parse_search_refs(data)
        if not page_refs:
            break

        # Deduplicate on patent_id
        seen = {pid for pid, _ in all_refs}
        new_refs = [(pid, ref) for pid, ref in page_refs if pid not in seen]
        all_refs.extend(new_refs)
        log.info("EPO OPS page start=%d: %d refs (total %d)", start, len(new_refs), len(all_refs))

        if len(page_refs) < _RESULTS_PER_PAGE:
            break  # last page

        start += _RESULTS_PER_PAGE
        time.sleep(0.5)  # respect rate limit between search pages

    if not all_refs:
        return []

    log.info("EPO OPS: fetching biblio+abstract for %d patents", len(all_refs))
    return _fetch_details_batch(all_refs)


def fetch_all_patents(query_terms: list[str]) -> list[PatentRecord]:
    """
    Search EPO OPS for *query_terms* and return combined PatentRecord list.

    Implements a 3-stage fallback:
    Stage 1 — all terms combined (most precise)
    Stage 2 — each term searched individually (broader)
    Stage 3 — first term only, no quotes (maximum recall)

    Falls back to fixture data if EPO credentials are not configured.
    """
    MIN_RESULTS = 15

    # Stage 1: all terms together
    try:
        results = _fetch_for_terms(query_terms)
        log.info("EPO stage 1 (%d terms): %d patents", len(query_terms), len(results))
        if len(results) >= MIN_RESULTS:
            return results
    except Exception as exc:
        log.error("EPO stage 1 failed: %s", exc)
        results = []

    # Stage 2: each term individually
    seen_ids: set[str] = {p.patent_id for p in results}
    for term in query_terms[:5]:  # cap to avoid rate-limit
        try:
            batch = _fetch_for_terms([term])
            new = [p for p in batch if p.patent_id not in seen_ids]
            results.extend(new)
            seen_ids.update(p.patent_id for p in new)
            log.info("EPO stage 2 term='%s': +%d patents (total %d)", term, len(new), len(results))
            if len(results) >= MIN_RESULTS:
                break
            time.sleep(0.5)
        except Exception as exc:
            log.warning("EPO stage 2 term '%s' failed: %s", term, exc)

    if len(results) >= MIN_RESULTS:
        return results

    # Stage 3: first term, single-word fallback (no quotes)
    if query_terms:
        first_word = query_terms[0].split()[0]
        try:
            data = _search_epo(f"ta={first_word}", start=1, count=_RESULTS_PER_PAGE)
            stage3_refs = _parse_search_refs(data)
            stage3_full = _fetch_details_batch(
                [(pid, ref) for pid, ref in stage3_refs if pid not in seen_ids]
            )
            new = [p for p in stage3_full if p.patent_id not in seen_ids]
            results.extend(new)
            log.info("EPO stage 3 word='%s': +%d patents (total %d)", first_word, len(new), len(results))
        except Exception as exc:
            log.warning("EPO stage 3 failed: %s", exc)

    if not results:
        log.warning("EPO returned 0 results after all stages — falling back to fixture")
        from services.fixture_search_service import fetch_all_patents as fixture_fetch
        return fixture_fetch(query_terms)

    return results
