"""
services/fixture_search_service.py

Offline fallback search backend using local fixture JSON files.

Used when no live API (PatentsView / Lens.org / EPO) is available.
Returns patents from tests/fixtures/ that fuzzy-match the query terms,
so ranking, embedding, and Claude stages can still be demonstrated.

Public API (same signature as search_service / lens_search_service)
----------
fetch_all_patents(query_terms) -> list[PatentRecord]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.schemas import PatentRecord  # noqa: E402
from utils.logger import get_logger  # noqa: E402

log = get_logger(__name__)

_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE_PATH = _ROOT / "tests" / "fixtures" / "patentsview_page_full.json"


def _load_fixtures() -> list[PatentRecord]:
    """Load all patents from the fixture file."""
    if not _FIXTURE_PATH.exists():
        log.warning("Fixture file not found: %s", _FIXTURE_PATH)
        return []
    try:
        data = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
        records = []
        for p in data.get("patents", []):
            try:
                records.append(PatentRecord(**p))
            except Exception as exc:
                log.warning("Skipping bad fixture record: %s", exc)
        return records
    except Exception as exc:
        log.error("Failed to load fixtures: %s", exc)
        return []


def _score_relevance(patent: PatentRecord, term: str) -> int:
    """Simple keyword overlap score between patent text and a search term."""
    term_words = set(term.lower().split())
    text = f"{patent.patent_title or ''} {patent.patent_abstract or ''}".lower()
    return sum(1 for w in term_words if len(w) > 2 and w in text)


def fetch_all_patents(query_terms: list[str]) -> list[PatentRecord]:
    """
    Return fixture patents that best match the given query terms.

    All fixture patents are returned (the set is small), ordered by
    relevance to the first query term. This lets ranking/embedding
    stages operate on a realistic-looking result set.
    """
    log.info("fixture_search: using offline fixtures (no live API)")
    print("ℹ️  Using offline fixture data (no live API configured).")

    all_fixtures = _load_fixtures()
    if not all_fixtures:
        return []

    # Score each patent across all terms
    scored = []
    for patent in all_fixtures:
        score = sum(_score_relevance(patent, t) for t in query_terms)
        scored.append((score, patent))

    # Return all, sorted by relevance (best first)
    scored.sort(key=lambda x: x[0], reverse=True)
    results = [p for _, p in scored]

    log.info("fixture_search: returning %d fixture patents", len(results))
    return results
