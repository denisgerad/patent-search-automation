import os
import requests
import json
import sys
from pathlib import Path

# Ensure the project root is importable regardless of the launch directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import PATENT_FIELDS, PATENT_PAGE_SIZE, MAX_RESULTS, MAX_PAGES  # noqa: E402


def fetch_patents_by_keywords(patent_title, patent_abstract, patent_type):
    """Fetches patents from the PatentsView API based on keywords."""
    # Use documented PatentsView endpoint
    url = "https://search.patentsview.org/api/v1/patent/"

    # Build query using operator-first format per API docs
    or_clauses = []
    if patent_title:
        or_clauses.append({"_text_any": {"patent_title": patent_title}})
    if patent_abstract:
        or_clauses.append({"_text_any": {"patent_abstract": patent_abstract}})
    if patent_type:
        # use equality for patent_type when provided
        or_clauses.append({"_eq": {"patent_type": patent_type}})

    query = {"_or": or_clauses} if or_clauses else {"_text_any": {"patent_title": ""}}

    # Issue 1 — API key is loaded by load_dotenv() in config.py at startup;
    # no manual .env parsing needed here.
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    api_key = os.getenv("api_key") or os.getenv("API_KEY")
    if api_key:
        headers["X-Api-Key"] = api_key

    # Paginate through results using cursor-based pagination (the 'after' parameter).
    # PatentsView replaced the old 'offset' parameter with 'after', which takes the
    # patent_id of the last record from the previous page.
    # Both MAX_RESULTS and MAX_PAGES limits are defined in config.py;
    # whichever is hit first stops the loop and notifies the user.
    all_patents: list = []
    after_cursor = None   # None on the first request; set to last patent_id thereafter
    page_number = 0       # counts how many requests have been made

    while True:
        # ── Page cap check ──────────────────────────────────────────────────
        # MAX_PAGES is set in config.py (currently 10).
        # Change MAX_PAGES in config.py to fetch more or fewer pages.
        if MAX_PAGES is not None and page_number >= MAX_PAGES:
            print(
                f"⚠️  Results limited: reached the maximum of {MAX_PAGES} page(s). "
                f"Returning {len(all_patents)} document(s). "
                "To fetch more, increase MAX_PAGES in config.py."
            )
            break

        # ── Document cap check ──────────────────────────────────────────────
        # MAX_RESULTS is set in config.py (currently 5).
        # Change MAX_RESULTS in config.py to retrieve more or fewer documents.
        if MAX_RESULTS is not None:
            remaining = MAX_RESULTS - len(all_patents)
            if remaining <= 0:
                print(
                    f"⚠️  Results limited to {MAX_RESULTS} document(s). "
                    "To fetch more, increase MAX_RESULTS in config.py."
                )
                break
            # Only ask for as many records as we still need on this page.
            size = min(PATENT_PAGE_SIZE, remaining)
        else:
            size = PATENT_PAGE_SIZE

        # Build the options object.
        # PatentsView cursor pagination:
        #   - First page:  {"size": N}
        #   - Later pages: {"size": N, "after": "<last_patent_id_from_previous_page>"}
        options: dict = {"size": size}
        if after_cursor is not None:
            options["after"] = after_cursor

        # Issue 2 — Field list comes from config.PATENT_FIELDS so all services
        # agree on the schema without repeating the array.
        params = {
            "q": query,
            "f": PATENT_FIELDS,
            "o": options,
        }

        try:
            response = requests.post(url, headers=headers, json=params)
            page_number += 1

            # If non-200, print short debug info and abort pagination.
            if response.status_code != 200:
                print(f"❌ API Request Error: {response.status_code} {response.reason} for url: {response.url}")
                for h in ("X-Status-Reason", "X-Status-Reason-Code", "Content-Type"):
                    if h in response.headers:
                        print(f"  {h}: {response.headers[h]}")
                print((response.text or "")[:1000])
                break

            try:
                data = response.json()
            except json.JSONDecodeError:
                print("❌ Failed to decode JSON response.")
                break

            # Handle API-level error payloads like {"error": true, "reason": "..."}
            if isinstance(data, dict) and data.get("error"):
                reason = data.get("reason") or data.get("message") or "unknown"
                print(f"❌ API returned error: {reason}")
                break

            page = data.get("patents", []) if isinstance(data, dict) else []
            all_patents.extend(page)

            # A page shorter than the requested size means the API has no more results.
            if len(page) < size:
                break

            # Advance the cursor to the patent_id of the last record on this page.
            # This value is passed as 'after' in the next request.
            after_cursor = page[-1].get("patent_id")
            if after_cursor is None:
                # Cannot paginate further without a cursor value.
                print("⚠️  Could not determine cursor for next page (patent_id missing). Stopping.")
                break

        except requests.exceptions.RequestException as e:
            print(f"❌ API Request Error: {e}")
            break

    return all_patents


if __name__ == "__main__":
    # Simple CLI demo: prints fetched patents (safe against API errors)
    title = sys.argv[1] if len(sys.argv) > 1 else "machine learning"
    abstract = sys.argv[2] if len(sys.argv) > 2 else "neural network"
    ptype = sys.argv[3] if len(sys.argv) > 3 else ""

    patents = fetch_patents_by_keywords(title, abstract, ptype)
    print(json.dumps(patents, indent=2))