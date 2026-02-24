"""
app/config.py

Central configuration for the patent-search application.

This is the single source of truth for all runtime settings.
It lives in app/ alongside the FastAPI entrypoint (main.py, api.py).

Usage in any service or module:
    from app.config import PATENT_FIELDS, MAX_RESULTS, ...

load_dotenv() is called once here at import time.  Every subsequent
os.getenv() call anywhere in the codebase will see the values from .env
without any module needing to parse the file itself.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# .env lives at the project root, one level above this file (app/)
_env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=_env_path)

# ------------------------------------------------------------------
# PatentsView API
# ------------------------------------------------------------------

# Fields requested from the PatentsView API for every patent query.
# Centralised here so all services agree on the schema without
# repeating the list.
PATENT_FIELDS = ["patent_id", "patent_title", "patent_abstract", "patent_type"]

# Maximum results the API returns per request (hard API limit).
# NOTE: PatentsView uses the key 'size' (not 'per_page') in the options object.
PATENT_PAGE_SIZE = 1000

# ------------------------------------------------------------------
# Result limits — change these two values to control how much data
# fetch_patents_by_keywords() retrieves before stopping.
# ------------------------------------------------------------------

# Maximum total number of patent documents to return across all pages.
# The user will be notified when this cap is reached.
# Set to None to disable the document cap.
MAX_RESULTS = 5          # ← change this number to allow more documents

# Maximum number of API pages (requests) to fetch per query.
# Each page holds up to PATENT_PAGE_SIZE records.
# The user will be notified when this cap is reached.
# Set to None to disable the page cap.
MAX_PAGES = 10           # ← change this number to allow more pages
