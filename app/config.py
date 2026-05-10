"""
app/config.py

Central configuration for the patent-search application.
Uses pydantic-settings so every value is type-validated and sourced from .env
automatically — no manual os.getenv() calls needed in service files.

Usage in any service or module:
    from app.config import settings
    api_key = settings.patentsview_api_key
    top_k   = settings.top_k_results

Optional operational limits (pagination caps) are plain module constants
defined below the Settings class — set to None to disable.
"""

from pathlib import Path
from typing import Optional
from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    All application settings loaded from the .env file at the project root.

    Required fields (must be present in .env):
        patentsview_api_key  — PatentsView API authentication key

    Optional fields (leave blank in .env to use the default):
        anthropic_api_key    — Anthropic Claude API key (needed for comparison + report)
        mistral_model        — Ollama model tag for query expansion
        embedding_model      — HuggingFace model ID for BGE embeddings
        bm25_weight          — Weight for BM25 scores in hybrid ranking (0–1)
        cosine_weight        — Weight for cosine scores in hybrid ranking (0–1)
        top_k_results        — Number of top patents returned to the LLM analysis stage
        patent_fields        — Comma-separated list of PatentsView fields to request
    """

    model_config = SettingsConfigDict(
        # .env is one level above this file (app/)
        env_file=str(Path(__file__).resolve().parents[1] / ".env"),
        env_file_encoding="utf-8",
        # Extra fields in .env are silently ignored
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # API base URLs — override in .env if endpoints change
    # ------------------------------------------------------------------
    patentsview_api_url: str = "https://search.patentsview.org/api/v1/patent/"
    lens_api_url: str = "https://api.lens.org/patent/search"

    # ------------------------------------------------------------------
    # API keys — read from .env
    # ------------------------------------------------------------------
    patentsview_api_key: Optional[str] = None  # legacy PatentsView (now offline)
    lens_api_token: Optional[str] = None        # Lens.org Patent API bearer token
    anthropic_api_key: Optional[str] = None     # set when Claude analysis is enabled

    # ------------------------------------------------------------------
    # Search backend selection
    # ------------------------------------------------------------------
    # "lens"    — Lens.org Patent API (requires lens_api_token)
    # "fixture" — offline fixture data (for testing / demo without an API)
    # "patentsview" — legacy (offline as of March 2026, kept for reference)
    search_backend: str = "fixture"  # default to fixture until API token is set

    # ------------------------------------------------------------------
    # Model selection
    # ------------------------------------------------------------------
    mistral_model: str = "mistral"             # Ollama model tag; e.g. "mistral:7b-instruct"
    embedding_model: str = "BAAI/bge-large-en-v1.5"  # sentence-transformers model ID

    # ------------------------------------------------------------------
    # Hybrid ranking weights (must sum to 1.0)
    # ------------------------------------------------------------------
    bm25_weight: float = 0.3    # 0.7/0.3 split: semantic similarity matters more for patents
    cosine_weight: float = 0.7  # tune via .env: bm25_weight=... / cosine_weight=...

    # ------------------------------------------------------------------
    # Pipeline behaviour
    # ------------------------------------------------------------------
    top_k_results: int = 10   # patents forwarded to LLM comparison stage (PoC: top-10)

    # Number of top candidates to send to Claude / LLM for detailed analysis.
    # fix2.txt recommends verifying top 5 manually and running the LLM on top 3.
    claude_top_k: int = 3

    # Cosine threshold: patents scoring below this are discarded before ranking.
    # Empirical scale:  <0.30 unrelated | 0.30–0.40 weak | 0.50 meaningful | 0.65 strong
    # Set to 0.0 in .env to disable filtering entirely.
    cosine_threshold: float = 0.45       # hard floor (PoC default)
    cosine_threshold_min: float = 0.30   # never drop below this when auto-relaxing
    cosine_min_candidates: int = 5       # guarantee at least this many pass the filter

    # Hybrid threshold: require a minimum hybrid score for final candidates.
    # Set to 0.0 in .env to disable hybrid filtering entirely.
    hybrid_threshold: float = 0.45

    # ------------------------------------------------------------------
    # MySQL database credentials (read from .env)
    # ------------------------------------------------------------------
    db_host: str = "localhost"
    db_user: str = "root"
    db_password: str = ""
    db_name: str = "patent_db"

    # Constructed automatically from the four fields above.
    # Can be overridden by setting db_url directly in .env.
    db_url: str = ""

    @model_validator(mode="after")
    def _build_db_url(self) -> "Settings":
        """Build db_url from individual credential fields if not set explicitly."""
        if not self.db_url:
            from urllib.parse import quote_plus
            pwd = quote_plus(self.db_password)
            self.db_url = (
                f"mysql+mysqlconnector://{self.db_user}:{pwd}"
                f"@{self.db_host}/{self.db_name}"
            )
        return self

    @model_validator(mode="after")
    def _ensure_weights_sum(self) -> "Settings":
        """Ensure bm25_weight + cosine_weight sums to 1.0.

        If the user set weights in .env that don't sum to 1.0 we normalise them
        proportionally so the hybrid formula remains stable.
        """
        total = float(self.bm25_weight + self.cosine_weight)
        if abs(total - 1.0) > 1e-6 and total > 0.0:
            self.bm25_weight = float(self.bm25_weight / total)
            self.cosine_weight = float(self.cosine_weight / total)
        return self

    # PatentsView fields requested on every query.
    # Stored as a list; pydantic-settings reads comma-separated values from .env.
    # Includes patent_date for temporal analysis.
    patent_fields: list[str] = [
        "patent_id",
        "patent_title",
        "patent_abstract",
        "patent_type",
        "patent_date",
    ]

    @field_validator("bm25_weight", "cosine_weight")
    @classmethod
    def _weight_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("Ranking weights must be between 0.0 and 1.0")
        return v


# ---------------------------------------------------------------------------
# Module-level singleton — import this everywhere
# ---------------------------------------------------------------------------
settings = Settings()

# ---------------------------------------------------------------------------
# Operational constants — not environment-driven, but centralised here so
# all services agree on the same values.
# ---------------------------------------------------------------------------

# Hard API limit: PatentsView returns at most this many records per request.
# Uses the 'size' key in the options object (not 'per_page').
PATENT_PAGE_SIZE = 1000

# Maximum total patent documents to return across all pages.
# The user is notified when this cap is reached.
# Set to None to disable the document cap.
MAX_RESULTS = 500        # raised from 5 (was a debug POC limit)

# Maximum number of API pages (requests) per query.
# Set to None to disable the page cap.
MAX_PAGES = 10           # ← change to allow more pages
