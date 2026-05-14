"""
orchestrator/patent_pipeline.py

Single deterministic entry point for the patent intelligence pipeline.

Stage order:
  1. Query expansion  (Mistral)
  2. Patent search    (PatentsView API)
  3. Deduplication
  4. Embedding        (BGE-large)
  5. Hybrid ranking   (cosine + BM25)
  6. Comparison       (Claude)
  7. Report           (Claude)
  8. Persist to MySQL + JSON cache

Each stage logs its input and output count so you can trace exactly where
the funnel narrows.  All external calls are injected as arguments so the
pipeline can be tested with mocks without monkey-patching.
"""
import logging
from enum import Enum
from pathlib import Path

from app.config import settings
from models.claude_client import ClaudeClient
from models.embedding_model import EmbeddingModel
from models.schemas import PipelineResult, RankedPatent
from services import (
    comparison_service,
    constraint_validator,
    dedup_service,
    query_expansion,
    report_service,
)
from services import embedding_service, ranking_service
from services.relevance_filter import filter_irrelevant_patents
from services.search_service import fetch_patents_for_query_dict, fetch_patents_with_fallback
from storage.patent_repository import save_patents

logger = logging.getLogger(__name__)

# Minimum number of patents required from the broad tier before we consider
# the search adequately seeded.  Below this threshold the pipeline still
# continues — the narrow tiers may still contribute results.
_MIN_RESULTS_THRESHOLD: int = 10


# ---------------------------------------------------------------------------
# Backend routing
# ---------------------------------------------------------------------------

class SearchBackend(Enum):
    PATENTSVIEW = "patentsview"
    EPO = "epo"


def _resolve_backend() -> SearchBackend:
    """
    Single source of truth for backend selection.
    Evaluated ONCE at pipeline start.

    Bug 1 fix: .strip() before truth-check so trailing whitespace in .env
               never causes silent fallthrough.
    Raises immediately if no key is configured — never proceeds undefined.
    """
    epo_key = (settings.epo_consumer_key or "").strip()
    pv_key  = (settings.patentsview_api_key or "").strip()

    if epo_key:
        logger.info("Backend resolved: EPO (epo_consumer_key present)")
        return SearchBackend.EPO

    if pv_key:
        logger.info("Backend resolved: PatentsView (patentsview_api_key present)")
        return SearchBackend.PATENTSVIEW

    raise RuntimeError(
        "No API key found for any search backend. "
        "Set patentsview_api_key or epo_consumer_key in .env"
    )


def _search(
    backend: SearchBackend,
    tokens,
    validated_queries: list[str],
) -> list:
    """
    Bug 3 fix: backends are mutually exclusive — no cross-fallback.
    PatentsView is never called when backend == EPO, and vice versa.
    """
    if backend == SearchBackend.EPO:
        return _search_epo(tokens, validated_queries)
    if backend == SearchBackend.PATENTSVIEW:
        return _search_patentsview(tokens, validated_queries)
    raise ValueError(f"Unhandled backend: {backend}")


def _search_epo(tokens, validated_queries: list[str]) -> list:
    """
    Uses tokens.epo_search_order (most discriminating term first) so EPO's
    AND chain filters on the most specific concept before less specific ones.
    Falls back to critical_tokens / long words if epo_search_order is empty.
    """
    from services.epo_search_service import epo_fetch_by_keywords

    # Primary: use epo_search_order set by Claude pre-call or specificity sort
    keyword_terms: list[str] = list(getattr(tokens, "epo_search_order", []))[:4]

    # Backward compat fallbacks
    if not keyword_terms:
        keyword_terms = list(tokens.critical_tokens[:4])
    if not keyword_terms and validated_queries:
        keyword_terms = [
            w for w in validated_queries[0].lower().split() if len(w) > 5
        ][:4]

    logger.info(
        "EPO CQL order (most→least discriminating): %s", keyword_terms,
    )
    results = epo_fetch_by_keywords(keyword_terms)
    logger.info("EPO returned %d patents", len(results))

    if not results:
        logger.warning(
            "EPO returned 0 patents. Keywords: %s — "
            "consider broadening taxonomy in token_extractor.py",
            keyword_terms,
        )
    return results


def _search_patentsview(tokens, validated_queries: list[str]) -> list:
    """
    Three-stage PatentsView search. Never calls EPO.
    Returns [] with a clear error log if all stages fail.
    """
    MIN = 15
    results: list = []
    seen_ids: set[str] = set()

    def _add(batch: list) -> None:
        for p in batch:
            if p.patent_id not in seen_ids:
                seen_ids.add(p.patent_id)
                results.append(p)

    from services.search_service import fetch_all_patents

    # Stage 1: phrase queries from Mistral expansion
    _add(fetch_all_patents(validated_queries))
    logger.info("PatentsView stage 1: %d patents", len(results))
    if len(results) >= MIN:
        return results

    # Stage 2: single critical token, _text_any
    for token in tokens.critical_tokens[:4]:
        batch = fetch_patents_for_query_dict({
            "_or": [
                {"_text_any": {"patent_title":    token}},
                {"_text_any": {"patent_abstract": token}},
            ]
        })
        _add(batch)
        logger.info("PatentsView stage 2 token '%s': %d", token, len(batch))
        if len(results) >= MIN:
            return results

    # Stage 3: domain concept, abstract only
    for concept in tokens.domain_concepts[:2]:
        term = concept.replace("_", " ")
        _add(fetch_patents_for_query_dict({"_text_any": {"patent_abstract": term}}))
        logger.info("PatentsView stage 3 concept '%s': %d total", concept, len(results))

    if not results:
        logger.error(
            "PatentsView returned 0 patents across all 3 stages. "
            "API key: %s | Check connectivity and field names.",
            "present" if settings.patentsview_api_key else "MISSING",
        )
    return results


def _assert_source(patents: list, backend: SearchBackend) -> None:
    """Log a loud warning if records don't match the expected backend."""
    for p in patents[:5]:
        is_numeric = (p.patent_id or "").replace("-", "").isdigit()
        if backend == SearchBackend.PATENTSVIEW and not is_numeric:
            raise RuntimeError(
                f"PatentsView backend returned non-USPTO ID: {p.patent_id!r}\n"
                "Routing bug — EPO records entering PatentsView pipeline."
            )
        if backend == SearchBackend.EPO and is_numeric:
            logger.warning(
                "EPO backend returned numeric ID %s — may be a USPTO patent via EPO.",
                p.patent_id,
            )


# Module-level singletons — constructed once, reused for every pipeline call.
# Construction is deferred to first use (lazy init) to keep import time low.
_mistral: ClaudeClient | None = None
_embedding: EmbeddingModel | None = None
_claude: ClaudeClient | None = None


def _get_mistral() -> ClaudeClient:
    global _mistral
    if _mistral is None:
        _mistral = ClaudeClient()
    return _mistral


def _get_embedding() -> EmbeddingModel:
    global _embedding
    if _embedding is None:
        _embedding = EmbeddingModel(model_name=settings.embedding_model)
    return _embedding


def _get_claude() -> ClaudeClient:
    global _claude
    if _claude is None:
        _claude = ClaudeClient()
    return _claude


def run_pipeline(
    query: str,
    *,
    mistral: ClaudeClient | None = None,
    embedding_model: EmbeddingModel | None = None,
    claude: ClaudeClient | None = None,
    top_k: int | None = None,
    save_to_db: bool = True,
    report_dir: Path | str = "data/processed",
) -> PipelineResult:
    """
    Run the full patent intelligence pipeline for *query*.

    Args:
        query:           Natural-language technology query.
        mistral:         Override the module-level MistralClient (e.g. mock).
        embedding_model: Override the module-level EmbeddingModel.
        claude:          Override the module-level ClaudeClient.
        top_k:           Maximum patents in the ranked output.
        save_to_db:      Persist unique patents to MySQL + JSON cache.
        report_dir:      Directory to save the Markdown report.

    Returns:
        A populated PipelineResult Pydantic object.
    """
    mistral_client = mistral or _get_mistral()
    embed_model = embedding_model or _get_embedding()
    claude_client = claude or _get_claude()
    k = top_k if top_k is not None else settings.top_k_results

    logger.info("Pipeline start: query='%s'", query)

    # Resolve backend ONCE — never re-check inside the pipeline
    backend = _resolve_backend()

    # ------------------------------------------------------------------
    # Stage 1 — Query expansion (constrained by domain token anchors)
    # ------------------------------------------------------------------
    expanded_queries, tokens, _ = query_expansion.expand_query(query, mistral_client)
    logger.info("Stage 1 done: %d queries (original + %d expanded)",
                len(expanded_queries), len(expanded_queries) - 1)

    # ------------------------------------------------------------------
    # Stage 1b — Constraint validation (drop any query that lost the
    #             domain anchor Mistral was given as a hard constraint)
    # ------------------------------------------------------------------
    validated_queries = constraint_validator.validate_queries(expanded_queries, tokens)
    logger.info(
        "Stage 1b done: %d/%d queries passed constraint validation",
        len(validated_queries),
        len(expanded_queries),
    )

    # ------------------------------------------------------------------
    # Stage 2 — Patent search (strictly routed, no cross-backend fallback)
    # ------------------------------------------------------------------
    raw_patents = _search(backend, tokens, validated_queries)
    logger.info("Stage 2 done: %d raw patents fetched (backend=%s)", len(raw_patents), backend.value)

    # ------------------------------------------------------------------
    # Stage 3 — Deduplication
    # ------------------------------------------------------------------
    patents = dedup_service.deduplicate(raw_patents)
    logger.info("Stage 3 done: %d unique patents after dedup", len(patents))

    # Source guard — crash loudly if wrong-backend records enter the pipeline
    _assert_source(patents, backend)

    if not patents:
        logger.warning("No patents found for query='%s'. Pipeline aborted.", query)
        return PipelineResult(
            query=query,
            expanded_queries=expanded_queries,
            ranked_patents=[],
            comparison_summary="No patents found.",
            report_markdown="# No Results\n\nNo patents were found for this query.",
        )

    # ------------------------------------------------------------------
    # Stage 4 — Embed patents + query
    # Enrich the query embedding with all validated expansions so the
    # vector is keyword-dense and aligns better with patent abstracts.
    # ------------------------------------------------------------------
    doc_vecs = embedding_service.embed_patents(patents, embed_model)
    query_vec = embedding_service.embed_query(query, embed_model, expanded=validated_queries)
    logger.info("Stage 4 done: embedded %d patents (dim=%d)", len(patents), doc_vecs.shape[1])

    # ------------------------------------------------------------------
    # Stage 5 — Hybrid ranking
    # Build a full anchor set from ALL extractor vocabulary groups:
    #   critical_tokens  — surface terms matched in the query (e.g. "camera")
    #   patent_synonyms  — patent vocabulary (e.g. "imaging system")
    # Patents missing every anchor from this combined set receive a 0.5
    # score penalty, pushing off-domain results (e.g. LADAR for a fintech
    # query) below genuinely relevant patents.
    # ------------------------------------------------------------------
    anchor_tokens: list[str] = list(
        dict.fromkeys(tokens.critical_tokens + tokens.patent_synonyms)
    )
    logger.info(
        "Stage 5 anchor set: %d tokens (%d critical + %d synonyms)",
        len(anchor_tokens),
        len(tokens.critical_tokens),
        len(tokens.patent_synonyms),
    )
    ranked: list[RankedPatent] = ranking_service.rank(
        query=query,
        patents=patents,
        doc_vecs=doc_vecs,
        query_vec=query_vec,
        top_k=k,
        critical_tokens=anchor_tokens,
        tokens=tokens,
    )
    logger.info("Stage 5 done: %d patents ranked", len(ranked))

    # ------------------------------------------------------------------
    # Stage 5b — Relevance sanity filter (Claude binary classification)
    # Catches off-domain patents that passed the embedding threshold due
    # to generic vocabulary overlap (e.g. "distributed" appearing in both
    # federated-learning and sensor-network patents).
    # ------------------------------------------------------------------
    ranked = filter_irrelevant_patents(
        query=query,
        domain_concepts=tokens.domain_concepts,
        ranked=ranked,
        client=claude_client,
    )
    logger.info("Stage 5b done: %d patents after relevance filter", len(ranked))

    # ------------------------------------------------------------------
    # Stage 6 — LLM comparison
    # ------------------------------------------------------------------
    comparison_summary = comparison_service.compare_patents(
        query, ranked, claude_client
    )
    logger.info("Stage 6 done: comparison received (%d chars)", len(comparison_summary))

    # ------------------------------------------------------------------
    # Stage 7 — Report generation
    # ------------------------------------------------------------------
    report_markdown = report_service.generate(
        query, ranked, comparison_summary, claude_client
    )
    logger.info("Stage 7 done: report generated (%d chars)", len(report_markdown))

    # ------------------------------------------------------------------
    # Stage 8 — Persist
    # ------------------------------------------------------------------
    if save_to_db:
        save_patents(patents, query)
        logger.info("Stage 8 done: %d patents persisted to MySQL + cache", len(patents))

        report_path = Path(report_dir)
        report_path.mkdir(parents=True, exist_ok=True)
        import hashlib, datetime
        slug = hashlib.sha256(query.lower().strip().encode()).hexdigest()[:12]
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = report_path / f"report_{slug}_{ts}.md"
        report_file.write_text(report_markdown, encoding="utf-8")
        logger.info("Report saved: %s", report_file)

    logger.info("Pipeline complete: query='%s'", query)

    return PipelineResult(
        query=query,
        expanded_queries=expanded_queries,
        ranked_patents=ranked,
        comparison_summary=comparison_summary,
        report_markdown=report_markdown,
    )