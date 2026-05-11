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
from services.query_builder import build_tiered_queries
from services.search_service import fetch_patents_for_query_dict, fetch_patents_with_fallback
from storage.patent_repository import save_patents

logger = logging.getLogger(__name__)

# Minimum number of patents required from the broad tier before we consider
# the search adequately seeded.  Below this threshold the pipeline still
# continues — the narrow tiers may still contribute results.
_MIN_RESULTS_THRESHOLD: int = 10


def _execute_tiered_search(
    original: str,
    expanded: list[str],
    tokens,
    min_threshold: int = _MIN_RESULTS_THRESHOLD,
) -> list:
    """
    Execute a broad → medium → narrow tiered patent search (PatentsView backend).

    For EPO backend, delegates to epo_search_service which has its own
    3-stage fallback.  For PatentsView, uses fetch_patents_with_fallback which
    guarantees at least min_threshold results by progressively broadening scope.

    Args:
        original:      The raw user query string.
        expanded:      Validated Mistral-expanded query strings.
        tokens:        ExtractedTokens from the deterministic token extractor.
        min_threshold: Minimum result target for fallback stages.

    Returns:
        Combined list of PatentRecord objects from all tiers (pre-dedup).
    """
    backend = settings.search_backend.lower()

    # --- EPO backend ---
    if backend == "epo" and settings.epo_consumer_key:
        from services.epo_search_service import fetch_all_patents as epo_fetch
        logger.info("Search backend: EPO OPS")
        return epo_fetch(expanded)

    # --- PatentsView / other backends: tiered + fallback ---
    logger.info("Search backend: %s (tiered + fallback)", backend)

    # Primary: tiered broad→medium→narrow queries
    tiers = build_tiered_queries(original, expanded, tokens)
    all_patents: list = []

    for tier in tiers:
        tier_results: list = []
        for query_dict in tier.queries:
            results = fetch_patents_for_query_dict(query_dict)
            tier_results.extend(results)

        logger.info(
            "Tier '%s' returned %d patents (%s)",
            tier.strategy,
            len(tier_results),
            tier.description,
        )
        all_patents.extend(tier_results)

        if tier.strategy == "narrow" and len(tier_results) == 0:
            logger.warning("Narrow tier returned 0 results.")

    # If tiered search is thin, apply 3-stage fallback to top up results
    if len(all_patents) < min_threshold:
        logger.warning(
            "Tiered search returned only %d patents (threshold=%d) — "
            "activating 3-stage keyword fallback.",
            len(all_patents), min_threshold,
        )
        fallback = fetch_patents_with_fallback(tokens, expanded, min_results=min_threshold)
        seen = {p.patent_id for p in all_patents}
        for p in fallback:
            if p.patent_id not in seen:
                all_patents.append(p)
                seen.add(p.patent_id)
        logger.info("After fallback: %d patents total", len(all_patents))

    return all_patents


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
    # Stage 2 — Tiered patent search (broad → medium → narrow)
    # ------------------------------------------------------------------
    raw_patents = _execute_tiered_search(query, validated_queries, tokens)
    logger.info("Stage 2 done: %d raw patents fetched (tiered)", len(raw_patents))

    # ------------------------------------------------------------------
    # Stage 3 — Deduplication
    # ------------------------------------------------------------------
    patents = dedup_service.deduplicate(raw_patents)
    logger.info("Stage 3 done: %d unique patents after dedup", len(patents))

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