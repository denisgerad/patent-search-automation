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
from models.mistral_client import MistralClient
from models.schemas import PipelineResult, RankedPatent
from services import (
    comparison_service,
    dedup_service,
    query_expansion,
    report_service,
)
from services import embedding_service, ranking_service
from services.search_service import fetch_all_patents
from storage.patent_repository import save_patents

logger = logging.getLogger(__name__)

# Module-level singletons — constructed once, reused for every pipeline call.
# Construction is deferred to first use (lazy init) to keep import time low.
_mistral: MistralClient | None = None
_embedding: EmbeddingModel | None = None
_claude: ClaudeClient | None = None


def _get_mistral() -> MistralClient:
    global _mistral
    if _mistral is None:
        _mistral = MistralClient(model=settings.mistral_model)
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
    mistral: MistralClient | None = None,
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
    # Stage 1 — Query expansion
    # ------------------------------------------------------------------
    expanded_queries = query_expansion.expand_query(query, mistral_client)
    logger.info("Stage 1 done: %d queries (original + %d expanded)",
                len(expanded_queries), len(expanded_queries) - 1)

    # ------------------------------------------------------------------
    # Stage 2 — Patent search
    # ------------------------------------------------------------------
    raw_patents = fetch_all_patents(expanded_queries)
    logger.info("Stage 2 done: %d raw patents fetched", len(raw_patents))

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
    # ------------------------------------------------------------------
    doc_vecs = embedding_service.embed_patents(patents, embed_model)
    query_vec = embedding_service.embed_query(query, embed_model)
    logger.info("Stage 4 done: embedded %d patents (dim=%d)", len(patents), doc_vecs.shape[1])

    # ------------------------------------------------------------------
    # Stage 5 — Hybrid ranking
    # ------------------------------------------------------------------
    ranked: list[RankedPatent] = ranking_service.rank(
        query=query,
        patents=patents,
        doc_vecs=doc_vecs,
        query_vec=query_vec,
        top_k=k,
    )
    logger.info("Stage 5 done: %d patents ranked", len(ranked))

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