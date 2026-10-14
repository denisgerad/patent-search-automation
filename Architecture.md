# Patent Intelligence System -- Architecture & Code Flow

## Current Implementation Notes — 2026-10-12

> This section records behaviour verified against the source excerpts reviewed
> on 2026-10-12. Where older sections conflict with this section, the discrepancy
> must be checked against the current source before the older description is relied on.

### Application entry paths

The repository contains two distinct application paths:

1. **Streamlit search workflow** — `streamlit_app.py` contains separate query
   expansion, search and deduplication workflows. The inspected code references
   `_run_expand`, `_run_search`, `_run_search_json` and
   `search_patents_by_strategy`. Do not assume every Streamlit search invokes
   `orchestrator.patent_pipeline.run_pipeline`.
2. **Orchestrated pipeline** — `orchestrator/patent_pipeline.py` exposes
   `run_pipeline(query, ...)` and returns a `PipelineResult`.

The exact relationship between all Streamlit search modes and the orchestrated
pipeline requires further source verification.

### Verified orchestrated pipeline

For a normal query that returns patents, the observed stage order is:

1. Resolve the configured search backend.
2. Extract critical tokens and expand the query.
3. Validate expanded queries against domain constraints.
4. Retrieve patents through the selected backend.
5. Deduplicate patent records.
6. Check records against source/backend expectations.
7. Embed patent text and the query.
8. Rank candidates using the ranking service and anchor tokens.
9. Apply the Claude relevance filter.
10. Generate a structured patent comparison.
11. Generate a Markdown report.
12. If `save_to_db=True`, persist patents and write a timestamped Markdown
    report under the configured report directory.

If no patents are returned, the pipeline returns a no-results `PipelineResult`
before embedding, ranking, comparison, or persistence.

### Search backend selection

The orchestrator's `_resolve_backend()` supports `epo`, `uspto`,
`patentsview` and `auto`. In `auto` mode, it selects EPO when an EPO consumer
key is configured; otherwise it selects PatentsView when its API key is
configured. If neither key is available, it raises an error.

The orchestrator does not cross-fallback between EPO and PatentsView during
search. Other provider modules in the repository do not, by their presence
alone, establish that those providers are used by this orchestrated path.

### AI client wiring

`services/query_expansion.py` imports and calls `models.claude_client.ClaudeClient`.
The orchestrator's `_get_mistral()` also constructs `ClaudeClient`, despite
the variable and parameter names containing `mistral`. Therefore, query
expansion in this path is Claude-powered in the inspected implementation.

A separate `models/mistral_client.py` implements local inference through Ollama,
but its existence does not establish that it is wired into this pipeline.

Claude is also passed to the relevance filter, comparison service and report
service. The embedding model is constructed using `settings.embedding_model`.

### Ranking and output contracts

`models/schemas.py` defines `PatentRecord`, `RankedPatent` and `PipelineResult`.
A ranked record includes cosine, BM25 and hybrid scores, coverage, concept-hit
information, importance-related fields and text-availability metadata.

The current configuration defaults include the BGE-large-en-v1.5 embedding
model, cosine weight 0.7, BM25 weight 0.3 and top-k value 10. These are
configuration defaults, not evidence that every search mode uses identical
ranking settings.

`PipelineResult` contains the original query, expanded queries, ranked patents,
comparison summary and Markdown report.

### Persistence

When persistence is enabled, the orchestrator calls `save_patents()`, whose
repository documentation describes MySQL storage and a JSON cache. It also
writes the generated Markdown report to the configured report directory.
When `save_to_db=False`, this persistence block is skipped.

### Documentation and verification gaps

- `README.md` and `tests/test_pipeline.py` were empty in the inspected working
  tree on 2026-10-12.
- The existing architecture sections contain older descriptions that may not
  match current code, including the query-expansion client and search-flow
  descriptions.
- The latest reported test-suite result was 41 passing tests. This verifies the
  tests that were run; it does not by itself validate live-provider search
  quality or establish real-world patent relevance.
- The contractor-matching search example is a candidate real-world search-quality
  test case. Its returned result and scores should be evaluated in the search
  quality workstream, not treated as proof of patent relevance or patentability.

---


---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Project Structure](#2-project-structure)
3. [High-Level Architecture](#3-high-level-architecture)
4. [Pipeline Code Flow](#4-pipeline-code-flow)
5. [Module Reference](#5-module-reference)
6. [Data Contracts](#6-data-contracts)
7. [Search Backend Routing](#7-search-backend-routing)
8. [Ranking & Scoring](#8-ranking--scoring)
9. [Configuration Reference](#9-configuration-reference)
10. [Key Design Decisions](#10-key-design-decisions)

---

## 1. System Overview

The Patent Intelligence System supports patent prior-art discovery from a natural-language invention description. It retrieves candidate patent records, deduplicates them, ranks them using semantic and keyword signals, and can generate an evidence-oriented comparison and Markdown report.

### Current technology and implementation

| Layer | Current implementation |
|---|---|
| User interface | Streamlit application (`streamlit_app.py`) |
| Orchestration | `orchestrator/patent_pipeline.py::run_pipeline` |
| Search | Configurable EPO, USPTO and PatentsView backend paths |
| Query expansion | Claude client in the inspected orchestrated pipeline |
| Embeddings | Sentence Transformers with `BAAI/bge-large-en-v1.5` by default |
| Keyword ranking | BM25 |
| Ranking | Cosine filtering, weighted hybrid scoring, coverage/anchor adjustment, importance-aware scoring and continuity-family deduplication |
| Comparison and report | Claude-backed services when configured |
| Persistence | MySQL repository and JSON cache/report output through the persistence path |
| Tests | Pytest regression and acceptance tests |

The Streamlit UI also has search workflows that call search services directly. Do not assume every UI action goes through `run_pipeline()`.

The existence of `models/mistral_client.py` does not establish that Mistral is used by the orchestrated query-expansion path. Likewise, do not describe FastAPI, SQLite or a FAISS search corpus as active components without separate evidence from the current code.

## 2. Project Structure

```
patent-intelligence/
│
├── app/
│   ├── main.py                  # CLI entry point
│   ├── api.py                   # FastAPI routes (future SaaS)
│   └── config.py                # Pydantic Settings -- single source of truth
│
├── orchestrator/
│   └── patent_pipeline.py       # Top-level pipeline -- calls services in order
│
├── services/
│   ├── query_expansion.py       # Mistral-powered query expansion
│   ├── token_extractor.py       # Deterministic critical token extraction
│   ├── constraint_validator.py  # Filter Mistral drift from expanded queries
│   ├── search_service.py        # PatentsView fetch + pagination
│   ├── epo_search_service.py    # EPO OPS search -- tiered CQL strategy
│   ├── dedup_service.py         # Patent deduplication by patent_id
│   ├── embedding_service.py     # Build patent text + call embedding model
│   ├── ranking_service.py       # Hybrid score + coverage penalty
│   ├── relevance_filter.py      # Claude binary relevance pre-filter
│   ├── comparison_service.py    # Claude structured patent comparison
│   └── report_service.py        # Claude Markdown report generation
│
├── models/
│   ├── mistral_client.py        # Ollama/Mistral API wrapper (generate only)
│   ├── claude_client.py         # Anthropic SDK wrapper (complete only)
│   ├── embedding_model.py       # BGE-large wrapper (embed_query / embed_documents)
│   ├── epo_client.py            # EPO OPS HTTP + auth token lifecycle
│   └── schemas.py               # Pydantic data contracts
│
├── retrieval/
│   ├── bm25_ranker.py           # BM25 index build + score
│   ├── vector_store.py          # Embedding cache (NOT the search corpus)
│   └── similarity.py            # Cosine similarity + dynamic threshold +
│                                #   token coverage + coverage penalty
│
├── prompts/
│   ├── query_expansion.txt      # Mistral expansion prompt template
│   ├── comparison_system.txt    # Claude comparison system prompt
│   └── report_system.txt        # Claude report generation prompt
│
├── data/
│   ├── raw/                     # Raw API responses (JSON, keyed by query hash)
│   ├── processed/               # Cleaned PatentRecord objects
│   └── vectors/                 # FAISS index (optional persistence)
│
├── storage/
│   ├── database.py              # SQLAlchemy connection
│   ├── patent_repository.py     # CRUD -- save/load PatentRecord
│   └── schema.sql               # Table definitions
│
├── utils/
│   ├── logger.py                # Structured JSON logger
│   ├── helpers.py               # Shared utilities
│   └── text_cleaning.py         # Strip stop words, normalise text
│
└── tests/
    ├── fixtures/                # Recorded API responses for offline testing
    ├── test_search.py
    ├── test_ranking.py
    ├── test_token_extractor.py
    └── test_pipeline.py
```

---

## 3. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          USER INPUT                                      │
│   "A system for autonomous vehicle lane detection using infrared sensors" │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     STAGE 1 -- QUERY INTELLIGENCE                        │
│                                                                          │
│  ┌─────────────────────────┐    ┌──────────────────────────────────┐    │
│  │   token_extractor.py    │    │      query_expansion.py          │    │
│  │                         │    │                                  │    │
│  │  DOMAIN_TAXONOMY lookup │    │  Mistral 7B receives:            │    │
│  │  → primary_anchor       │    │  - original query                │    │
│  │  → concept_groups       │    │  - critical_tokens (mandatory)   │    │
│  │  → domain_concepts      │    │  - patent_synonyms               │    │
│  │  → patent_synonyms      │───▶│  → 5 structural variants         │    │
│  │                         │    │    (apparatus/method/compound/   │    │
│  │  Ranked by:             │    │     claim-style/keyword-only)    │    │
│  │  task(1) > domain(3)    │    │                                  │    │
│  │  > sensor(4)            │    └──────────────┬───────────────────┘    │
│  └─────────────────────────┘                   │                        │
│                                                 ▼                        │
│                              ┌──────────────────────────────────┐       │
│                              │    constraint_validator.py        │       │
│                              │                                  │       │
│                              │  Drop any expanded query that    │       │
│                              │  lost domain anchor tokens       │       │
│                              │  → validated_queries[]           │       │
│                              └──────────────────────────────────┘       │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     STAGE 2 -- PATENT RETRIEVAL                          │
│                                                                          │
│   _resolve_backend() ── reads config once ── never changes mid-pipeline │
│                                                                          │
│   ┌──────────────────────────┐    ┌──────────────────────────────────┐  │
│   │   EPO OPS Backend        │    │   PatentsView Backend            │  │
│   │                          │    │                                  │  │
│   │  CQL Tier 1 (narrow):    │    │  Stage 1: phrase queries         │  │
│   │  anchor AND domain       │    │  Stage 2: _text_any per token    │  │
│   │       ↓ (if < 10 hits)   │    │  Stage 3: domain concept only    │  │
│   │  CQL Tier 2 (medium):    │    │                                  │  │
│   │  anchor only             │    │  Returns: US numeric patent IDs  │  │
│   │       ↓ (if < 10 hits)   │    │  Fields: id, title, abstract,    │  │
│   │  CQL Tier 3 (broad):     │    │           type, date             │  │
│   │  anchor synonyms         │    │                                  │  │
│   │                          │    └──────────────────────────────────┘  │
│   │  Per ID: fetch           │                                          │
│   │  /biblio,claims          │    ← Backends are mutually exclusive.    │
│   │  → title + abstract      │      No cross-fallback between them.     │
│   │    + claims (optional)   │                                          │
│   └──────────────────────────┘                                          │
│                                                                          │
│   Both paths → dedup_service.py → list[PatentRecord]                   │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     STAGE 3 -- EMBEDDING & RANKING                       │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │  embedding_service.build_patent_text(patent)                   │     │
│  │                                                                 │     │
│  │  "{title}. {title}. {title}. {claims}. {claims}. {abstract}"  │     │
│  │        3x weight         2x weight          1x weight          │     │
│  └────────────────────────────────────────────────────────────────┘     │
│                          │                                               │
│          ┌───────────────┴──────────────────┐                           │
│          ▼                                   ▼                           │
│  ┌───────────────────┐           ┌───────────────────────────┐          │
│  │  embedding_model  │           │      bm25_ranker.py       │          │
│  │                   │           │                           │          │
│  │  embed_query():   │           │  build_index(patents)     │          │
│  │  BGE prefix +     │           │  → BM25Okapi on           │          │
│  │  query string     │           │    title+abstract tokens  │          │
│  │  → (1024,) vec    │           │                           │          │
│  │                   │           │  get_scores(query)        │          │
│  │  embed_documents():│           │  → score per patent       │          │
│  │  NO prefix        │           │    (keyword frequency)    │          │
│  │  batched 32       │           └───────────────────────────┘          │
│  │  → (n, 1024) mat  │                                                  │
│  └────────────────┬──┘                                                  │
│                   │                                                      │
│                   ▼                                                      │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │  similarity.py                                                  │     │
│  │                                                                 │     │
│  │  cosine  = doc_vecs @ query_vec   (L2-normalized dot product)  │     │
│  │  bm25    = BM25Okapi.get_scores() (normalised to 0-1)          │     │
│  │  hybrid  = 0.6 * cosine + 0.4 * bm25                          │     │
│  │                                                                 │     │
│  │  coverage = matched_concepts / total_concepts                  │     │
│  │  penalised = hybrid * coverage^2  (quadratic curve)            │     │
│  │                                                                 │     │
│  │  threshold = compute_dynamic_threshold(scores, "elbow")        │     │
│  └────────────────────────────────────────────────────────────────┘     │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     STAGE 4 -- LLM ANALYSIS                              │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │  relevance_filter.py  (Claude binary pre-filter)             │       │
│  │  Input : top 20 ranked patents (title only, cheap)           │       │
│  │  Output: {patent_id, relevant: bool, reason: str}[]          │       │
│  │  Effect: drops off-domain patents before expensive analysis   │       │
│  └──────────────────────────┬───────────────────────────────────┘       │
│                             │                                            │
│                             ▼                                            │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │  comparison_service.py  (Claude structured analysis)         │       │
│  │  Input : top 10 filtered patents (title + abstract + claims) │       │
│  │  Prompt: comparison_system.txt                               │       │
│  │  Output: JSON[] {patent_id, overlap_level, key_claims,       │       │
│  │                   differentiation_notes}                      │       │
│  └──────────────────────────┬───────────────────────────────────┘       │
│                             │                                            │
│                             ▼                                            │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │  report_service.py  (Claude Markdown report)                 │       │
│  │  Input : comparison JSON + ranked patent list                │       │
│  │  Prompt: report_system.txt                                   │       │
│  │  Output: Markdown with sections:                             │       │
│  │    1. Executive Summary                                      │       │
│  │    2. Detailed Claim Analysis                                │       │
│  │    3. Risk Assessment (HIGH / MEDIUM / LOW)                  │       │
│  │    4. Recommended Actions                                    │       │
│  └──────────────────────────────────────────────────────────────┘       │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │     PipelineResult    │
                    │                       │
                    │  query               │
                    │  expanded_queries[]  │
                    │  ranked_patents[]    │
                    │  comparison_json     │
                    │  report_markdown     │
                    └───────────────────────┘
```

---

## 4. Pipeline Code Flow

### Entry Point

```
app/main.py
└── run_pipeline(query)                         orchestrator/patent_pipeline.py
```

### Stage 1 -- Query Intelligence

```
run_pipeline(query)
│
├── _resolve_backend()
│   └── reads settings.epo_consumer_key / patentsview_api_key
│   └── returns SearchBackend.EPO | SearchBackend.PATENTSVIEW
│   └── raises RuntimeError if neither key is set
│
├── token_extractor.extract_critical_tokens(query)
│   ├── query.lower() → match against DOMAIN_TAXONOMY
│   ├── sort matched concepts by CONCEPT_DISCRIMINATING_POWER
│   │     task(1) > application(2) > domain(3) > sensor(4)
│   ├── primary_anchor = most discriminating matched term
│   ├── concept_groups = {concept: {terms, synonyms, matched}}
│   └── returns ExtractedTokens
│       ├── critical_tokens  : flat list of matched terms
│       ├── domain_concepts  : ordered list of concept names
│       ├── patent_synonyms  : flat list of all synonyms
│       ├── concept_groups   : grouped by concept (for coverage scoring)
│       ├── primary_anchor   : single best anchor string
│       ├── primary_concept  : concept name of primary anchor
│       └── taxonomy_miss    : True if no taxonomy match found
│
├── query_expansion.expand_query(query, mistral, tokens)
│   ├── loads prompts/query_expansion.txt
│   ├── injects: original_query, critical_tokens, patent_synonyms
│   ├── Mistral generates 5 structural variants:
│   │     1. APPARATUS  -- "System/apparatus for [function] using [tech]"
│   │     2. METHOD     -- "Method comprising [tech] steps"
│   │     3. COMPOUND   -- "[Tech]-based [function] in [domain]"
│   │     4. CLAIM-STYLE-- "...system wherein..."
│   │     5. KEYWORD    -- 3-5 keywords only, no filler
│   ├── parse JSON response → list[str]
│   └── returns ([original] + expanded, tokens)
│
└── constraint_validator.validate_queries(queries, tokens)
    ├── for each query: check if any anchor/synonym present
    ├── drop queries where no anchor found (Mistral drift)
    └── safety: always retain original query
```

### Stage 2 -- Patent Retrieval

```
_search(backend, tokens, validated_queries)
│
├── [EPO path] _search_epo(tokens, validated_queries)
│   │
│   └── epo_search_service.epo_fetch_by_keywords(tokens)
│       │
│       ├── _build_cql_tiered(tokens)
│       │   ├── Tier 1 (narrow):  "anchor" AND ("domain_term" OR "synonym")
│       │   ├── Tier 2 (medium):  ti="anchor" OR ab="anchor"
│       │   └── Tier 3 (broad):   ti=(synonyms) OR ab=(synonyms)
│       │
│       ├── for each CQL tier:
│       │   └── epo_client.search(cql, max_results=25)
│       │       ├── GET /published-data/search/biblio?q={cql}
│       │       ├── parse XML → exchange:exchange-document elements
│       │       └── returns ["US.12345.A1", "EP.98765.A1", ...]
│       │
│       ├── stop when accumulated IDs >= MIN_RESULTS (10)
│       │
│       └── for each patent_id:
│           └── epo_client.fetch_biblio(id)         [or fetch_biblio_and_claims]
│               ├── GET /published-data/publication/epodoc/{id}/biblio,claims
│               ├── parse title (prefer lang="en")
│               ├── parse abstract paragraphs
│               ├── parse independent claims
│               └── returns {patent_id, title, abstract, claims}
│
├── [PatentsView path] _search_patentsview(tokens, validated_queries)
│   │
│   └── search_service.fetch_patents_with_fallback(tokens, validated_queries)
│       ├── Stage 1: fetch_all_patents(validated_queries)
│       │   └── _fetch_all_for_query(_build_term_query(term))
│       │       ├── _build_term_query: _text_phrase bigrams for long queries
│       │       ├── POST https://search.patentsview.org/api/v1/patent/
│       │       ├── cursor-based pagination (after = last patent_id)
│       │       └── stops at MAX_PAGES or MAX_RESULTS
│       │
│       ├── Stage 2 (if < 15 results): _text_any per critical_token
│       └── Stage 3 (if < 15 results): _text_any on domain concept term
│
└── dedup_service.deduplicate(raw_patents)
    └── {p.patent_id: p for p in patents}.values()
        → list[PatentRecord]
```

### Stage 3 -- Embedding & Ranking

```
embed + rank
│
├── embedding_service.embed_patents(patents, model)
│   ├── build_patent_text(patent) per patent
│   │   └── "{title}. {title}. {title}. {claims}. {claims}. {abstract}"
│   │         3x title weight    2x claims weight   1x abstract
│   └── model.embed_documents(texts)
│       └── SentenceTransformer.encode(normalize_embeddings=True, batch_size=32)
│           → np.ndarray shape (n_patents, 1024)
│
├── embedding_model.embed_query(query)
│   └── prefix = "Represent this sentence for searching relevant passages: "
│       SentenceTransformer.encode(prefix + query, normalize_embeddings=True)
│       → np.ndarray shape (1024,)
│
├── similarity.cosine_similarity(query_vec, doc_vecs)
│   └── doc_vecs @ query_vec       (dot product of L2-normalised vecs)
│       → np.ndarray shape (n_patents,)   range: 0.0-1.0
│
├── bm25_ranker.build_index(patents)
│   └── BM25Okapi([text.lower().split() for text in patent_texts])
│
├── bm25_ranker.get_scores(index, query)
│   └── index.get_scores(query.split())
│       → list[float]   (raw BM25, unbounded)
│
├── ranking_service.hybrid_rank(patents, cosine, bm25, tokens)
│   ├── _normalize(cosine)     → 0-1
│   ├── _normalize(bm25)       → 0-1
│   ├── raw_hybrid = 0.6 * cosine_norm + 0.4 * bm25_norm
│   │
│   ├── per patent:
│   │   ├── compute_token_coverage(patent, tokens)
│   │   │   ├── for each concept in tokens.concept_groups:
│   │   │   │   check if any term/synonym present in title+abstract
│   │   │   ├── coverage = matched_concepts / total_concepts
│   │   │   └── returns {coverage, matched_concepts, concept_hits}
│   │   │
│   │   └── apply_coverage_penalty(raw_hybrid, coverage, "quadratic")
│   │       └── penalised = raw_hybrid * coverage^2
│   │           coverage=1.0 -> no penalty (*1.00)
│   │           coverage=0.67-> moderate  (*0.45)
│   │           coverage=0.33-> heavy     (*0.11)
│   │
│   └── sort by penalised hybrid_score descending
│       → list[RankedPatent]
│
└── similarity.compute_dynamic_threshold(hybrid_scores, "elbow")
    ├── sort scores descending
    ├── find sharpest drop in consecutive scores
    └── threshold = score at elbow point
        → filter_by_threshold(scores, threshold)
```

### Stage 4 -- LLM Analysis

```
LLM analysis
│
├── relevance_filter.filter_irrelevant_patents(query, domain_concepts, ranked, claude)
│   ├── candidates = ranked[:20]
│   ├── format: "- ID: {id} | Title: {title}" per patent
│   ├── Claude prompt: classify each as relevant=true/false
│   │   (binary, strict -- based on domain concepts)
│   ├── parse JSON response → relevant_ids set
│   └── returns filtered list[RankedPatent]
│
├── comparison_service.compare_patents(query, filtered_patents, claude)
│   ├── candidates = filtered[:10]
│   ├── format: title + abstract + claims per patent
│   ├── system prompt: prompts/comparison_system.txt
│   ├── Claude returns JSON[]:
│   │   {patent_id, overlap_level, key_overlapping_claims,
│   │    differentiation_notes, risk_level}
│   └── returns comparison_json: str
│
└── report_service.generate(query, ranked, comparison_json, claude)
    ├── system prompt: prompts/report_system.txt
    ├── Claude generates Markdown report:
    │   ## Executive Summary
    │   ## Detailed Claim Analysis
    │   ## Risk Assessment
    │   ## Recommended Actions
    └── saves to data/processed/{query_hash}_report.md
        returns report_markdown: str
```

---

## 5. Module Reference

### `models/` -- Wrappers only, zero business logic

| File | Responsibility | Key method |
|---|---|---|
| `mistral_client.py` | Ollama/Mistral API call | `generate(prompt) → str` |
| `claude_client.py` | Anthropic SDK call | `complete(system, user, max_tokens) → str` |
| `embedding_model.py` | BGE-large-en-v1.5 | `embed_query(str) → (1024,)` / `embed_documents(list) → (n,1024)` |
| `epo_client.py` | EPO OPS HTTP + token cache | `search(cql) → list[str]` / `fetch_biblio(id) → dict` |
| `schemas.py` | Pydantic data contracts | `PatentRecord`, `RankedPatent`, `ExtractedTokens`, `PipelineResult` |

### `services/` -- All business logic

| File | Input | Output |
|---|---|---|
| `token_extractor.py` | raw query string | `ExtractedTokens` |
| `query_expansion.py` | query + `ExtractedTokens` | `list[str]` expanded queries |
| `constraint_validator.py` | expanded queries + `ExtractedTokens` | validated `list[str]` |
| `search_service.py` | validated queries + tokens | `list[PatentRecord]` (PatentsView) |
| `epo_search_service.py` | `ExtractedTokens` | `list[PatentRecord]` (EPO) |
| `dedup_service.py` | `list[PatentRecord]` | `list[PatentRecord]` (unique) |
| `embedding_service.py` | `list[PatentRecord]` | `np.ndarray (n, 1024)` |
| `ranking_service.py` | patents + scores + tokens | `list[RankedPatent]` |
| `relevance_filter.py` | ranked patents + claude | `list[RankedPatent]` (filtered) |
| `comparison_service.py` | filtered patents + claude | comparison JSON string |
| `report_service.py` | comparison JSON + claude | Markdown string |

### `retrieval/` -- Infrastructure, no business logic

| File | Responsibility |
|---|---|
| `similarity.py` | Cosine similarity, dynamic threshold, token coverage, coverage penalty |
| `bm25_ranker.py` | BM25 index build, score retrieval |
| `vector_store.py` | Embedding cache by patent_id -- **not** the search corpus |

---

## 6. Data Contracts

The current Pydantic schemas in `models/schemas.py` are the source of truth for these contracts.

### `PatentRecord`

A retrieved patent record contains:

- `patent_id` and `patent_title`
- Optional `patent_abstract`, `patent_claims`, `patent_type` and `patent_date`
- Optional `continuity_family_id`
- `search_paths`
- Optional USPC fields and CPC/IPC classification lists

### `SearchStrategy`

The approved search strategy contains the original input, key inventive points, and structured `SearchConcept` objects. Each concept has a name, importance, terms and an operator. The schema also defines proximity rules and related strategy fields; consult `models/schemas.py` for the complete contract.

### `RankedPatent`

Each ranked record includes the underlying `PatentRecord`, cosine, BM25 and hybrid scores, coverage, concept hits, search paths, importance score/hits, and text-availability status/character count.

`importance_score` may be `None` when the patent text is insufficient to support the importance calculation. Coverage and importance values should therefore be interpreted alongside text availability.

### `PipelineResult`

The current pipeline result contains:

- `query`
- `expanded_queries`
- `ranked_patents`
- `comparison_summary`
- `report_markdown`

Do not document older fields such as `comparison_json`, `validated_queries`, `backend` or `primary_anchor` as `PipelineResult` fields unless the schema is changed and verified.

## 7. Search Backend Routing

The orchestrator resolves a backend before searching and then dispatches to the corresponding search path.

The supported configuration values are `auto`, `epo`, `uspto` and `patentsview`. The explicit `uspto` path uses `fetch_patents_with_fallback`; the `patentsview` path uses the PatentsView search implementation; and `epo` uses the EPO search implementation.

In `auto` mode, the inspected resolver selects EPO when the EPO consumer key is configured; otherwise it selects the PatentsView path when the PatentsView API key is configured. If neither credential is available, it raises an error. Backend resolution does not mean the pipeline automatically retries a failed EPO search against a different backend.

The orchestrated pipeline also validates source/identifier consistency. Do not describe all backend paths as interchangeable: their retrieval methods and expected identifier formats differ.

The Streamlit application's direct search workflows are separate entry paths and should be documented independently from `run_pipeline()`.

## 8. Ranking & Scoring

The active ranking implementation is `services/ranking_service.py::rank`. Its current behaviour is:

1. Compute cosine similarity between the query embedding and candidate document embeddings.
2. Apply a cosine threshold, starting from `settings.cosine_threshold` and relaxing in steps of 0.05 down to `settings.cosine_threshold_min` if too few candidates pass.
3. Build a BM25 index and calculate BM25 scores for the cosine-filtered candidate set.
4. Combine raw cosine similarity with batch-min-max-normalised BM25:

   `hybrid = cosine_weight * cosine + bm25_weight * normalised_BM25`

   The default weights are 0.7 cosine and 0.3 BM25. Configuration normalises the weights when their positive sum differs from 1.
5. If extracted token concept groups are available, calculate concept coverage and apply a **linear** coverage multiplier in the active path. Otherwise, apply the binary anchor penalty to candidates that contain none of the supplied anchor terms.
6. If a search strategy is provided and sufficient text is available, calculate importance-weighted concept coverage. Critical concepts have a greater weight than important and supporting concepts. Missing critical concepts incur an additional multiplier. When text is insufficient, this adjustment is skipped.
7. Calculate the elbow threshold as a quality signal. If fewer than the requested number of candidates pass, fall back to the highest-scoring candidates from the cosine-filtered set.
8. Sort by adjusted hybrid score and return up to `top_k` distinct continuity families. If family deduplication leaves too few results, consider remaining candidates below the dynamic threshold, in score order.

### Important interpretation notes

- BM25 is normalised over the cosine-filtered candidate batch, so its normalised values depend on that batch.
- Cosine scores are used directly rather than min-max-normalised across the batch.
- The active coverage call uses `penalty_curve="linear"`, even though the helper supports other curves.
- `retrieval/similarity.py::compute_dynamic_threshold` supports elbow, mean-plus-standard-deviation and percentile strategies, but the active ranking path calls it with `strategy="elbow"`.
- The dynamic threshold does not necessarily reduce the final output count: fallback logic attempts to fill the requested `top_k` from available candidates.
- Family deduplication applies when a record has a `continuity_family_id`; records without one are treated as unique.
- Ranking scores are candidate-ordering signals, not proof of novelty, patentability or legal relevance.

## 9. Configuration Reference

The current settings are defined in `app/config.py`. The following defaults are relevant to the inspected pipeline:

| Setting | Default or behaviour |
|---|---|
| `patent_search_backend` | `auto` |
| `anthropic_model` | `claude-sonnet-4-6` |
| `mistral_model` | `mistral` (configured, but not evidence that the orchestrated path uses it) |
| `embedding_model` | `BAAI/bge-large-en-v1.5` |
| `cosine_weight` | `0.7` |
| `bm25_weight` | `0.3` |
| `top_k_results` | `10` |
| `claude_top_k` | `3` |
| `cosine_threshold` | `0.70` |
| `cosine_threshold_min` | `0.30` |
| `cosine_min_candidates` | `5` |
| `hybrid_threshold` | `0.65` |
| Database | MySQL; defaults to `localhost`, user `root`, database `patent_db` |

Environment values can override settings where supported. API credentials are optional at configuration level, but individual backends or Claude-backed features may require their corresponding credentials. Check the selected workflow before assuming that every configured setting is used by every entry path.

`PATENT_PAGE_SIZE` is a central operational constant; it is not the same setting as `top_k_results`.

## 10. Key Design Decisions

### Separation of concerns

The code separates model/API wrappers (`models/`), application services (`services/`), retrieval utilities (`retrieval/`) and the orchestrator (`orchestrator/patent_pipeline.py`). The Streamlit application also contains direct search paths, so not all user workflows are necessarily routed through the orchestrator.

### Vector storage and persistence

Describe a vector store as a cache only if the relevant implementation confirms that role. The inspected pipeline embeds the current retrieved candidate set for ranking; persistent patent/report saving occurs through the pipeline's persistence path when `save_to_db=True`. Do not describe SQLite or a FAISS corpus as the active database/search corpus without further implementation evidence.

### Search strategy and anchors

The ranking service can use extracted token concept groups for coverage adjustment and uses anchor terms as a fallback when those groups are unavailable. When an approved `SearchStrategy` is supplied, concept importance can further adjust ranking. These mechanisms are distinct and should not be conflated into a single quadratic coverage formula.

### Embeddings and ranking

The default embedding model is BGE-large-en-v1.5. The embedding service is responsible for embedding query and patent text. Ranking uses cosine similarity and BM25 with configurable weights; the exact score adjustments are described in Section 8.

### Backend failure behaviour

The orchestrated pipeline resolves one backend for a run and dispatches to that backend's implementation. Do not claim cross-backend failover unless a specific search function implements and tests it.
