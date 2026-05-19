# Patent Intelligence System -- Architecture & Code Flow

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

The Patent Intelligence System automates the manual process of patent prior-art search. A user submits a natural-language invention description; the system returns a ranked, domain-validated list of relevant patents with a Claude-generated comparison report.

**Technology Stack**

| Layer | Technology |
|---|---|
| Query expansion | Mistral 7B (via Ollama or Mistral API) |
| Patent search | EPO OPS 3.2 API / PatentsView API |
| Embeddings | BGE-large-en-v1.5 (sentence-transformers) |
| Keyword ranking | BM25 (rank-bm25) |
| Comparison & report | Claude 3.5 Sonnet (Anthropic API) |
| API layer | FastAPI |
| Storage | SQLite (SQLAlchemy Core) + FAISS vector cache |

---

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

### `ExtractedTokens`

```python
@dataclass
class ExtractedTokens:
    critical_tokens:  list[str]    # flat -- ["infrared", "lane detection"]
    domain_concepts:  list[str]    # ordered by discriminating power
    patent_synonyms:  list[str]    # flat -- used in CQL + validation
    original_query:   str
    concept_groups:   dict         # {concept: {type, terms, synonyms, matched}}
    primary_anchor:   str          # single best anchor -- "lane detection"
    primary_concept:  str          # concept name -- "lane_detection"
    taxonomy_miss:    bool = False  # True if no taxonomy match
```

### `PatentRecord`

```python
class PatentRecord(BaseModel):
    patent_id:       str
    patent_title:    str
    patent_abstract: Optional[str] = None
    patent_claims:   Optional[str] = None   # from /biblio,claims constituent
    patent_type:     Optional[str] = None
    patent_date:     Optional[str] = None
```

### `RankedPatent`

```python
class RankedPatent(BaseModel):
    patent:          PatentRecord
    cosine_score:    float         # raw cosine (0.0-1.0)
    bm25_score:      float         # raw BM25 (unbounded, pre-normalisation)
    hybrid_score:    float         # penalised hybrid (0.0-1.0)
    coverage:        float         # concept coverage (0.0-1.0)
    concept_hits:    dict          # {"lane_detection": True, "sensor": False}
```

### `PipelineResult`

```python
class PipelineResult(BaseModel):
    query:              str
    backend:            str                  # "epo" | "patentsview"
    expanded_queries:   list[str]
    validated_queries:  list[str]
    primary_anchor:     str
    domain_concepts:    list[str]
    ranked_patents:     list[RankedPatent]
    comparison_json:    str
    report_markdown:    str
    taxonomy_miss:      bool = False
    error:              Optional[str] = None
```

---

## 7. Search Backend Routing

```
_resolve_backend()
│
├── reads .env via Pydantic Settings (strip() applied to all keys)
│
├── epo_consumer_key present and non-empty?
│   └── YES → SearchBackend.EPO
│
├── patentsview_api_key present and non-empty?
│   └── YES → SearchBackend.PATENTSVIEW
│
└── neither → RuntimeError (fail fast, no silent fallback)

Rules:
  - Backend resolved ONCE at pipeline start
  - EPO and PatentsView are mutually exclusive -- no cross-fallback
  - _assert_source() validates patent_id format matches backend
    PatentsView IDs: numeric only (10000000)
    EPO IDs:         country.number.kind (US.12345.A1)
```

---

## 8. Ranking & Scoring

### Score Computation

```
For each patent in corpus:

  cosine  = query_vec . doc_vec           (L2-normalised vectors)
  bm25    = BM25Okapi.get_scores(query)   (raw keyword frequency score)

  cosine_norm = (cosine - min) / (max - min + eps)
  bm25_norm   = (bm25   - min) / (max - min + eps)

  raw_hybrid  = 0.6 * cosine_norm + 0.4 * bm25_norm

  coverage    = matched_concept_groups / total_concept_groups
  penalty_multiplier = coverage^2         (quadratic curve)

  final_score = raw_hybrid * penalty_multiplier
```

### Coverage Penalty Examples

| Patent matches | Coverage | Multiplier | Effect |
|---|---|---|---|
| All 3 concepts | 1.00 | 1.00 | No penalty |
| 2 of 3 concepts | 0.67 | 0.45 | Moderate drop |
| 1 of 3 concepts | 0.33 | 0.11 | Heavy penalty |
| 0 concepts | 0.00 | 0.00 | Eliminated |

### Dynamic Threshold (Elbow Strategy)

```
scores sorted descending: [0.85, 0.81, 0.74, 0.71, 0.38, 0.12, 0.09]
diffs:                     [0.04, 0.07, 0.03, 0.33, 0.26, 0.03]
                                                ↑
                                         sharpest drop
threshold = 0.71  (score just before the elbow)
→ patents below 0.71 discarded
```

---

## 9. Configuration Reference

```python
# app/config.py

class Settings(BaseSettings):
    # API Keys (strip whitespace -- prevents .env trailing space bugs)
    patentsview_api_key:  str   = ""
    epo_consumer_key:     str   = ""
    epo_consumer_secret:  str   = ""
    anthropic_api_key:    str   = ""

    # Models
    mistral_model:        str   = "mistral"
    embedding_model:      str   = "BAAI/bge-large-en-v1.5"
    claude_model:         str   = "claude-3-5-sonnet-20241022"

    # Search
    patent_fields:        list  = ["patent_id","patent_title",
                                   "patent_abstract","patent_type","patent_date"]
    patent_page_size:     int   = 100
    max_results:          int   = 500
    max_pages:            int   = 5
    epo_max_results:      int   = 25
    epo_min_results:      int   = 10

    # Ranking
    cosine_weight:        float = 0.6
    bm25_weight:          float = 0.4
    coverage_curve:       str   = "quadratic"   # linear | quadratic | step
    threshold_strategy:   str   = "elbow"        # elbow | mean_plus_std | top_k_pct

    # LLM
    comparison_top_k:     int   = 10
    relevance_filter_k:   int   = 20
    max_tokens_compare:   int   = 4096
    max_tokens_report:    int   = 8192

    class Config:
        env_file = ".env"

    @validator("*", pre=True)
    def strip_strings(cls, v):
        return v.strip() if isinstance(v, str) else v
```

---

## 10. Key Design Decisions

### Separation of concerns

Models (`models/`) are pure HTTP/SDK wrappers with no business logic. Business logic lives exclusively in services (`services/`). Infrastructure (BM25, vector store, cosine) lives in retrieval (`retrieval/`). The orchestrator (`orchestrator/patent_pipeline.py`) calls services in order but contains no logic of its own.

### Vector store is a cache, not a corpus

The ranking corpus always comes from live search results. The vector store only caches embeddings by `patent_id` to avoid recomputing them on repeated queries. These two roles must never be swapped.

### Primary anchor drives discriminating search

The token extractor ranks matched concepts by discriminating power before passing them to the EPO CQL builder. Task concepts (`"lane detection"`, power=1) anchor the search. Sensor concepts (`"infrared"`, power=4) are supporting constraints. This prevents generic terms from dominating CQL and returning off-domain results.

### Coverage penalty enforces multi-concept relevance

A patent that mentions only `"infrared"` (1 of 3 concepts) scores `0.11*` its raw hybrid score due to the quadratic coverage penalty. This handles the case where BM25 inflates a score because a single term appears frequently -- the penalty demotes it without requiring manual threshold tuning.

### BGE-large asymmetric embedding

Queries are embedded with the instruction prefix `"Represent this sentence for searching relevant passages: "`. Documents are embedded without any prefix. This asymmetry is required by the BGE-large model for retrieval tasks -- symmetric embedding produces unreliable similarity scores.

### No silent cross-backend fallback

If EPO returns fewer than `MIN_RESULTS` patents, the system logs a warning and returns what it has. It does not fall through to PatentsView. This prevents the bug where PatentsView (which may be unreachable or misconfigured) silently receives queries intended for EPO, returning entirely wrong results.
