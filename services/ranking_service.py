"""
Hybrid patent ranking service.

Combines cosine-similarity scores and BM25 scores into a single hybrid
ranking for each patent, then returns a sorted list of RankedPatent objects.

Algorithm (clean / mathematically sound)
-----------------------------------------
1. Cosine similarity is already in [0, 1] for L2-normalised vectors — use raw.
   Do NOT re-normalise cosine relative to the batch; that would distort the
   absolute similarity signal (e.g. a batch where all patents score 0.45–0.50
   would be stretched to 0–1, making weak matches look strong).
2. BM25 is unbounded — normalise to [0, 1] with per-batch min-max scaling.
3. Hybrid = cosine_weight * cosine + bm25_weight * bm25_norm.
   Default weights: 0.7 cosine + 0.3 BM25 (semantic > keyword for patents).
   Weights live in app.config.settings and can be tuned via .env.
4. Sort descending, return the top-k results as RankedPatent objects.
"""

import logging

import numpy as np

from app.config import settings
from models.schemas import PatentRecord, RankedPatent

logger = logging.getLogger(__name__)

_EPSILON = 1e-9   # prevents division-by-zero in normalisation


def normalize(scores: np.ndarray) -> np.ndarray:
    """
    Min-max normalise *scores* to [0, 1].

    Args:
        scores: 1-D float array of arbitrary range.

    Returns:
        1-D float array with values in [0, 1].
        If all scores are identical the array is mapped to all-zeros.
    """
    mn, mx = float(scores.min()), float(scores.max())
    return (scores - mn) / (mx - mn + _EPSILON)


def hybrid_rank(
    cosine_scores: np.ndarray,
    bm25_scores: np.ndarray,
) -> np.ndarray:
    """
    Compute hybrid scores: raw cosine + normalised BM25.

    Cosine is used as-is (already in [0, 1] for unit-normalised vectors).
    BM25 is min-max normalised to [0, 1] per batch before combining.

    Weights are read from ``settings.cosine_weight`` (default 0.7) and
    ``settings.bm25_weight`` (default 0.3).

    Args:
        cosine_scores: Raw cosine similarity scores, shape (n,).  Must be in [0, 1].
        bm25_scores: Raw BM25 scores, shape (n,).  Unbounded — normalised here.

    Returns:
        Hybrid score array, shape (n,), in [0, 1].
    """
    # Cosine: use raw — do NOT normalise relative to batch.
    c = cosine_scores

    # BM25: unbounded → normalise to [0, 1].
    b = normalize(bm25_scores)

    return settings.cosine_weight * c + settings.bm25_weight * b


def _apply_cosine_threshold(
    cosine: np.ndarray,
    indices: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Filter *indices* to those whose cosine score meets the threshold.

    Starts at ``settings.cosine_threshold`` (default 0.40) and relaxes
    by 0.05 steps down to ``settings.cosine_threshold_min`` (default 0.30)
    until at least ``settings.cosine_min_candidates`` candidates remain.

    Returns:
        (filtered_indices, threshold_used)
    """
    threshold = settings.cosine_threshold
    floor = settings.cosine_threshold_min
    min_cands = settings.cosine_min_candidates

    while threshold >= floor:
        mask = cosine[indices] >= threshold
        passing = indices[mask]

        if len(passing) >= min_cands:
            break

        if threshold <= floor:
            # Already at minimum — accept whatever we have.
            break

        prev = threshold
        threshold = max(round(threshold - 0.05, 2), floor)

        logger.warning(
            "Cosine threshold relaxed: %.2f → %.2f (only %d candidates passed %.2f)",
            prev,
            threshold,
            len(passing),
            prev,
        )

    final_mask = cosine[indices] >= threshold
    filtered = indices[final_mask]
    discarded = len(indices) - len(filtered)

    logger.info(
        "Cosine threshold %.2f applied: %d passed, %d discarded",
        threshold,
        len(filtered),
        discarded,
    )

    return filtered, threshold


def _apply_token_anchor_penalty(
    hybrid: np.ndarray,
    patents: list[PatentRecord],
    critical_tokens: list[str],
    penalty_factor: float = 0.5,
) -> np.ndarray:
    """
    Multiply the hybrid score by *penalty_factor* for every patent whose
    text (title + abstract) contains **none** of the *critical_tokens*.

    *critical_tokens* should be the full anchor vocabulary built from ALL
    extractor groups::

        anchor_tokens = tokens.critical_tokens + tokens.patent_synonyms

    This means a patent must contain at least one of:
      - a surface term matched in the query (e.g. "camera", "pedestrian")
      - a patent-vocabulary synonym (e.g. "imaging system", "visual sensor")

    A patent about LADAR that shares only a single generic word will contain
    none of these anchors and will be penalised down the list.

    Args:
        hybrid:          Array of hybrid scores, shape (n,).
        patents:         List of PatentRecord objects aligned to *hybrid*.
        critical_tokens: Combined anchor vocabulary (surface terms + synonyms).
        penalty_factor: Multiplier applied to off-domain patents (default 0.5).

    Returns:
        New score array with penalties applied (does not modify *hybrid* in-place).
    """
    if not critical_tokens:
        return hybrid

    anchors = [t.lower() for t in critical_tokens]
    scores = hybrid.copy()

    for i, patent in enumerate(patents):
        text = (
            (patent.patent_title or "") + " " + (patent.patent_abstract or "")
        ).lower()

        if not any(anchor in text for anchor in anchors):
            scores[i] *= penalty_factor

            logger.debug(
                "Token anchor penalty applied to patent %s (factor=%.2f)",
                patent.patent_id,
                penalty_factor,
            )

    penalised = int(np.sum(scores != hybrid))

    if penalised:
        logger.info(
            "Token anchor enforcement: %d/%d patents penalised (missing all anchors).",
            penalised,
            len(patents),
        )

    return scores


def rank(
    query: str,
    patents: list[PatentRecord],
    doc_vecs: np.ndarray,
    query_vec: np.ndarray,
    top_k: int | None = None,
    critical_tokens: list[str] | None = None,
    tokens=None,
) -> list[RankedPatent]:
    """
    Rank *patents* using hybrid cosine + BM25 scoring.

    Pipeline:
      1. Compute raw cosine similarity for all candidates.
      2. Apply cosine threshold.
      3. Normalise BM25 over the filtered candidate set only.
      4. Compute hybrid = cosine + BM25.
      4b. Apply token coverage penalty.
      4c. Compute dynamic elbow threshold as a quality signal.
      5. Return up to top-k candidates.

    The dynamic threshold identifies the strongest result group, but does
    not reduce the requested top-k result count when enough candidates exist.

    Args:
        query:           Original query string — used for BM25.
        patents:         List of PatentRecord objects aligned to *doc_vecs*.
        doc_vecs:        L2-normalised embedding matrix.
        query_vec:       L2-normalised query embedding.
        top_k:           Maximum results to return.
        critical_tokens: Mandatory domain tokens from the token extractor.
        tokens:          Extracted token information used for concept coverage.

    Returns:
        List of RankedPatent objects sorted by hybrid_score descending.
    """
    from retrieval.similarity import cosine_similarity_matrix
    from retrieval.bm25_index import build_index
    from retrieval.bm25_ranker import get_scores

    k = top_k if top_k is not None else settings.top_k_results

    if not patents or doc_vecs.ndim < 2:
        logger.warning("rank() called with no patents; returning empty list.")
        return []

    # 1. Cosine similarity — raw, already in [0, 1] for L2-normalised vectors.
    cosine = cosine_similarity_matrix(query_vec, doc_vecs)

    # 2. Cosine threshold filter.
    all_indices = np.arange(len(patents))
    keep_indices, threshold_used = _apply_cosine_threshold(
        cosine,
        all_indices,
    )

    if len(keep_indices) == 0:
        logger.warning(
            "All patents filtered out by cosine threshold; returning empty list."
        )
        return []

    # Subset everything to the passing candidates.
    patents_f = [patents[i] for i in keep_indices]
    doc_vecs_f = doc_vecs[keep_indices]
    cosine_f = cosine[keep_indices]

    # 3. BM25 over the filtered candidate set only.
    bm25_raw = np.array(
        get_scores(build_index(patents_f), query),
        dtype=float,
    )

    # 4. Hybrid score.
    bm25_norm = normalize(bm25_raw)
    hybrid = hybrid_rank(cosine_f, bm25_raw)

    # 4b. Coverage-based scoring or binary anchor penalty.
    from retrieval.similarity import compute_token_coverage, apply_coverage_penalty

    coverage_data_list: list[dict] = []

    if tokens is not None and getattr(tokens, "concept_groups", None):
        penalised_hybrid = np.empty_like(hybrid)

        for i, patent in enumerate(patents_f):
            cov_data = compute_token_coverage(patent, tokens)
            coverage_data_list.append(cov_data)

            penalised_hybrid[i] = apply_coverage_penalty(
                float(hybrid[i]),
                cov_data["coverage"],
                penalty_curve="linear",
            )

            logger.debug(
                "Coverage penalty: patent=%s coverage=%.2f raw=%.4f "
                "penalised=%.4f hits=%s",
                patent.patent_id,
                cov_data["coverage"],
                float(hybrid[i]),
                float(penalised_hybrid[i]),
                cov_data["concept_hits"],
            )

        hybrid = penalised_hybrid

    else:
        hybrid = _apply_token_anchor_penalty(
            hybrid,
            patents_f,
            critical_tokens or [],
        )

        coverage_data_list = [
            {"coverage": 0.0, "concept_hits": {}}
            for _ in patents_f
        ]

    # 4c. Dynamic threshold as a quality signal.
    #
    # The elbow threshold identifies the strongest group of candidates.
    # However, it must not prevent top_k results from being returned when
    # enough candidates exist.
    from retrieval.similarity import compute_dynamic_threshold, filter_by_threshold

    dyn_threshold = compute_dynamic_threshold(
        hybrid,
        strategy="elbow",
    )

    passed = filter_by_threshold(
        hybrid,
        dyn_threshold,
    )

    threshold_pass_count = len(passed)

    if threshold_pass_count < k:
        # There are fewer threshold-qualified candidates than requested.
        # Fall back to the best k candidates from the complete cosine-filtered
        # set rather than returning fewer results than requested.
        fallback_count = min(k, len(hybrid))

        passed = np.argsort(hybrid)[::-1][:fallback_count]

        logger.info(
            "Dynamic threshold passed %d candidates; "
            "expanded to top-%d candidates to satisfy requested top_k=%d.",
            threshold_pass_count,
            len(passed),
            k,
        )

    if len(passed) == 0:
        logger.warning(
            "No candidates available after hybrid scoring; returning empty list."
        )
        return []

    # Preserve the complete cosine-filtered candidate set for fallback.
    patents_all = patents_f
    cosine_all = cosine_f
    bm25_all = bm25_norm
    hybrid_all = hybrid
    coverage_all = coverage_data_list

    # Preserve the dynamic-threshold-passed candidate set.
    patents_selected = [patents_all[i] for i in passed]
    cosine_selected = cosine_all[passed]
    bm25_selected = bm25_all[passed]
    hybrid_selected = hybrid_all[passed]
    coverage_selected = [
        coverage_all[i]
        for i in passed
    ]

    # 5. Sort by hybrid score and select top-k distinct continuity families.
    #
    # For USPTO records, continuation/divisional applications can represent
    # the same continuity family. Keep the highest-ranked member of each
    # family rather than showing multiple members of the same family.
    #
    # Records without a continuity_family_id are treated as unique records.
    sorted_idx = np.argsort(hybrid_selected)[::-1]

    ranked: list[RankedPatent] = []
    seen_families: set[str] = set()

    for i in sorted_idx:
        patent = patents_selected[i]

        family_id = getattr(
            patent,
            "continuity_family_id",
            None,
        )

        if family_id:
            family_key = f"family:{family_id}"

            if family_key in seen_families:
                continue

            seen_families.add(family_key)

        ranked.append(
            RankedPatent(
                patent=patent,
                cosine_score=float(cosine_selected[i]),
                bm25_score=float(bm25_selected[i]),
                hybrid_score=float(hybrid_selected[i]),
                coverage=float(
                    coverage_selected[i]["coverage"]
                ),
                concept_hits=coverage_selected[i]["concept_hits"],
            )
        )

        if len(ranked) >= k:
            break

    # If family deduplication reduced the result count below top_k,
    # continue with candidates that were below the dynamic threshold.
    if len(ranked) < k:
        passed_set = set(int(i) for i in passed)

        fallback_sorted_idx = np.argsort(hybrid_all)[::-1]

        for original_i in fallback_sorted_idx:
            original_i = int(original_i)

            if original_i in passed_set:
                continue

            patent = patents_all[original_i]

            family_id = getattr(
                patent,
                "continuity_family_id",
                None,
            )

            if family_id:
                family_key = f"family:{family_id}"

                if family_key in seen_families:
                    continue

                seen_families.add(family_key)

            ranked.append(
                RankedPatent(
                    patent=patent,
                    cosine_score=float(cosine_all[original_i]),
                    bm25_score=float(bm25_all[original_i]),
                    hybrid_score=float(hybrid_all[original_i]),
                    coverage=float(
                        coverage_all[original_i]["coverage"]
                    ),
                    concept_hits=coverage_all[original_i]["concept_hits"],
                )
            )

            if len(ranked) >= k:
                break

    logger.info(
        "Family-aware ranking: %d distinct patent families returned "
        "for requested top_k=%d.",
        len(ranked),
        k,
    )

    logger.info(
        "rank() complete: %d input → %d passed threshold %.2f → top-%d returned "
        "(best hybrid=%.4f)",
        len(patents),
        len(patents_f),
        threshold_used,
        len(ranked),
        ranked[0].hybrid_score if ranked else 0.0,
    )

    # Log a per-patent breakdown for the top-5 results.
    anchors = [
        t.lower()
        for t in (critical_tokens or [])
    ]

    logger.info(
        "─── Top-%d ranking breakdown ───",
        min(5, len(ranked)),
    )

    for pos, rp in enumerate(ranked[:5], start=1):
        text = (
            (rp.patent.patent_title or "")
            + " "
            + (rp.patent.patent_abstract or "")
        ).lower()

        overlap = sum(
            1
            for a in anchors
            if a in text
        )

        logger.info(
            "  #%d  id=%-14s  cosine=%.4f  bm25=%.4f  "
            "overlap=%d/%d  hybrid=%.4f  | %s",
            pos,
            rp.patent.patent_id or "N/A",
            rp.cosine_score,
            rp.bm25_score,
            overlap,
            len(anchors),
            rp.hybrid_score,
            (rp.patent.patent_title or "")[:60],
        )

    logger.info(
        "────────────────────────────────────"
    )

    return ranked