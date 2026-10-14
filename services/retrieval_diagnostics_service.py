"""Diagnostics for conventional patent retrieval."""

import re


def _normalise_text(value: str) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[-_]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _critical_concepts(strategy) -> list:
    return [
        concept
        for concept in getattr(strategy, "concepts", [])
        if str(getattr(concept, "importance", "")).strip().lower()
        == "critical"
        and any(
            str(term).strip()
            for term in getattr(concept, "terms", [])
        )
    ]


def calculate_retrieval_concept_coverage(
    patents: list,
    strategy,
) -> dict:
    """Measure literal critical-concept coverage across retrieved records.

    A concept counts as covered when any approved alternative term appears
    in a patent's title, abstract, or claims. This is a diagnostic only;
    it does not establish semantic equivalence or novelty.
    """
    concepts = _critical_concepts(strategy)
    concept_hits = {
        str(concept.name): 0
        for concept in concepts
    }

    records_with_any_match = 0
    valid_records = [patent for patent in patents if patent is not None]

    for patent in valid_records:
        text = _normalise_text(" ".join([
            getattr(patent, "patent_title", "") or "",
            getattr(patent, "patent_abstract", "") or "",
            getattr(patent, "patent_claims", "") or "",
        ]))

        record_matched = False

        for concept in concepts:
            name = str(getattr(concept, "name", "")).strip()
            terms = [
                _normalise_text(term)
                for term in getattr(concept, "terms", [])
                if str(term).strip()
            ]

            matched = False
            for term in terms:
                if re.search(
                    rf"\b(?:no|not|without|excluding|lacks?|absence(?:\s+of)?|never)\s+{re.escape(term)}\b",
                    text,
                ):
                    continue

                if re.search(rf"\b{re.escape(term)}\b", text):
                    matched = True
                    break

            if matched:
                concept_hits[name] += 1
                record_matched = True

        if record_matched:
            records_with_any_match += 1

    total_records = len(valid_records)

    per_concept = [
        {
            "concept": name,
            "matched_records": count,
            "total_records": total_records,
            "coverage_pct": (
                round(count / total_records * 100, 1)
                if total_records else 0.0
            ),
        }
        for name, count in concept_hits.items()
    ]

    return {
        "total_records": total_records,
        "critical_concept_count": len(concepts),
        "records_with_any_critical_match": records_with_any_match,
        "per_concept": per_concept,
        "unmatched_concepts": [
            item["concept"]
            for item in per_concept
            if item["matched_records"] == 0
        ],
        "warning": (
            "Literal term matching is diagnostic only. It does not "
            "establish semantic equivalence, relevance, or novelty."
        ),
    }


def calculate_pairwise_query_overlap(query_family_sets: list) -> list:
    """Calculate pairwise family overlap between executed queries.

    Each input item is a tuple:
        (query_index, query_text, set_of_identifiable_family_keys)

    Jaccard overlap is shared families divided by the union of families.
    Empty unions return 0%. Missing family IDs should be excluded before
    calling this function.
    """
    diagnostics = []

    for i, (index_a, query_a, families_a) in enumerate(
        query_family_sets
    ):
        families_a = set(families_a or [])

        for index_b, query_b, families_b in query_family_sets[i + 1:]:
            families_b = set(families_b or [])

            shared_families = families_a & families_b
            union_families = families_a | families_b

            overlap_pct = (
                round(
                    len(shared_families) / len(union_families) * 100,
                    1,
                )
                if union_families
                else 0.0
            )

            diagnostics.append(
                {
                    "Query A #": index_a,
                    "Query A": query_a,
                    "Query B #": index_b,
                    "Query B": query_b,
                    "Families A": len(families_a),
                    "Families B": len(families_b),
                    "Shared families": len(shared_families),
                    "Unique to A": len(families_a - families_b),
                    "Unique to B": len(families_b - families_a),
                    "Jaccard overlap (%)": overlap_pct,
                }
            )

    return diagnostics
