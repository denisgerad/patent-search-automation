import numpy as np

from models.schemas import PatentRecord
from services import ranking_service


def make_patent(patent_id, title, family_id=None):
    return PatentRecord(
        patent_id=patent_id,
        patent_title=title,
        patent_abstract=(
            "A system for transaction fraud detection using "
            "device characteristics and historical behavior."
        ),
        continuity_family_id=family_id,
    )


def test_rank_returns_top_k_in_score_order_and_deduplicates_families(
    monkeypatch,
):
    patents = [
        make_patent("P100", "Adaptive fraud detection", "FAMILY-A"),
        make_patent("P101", "Transaction risk scoring", "FAMILY-A"),
        make_patent("P200", "Behavior-based fraud detection", "FAMILY-B"),
        make_patent("P300", "Transaction monitoring", "FAMILY-C"),
    ]

    doc_vecs = np.eye(4, dtype=float)
    query_vec = np.array([1.0, 0.0, 0.0, 0.0])

    monkeypatch.setattr(
        "retrieval.similarity.cosine_similarity_matrix",
        lambda query, docs: np.array([0.95, 0.85, 0.75, 0.65]),
    )
    monkeypatch.setattr(
        "retrieval.bm25_ranker.get_scores",
        lambda index, query: [4.0, 3.0, 2.0, 1.0],
    )
    monkeypatch.setattr(
        "retrieval.bm25_index.build_index",
        lambda records: object(),
    )
    monkeypatch.setattr(
        "retrieval.similarity.compute_dynamic_threshold",
        lambda scores, strategy="elbow": -1.0,
    )
    monkeypatch.setattr(
        "retrieval.similarity.filter_by_threshold",
        lambda scores, threshold: np.arange(len(scores)),
    )
    monkeypatch.setattr(
        ranking_service,
        "_apply_token_anchor_penalty",
        lambda scores, records, critical_tokens, penalty_factor=0.5: scores,
    )
    monkeypatch.setattr(
        ranking_service,
        "_apply_importance_score_if_text_available",
        lambda patent, hybrid, importance_score, importance_hits,
        search_strategy: hybrid,
    )

    ranked = ranking_service.rank(
        query="transaction fraud detection",
        patents=patents,
        doc_vecs=doc_vecs,
        query_vec=query_vec,
        top_k=3,
    )

    assert [item.patent.patent_id for item in ranked] == [
        "P100",
        "P200",
        "P300",
    ]
    assert len(ranked) == 3
    assert all(
        ranked[i].hybrid_score >= ranked[i + 1].hybrid_score
        for i in range(len(ranked) - 1)
    )


def test_fraud_detection_acceptance_cases_rank_by_concept_coverage():
    from services.ranking_service import _calculate_importance_score
    from models.schemas import SearchConcept, SearchStrategy

    strategy = SearchStrategy(
        original_input=(
            "Context-aware fraud detection with adaptive user thresholds"
        ),
        concepts=[
            SearchConcept(
                name="Transaction information",
                importance="critical",
                terms=["transaction information"],
                operator="AND",
            ),
            SearchConcept(
                name="Device characteristics",
                importance="critical",
                terms=["device characteristics"],
                operator="AND",
            ),
            SearchConcept(
                name="Historical behavior",
                importance="critical",
                terms=["historical behavioral patterns"],
                operator="AND",
            ),
            SearchConcept(
                name="Dynamic fraud scoring",
                importance="important",
                terms=["dynamic fraud score"],
                operator="AND",
            ),
            SearchConcept(
                name="Adaptive threshold",
                importance="important",
                terms=["adaptive threshold"],
                operator="AND",
            ),
        ],
    )

    cases = {
        "high": make_patent(
            "HIGH",
            "Context-aware fraud detection",
        ).model_copy(update={
            "patent_abstract": (
                "The system combines transaction information, "
                "device characteristics and historical behavioral patterns "
                "to calculate a dynamic fraud score and apply an "
                "adaptive threshold."
            )
        }),
        "partial": make_patent(
            "PARTIAL",
            "Transaction fraud scoring",
        ).model_copy(update={
            "patent_abstract": (
                "The system uses transaction information to calculate "
                "a dynamic fraud score."
            )
        }),
        "low": make_patent(
            "LOW",
            "Transaction processing system",
        ).model_copy(update={
            "patent_abstract": (
                "The system processes transaction records and generates "
                "a transaction confirmation."
            )
        }),
    }

    scores = {
        label: _calculate_importance_score(patent, strategy)[0]
        for label, patent in cases.items()
    }

    assert scores["high"] > scores["partial"]
    assert scores["partial"] > scores["low"]


def test_contractor_matching_acceptance_cases_rank_by_concept_coverage():
    """Distinguish full, partial, and weak overlap for contractor matching."""
    from models.schemas import SearchConcept, SearchStrategy
    from services.ranking_service import _calculate_importance_score

    strategy = SearchStrategy(
        original_input=(
            "Contractor matching with eligibility gating, weighted criteria, "
            "dependency-driven incremental recalculation, and incremental ranking"
        ),
        concepts=[
            SearchConcept(
                name="Contractor matching and ranking",
                importance="critical",
                terms=["contractor matching", "contractor ranking"],
                operator="OR",
            ),
            SearchConcept(
                name="Mandatory eligibility gating",
                importance="critical",
                terms=["mandatory eligibility gating", "eligibility filtering"],
                operator="OR",
            ),
            SearchConcept(
                name="Dependency-driven incremental recalculation",
                importance="critical",
                terms=[
                    "dependency graph",
                    "incremental recalculation",
                    "affected evaluation states",
                ],
                operator="OR",
            ),
            SearchConcept(
                name="Weighted criterion contributions",
                importance="important",
                terms=[
                    "weighted criterion contributions",
                    "criterion-level contribution states",
                ],
                operator="OR",
            ),
            SearchConcept(
                name="Incremental aggregate score and ranking updates",
                importance="important",
                terms=[
                    "aggregate contractor score",
                    "incremental ranking update",
                ],
                operator="OR",
            ),
        ],
    )

    cases = {
        "high": make_patent(
            "CONTRACTOR-HIGH",
            "Contractor matching and ranking with eligibility gating",
        ).model_copy(update={
            "patent_abstract": (
                "A contractor matching system performs mandatory eligibility "
                "gating, maintains a dependency graph for affected evaluation "
                "states, and supports incremental recalculation. It stores "
                "weighted criterion contributions and updates the aggregate "
                "contractor score and incremental ranking update when project "
                "criteria change."
            )
        }),
        "partial": make_patent(
            "CONTRACTOR-PARTIAL",
            "Incremental dependency graph evaluation for software builds",
        ).model_copy(update={
            "patent_abstract": (
                "A software build system maintains a dependency graph and "
                "performs incremental recalculation of affected evaluation "
                "states when source components change, while retaining "
                "unaffected build states."
            )
        }),
        "low": make_patent(
            "CONTRACTOR-LOW",
            "General data processing and report generation",
        ).model_copy(update={
            "patent_abstract": (
                "A data processing system receives records, stores the records, "
                "and generates a periodic report for a user."
            )
        }),
    }

    scores = {
        label: _calculate_importance_score(patent, strategy)[0]
        for label, patent in cases.items()
    }

    assert scores["high"] > scores["partial"]
    assert scores["partial"] > scores["low"]


def test_concept_coverage_reports_full_partial_and_zero_matches():
    from models.schemas import SearchConcept, SearchStrategy

    strategy = SearchStrategy(
        original_input="Contractor matching with incremental recalculation",
        concepts=[
            SearchConcept(
                name="Contractor matching",
                importance="critical",
                terms=["contractor matching", "contractor ranking"],
            ),
            SearchConcept(
                name="Dependency recalculation",
                importance="critical",
                terms=["dependency graph", "incremental recalculation"],
            ),
        ],
    )

    full = PatentRecord(
        patent_id="FULL",
        patent_title="CONTRACTOR MATCHING",
        patent_abstract="Incremental_recalculation using a dependency graph.",
    )
    partial = PatentRecord(
        patent_id="PARTIAL",
        patent_title="Dependency graph processing",
        patent_abstract="Updates affected states.",
    )
    none = PatentRecord(
        patent_id="NONE",
        patent_title="Weather monitoring",
        patent_abstract="Collects temperature measurements.",
    )

    full_result = ranking_service._calculate_concept_coverage(full, strategy)
    partial_result = ranking_service._calculate_concept_coverage(
        partial, strategy
    )
    none_result = ranking_service._calculate_concept_coverage(none, strategy)

    assert full_result["coverage"] == 1.0
    assert full_result["matched_concepts"] == 2
    assert full_result["total_concepts"] == 2

    assert partial_result["coverage"] == 0.5
    assert partial_result["concept_hits"] == {
        "Contractor matching": False,
        "Dependency recalculation": True,
    }

    assert none_result["coverage"] == 0.0
    assert none_result["matched_concepts"] == 0
