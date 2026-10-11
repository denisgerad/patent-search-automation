
from services.uspto_search_service import search_patents
from services.ranking_service import _calculate_importance_score
from models.schemas import SearchConcept, SearchStrategy


QUERY = "transaction fraud detection"

# Keep this aligned with the approved strategy used in the Streamlit app.
# Replace or extend these entries with your actual approved concepts.
CONCEPTS = [
    SearchConcept(
        name="Context-aware fraud detection",
        importance="critical",
        terms=[
            "context-aware fraud detection",
            "contextual fraud analysis",
            "transaction context analysis",
            "contextual indicators",
        ],
        operator="OR",
    ),
    SearchConcept(
        name="Dynamic fraud scoring",
        importance="critical",
        terms=[
            "fraud score",
            "risk score",
            "fraud probability",
            "fraud risk state",
        ],
        operator="OR",
    ),
    SearchConcept(
        name="Adaptive detection threshold",
        importance="important",
        terms=[
            "adaptive threshold",
            "adaptive decision threshold",
            "risk based threshold",
            "threshold adjusted based on user behavior",
        ],
        operator="OR",
    ),
    SearchConcept(
        name="Behavioral pattern analysis",
        importance="important",
        terms=[
            "behavioral analytics",
            "behavioral baselines",
            "historical behavioral patterns",
            "user actions over time",
        ],
        operator="OR",
    ),
    SearchConcept(
        name="Transaction information analysis",
        importance="important",
        terms=[
            "transaction analysis",
            "transaction attributes",
            "transaction features",
            "current transaction information",
        ],
        operator="OR",
    ),
    SearchConcept(
        name="Device characteristics fingerprinting",
        importance="important",
        terms=[
            "device fingerprinting",
            "device identification",
            "device attributes",
            "device characteristics",
            "device identifier",
        ],
        operator="OR",
    ),
    SearchConcept(
        name="Historical user behavior modeling",
        importance="important",
        terms=[
            "historical transaction behavior",
            "user activity history",
            "historical behavior baseline",
            "user behavior over time",
        ],
        operator="OR",
    ),
    SearchConcept(
        name="Multi factor fraud signal fusion",
        importance="important",
        terms=[
            "fraud signal fusion",
            "multi factor risk assessment",
            "feature vectors",
            "combined transaction and device data",
        ],
        operator="OR",
    ),
    SearchConcept(
        name="User specific baseline profiling",
        importance="important",
        terms=[
            "individual behavioral baseline",
            "personalized risk profile",
            "established behavioral baseline",
        ],
        operator="OR",
    ),
    SearchConcept(
        name="Real-time fraud evaluation",
        importance="important",
        terms=[
            "real-time fraud detection",
            "online fraud scoring",
            "real-time risk evaluation",
        ],
        operator="OR",
    ),
]

STRATEGY = SearchStrategy(
    original_input=QUERY,
    concepts=CONCEPTS,
)


def main():
    patents = search_patents(QUERY, limit=10)

    if not patents:
        print("No patents returned.")
        return

    results = []

    for patent in patents:
        score, hits = _calculate_importance_score(patent, STRATEGY)

        matched = [
            concept.name
            for concept in CONCEPTS
            if hits.get(concept.name, False)
        ]
        missing = [
            concept.name
            for concept in CONCEPTS
            if not hits.get(concept.name, False)
        ]

        results.append({
            "patent": patent,
            "score": score,
            "matched": matched,
            "missing": missing,
        })

    # Compact comparison of all retrieved patents
    print(f"\nRetrieved {len(results)} patents")
    print("Concept-coverage scores (not final ranking scores)")
    print("=" * 100)

    for i, result in enumerate(results, start=1):
        patent = result["patent"]
        matched = result["matched"]

        print(
            f"{i:2}. {patent.patent_id} | "
            f"Score: {result['score']:.4f} | "
            f"Matched: {len(matched)}/{len(CONCEPTS)}"
        )
        print(f"    Title: {patent.patent_title}")
        print(
            "    Hits: "
            + (", ".join(matched) if matched else "None")
        )

    # Detailed misses for follow-up investigation
    print("\n\nDETAILED CONCEPT MISSES")
    print("=" * 100)

    for result in results:
        patent = result["patent"]

        if not result["missing"]:
            continue

        print(
            f"\n{patent.patent_id} | "
            f"{patent.patent_title} | "
            f"Score: {result['score']:.4f}"
        )
        for concept_name in result["missing"]:
            print(f"  MISS: {concept_name}")


if __name__ == "__main__":
    main()
