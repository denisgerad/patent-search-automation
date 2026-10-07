from models.schemas import SearchStrategy


def _format_term(term: str) -> str:
    """Quote multi-word terms for phrase searching."""
    term = term.strip()

    if not term:
        return ""

    if " " in term:
        return f'"{term}"'

    return term


def build_boolean_search(strategy: SearchStrategy) -> str:
    """Build pairwise Boolean retrieval queries from critical concepts."""

    critical_concepts = [
        concept
        for concept in strategy.concepts
        if concept.importance.lower() == "critical"
        and any(term.strip() for term in concept.terms)
    ]

    if len(critical_concepts) < 2:
        return ""

    concept_expressions = []

    for concept in critical_concepts:
        terms = [
            _format_term(term)
            for term in concept.terms
            if term.strip()
        ]

        if len(terms) == 1:
            expression = terms[0]
        else:
            expression = f"({' OR '.join(terms)})"

        concept_expressions.append(expression)

    pairwise_queries = []

    for i in range(len(concept_expressions)):
        for j in range(i + 1, len(concept_expressions)):
            pairwise_queries.append(
                f"{concept_expressions[i]} AND {concept_expressions[j]}"
            )

    return "\n".join(pairwise_queries)


def build_search_queries(strategy: SearchStrategy) -> list[str]:
    """Build multiple retrieval-oriented queries from critical concepts."""

    critical_concepts = [
        concept
        for concept in strategy.concepts
        if concept.importance.lower() == "critical"
        and concept.terms
    ]

    if not critical_concepts:
        return []

    queries: list[str] = []

    for concept in critical_concepts:
        terms = [
            _format_term(term)
            for term in concept.terms
            if term.strip()
        ]

        if not terms:
            continue

        if len(terms) == 1:
            expression = terms[0]
        else:
            expression = f"({' OR '.join(terms)})"

        queries.append(expression)

    return queries


def build_core_pair_queries(strategy: SearchStrategy) -> list[str]:
    """Build pairwise queries from critical concepts.

    Each concept uses OR internally, while different concepts
    are combined with AND.
    """

    critical_concepts = [
        concept
        for concept in strategy.concepts
        if concept.importance.lower() == "critical"
        and any(term.strip() for term in concept.terms)
    ]

    if len(critical_concepts) < 2:
        return []

    concept_expressions: list[str] = []

    for concept in critical_concepts:
        terms = [
            _format_term(term)
            for term in concept.terms
            if term.strip()
        ]

        if not terms:
            continue

        if len(terms) == 1:
            expression = terms[0]
        else:
            expression = f"({' OR '.join(terms)})"

        concept_expressions.append(expression)

    pairwise_queries: list[str] = []

    for i in range(len(concept_expressions)):
        for j in range(i + 1, len(concept_expressions)):
            pairwise_queries.append(
                f"{concept_expressions[i]} AND "
                f"{concept_expressions[j]}"
            )

    return pairwise_queries


def build_uspto_search_strings(strategy: SearchStrategy) -> dict[str, str]:
    """
    Build field-specific USPTO Patent Public Search expressions
    from critical concepts only.

    The displayed expressions should match the concepts used by
    the approved retrieval strategy.
    """

    field_codes = {
        "title": ".TI.",
        "abstract": ".AB.",
        "claims": ".CLM.",
    }

    critical_concepts = [
        concept
        for concept in strategy.concepts
        if concept.importance.lower() == "critical"
        and any(term.strip() for term in concept.terms)
    ]

    results = {}

    for field in strategy.search_fields:
        field_code = field_codes.get(field.lower())

        if not field_code:
            continue

        concept_expressions = []

        for concept in critical_concepts:
            terms = [
                _format_term(term)
                for term in concept.terms
                if term.strip()
            ]

            if not terms:
                continue

            expression = f" {concept.operator} ".join(terms)

            if len(terms) > 1:
                expression = f"({expression})"

            expression = f"{expression}{field_code}"

            concept_expressions.append(expression)

        if not concept_expressions:
            continue

        results[field] = (
            f" {strategy.concept_operator} ".join(
                concept_expressions
            )
        )

    return results


def build_uspto_proximity_expression(rule) -> str:
    """
    Build a USPTO Patent Public Search proximity expression.

    ADJ and NEAR may use a numeric distance.
    WITH and SAME are represented without a numeric distance.
    """

    left = _format_term(rule.left_term)
    right = _format_term(rule.right_term)

    if not left or not right:
        return ""

    operator = rule.operator.upper()

    if operator not in {"ADJ", "NEAR", "WITH", "SAME"}:
        raise ValueError(
            f"Unsupported proximity operator: {rule.operator}"
        )

    # WITH and SAME do not use the numeric-distance form here.
    if operator in {"WITH", "SAME"}:
        proximity_operator = operator

    else:
        distance = rule.distance or 1

        if distance < 1:
            raise ValueError(
                "Proximity distance must be >= 1"
            )

        if operator == "ADJ" and distance == 1:
            proximity_operator = "ADJ"
        else:
            proximity_operator = f"{operator}{distance}"

    return f"({left} {proximity_operator} {right})"


def build_uspto_proximity_strings(
    strategy: SearchStrategy,
) -> list[str]:
    """Build field-specific USPTO proximity expressions."""

    field_codes = {
        "title": ".TI.",
        "abstract": ".AB.",
        "claims": ".CLM.",
    }

    results = []

    for rule in strategy.proximity_rules:
        expression = build_uspto_proximity_expression(rule)

        if not expression:
            continue

        field_code = field_codes.get(rule.field.lower())

        if field_code:
            expression = f"{expression}{field_code}"

        results.append(expression)

    return results
