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
    """Build a Boolean expression from a manual SearchStrategy."""

    concept_expressions = []

    for concept in strategy.concepts:
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

        concept_expressions.append(expression)

    if not concept_expressions:
        return ""

    return f" {strategy.concept_operator} ".join(
        concept_expressions
    )


def build_uspto_search_strings(strategy: SearchStrategy) -> dict[str, str]:
    """
    Build field-specific USPTO Patent Public Search expressions.

    Returns one search expression per selected field:
    title    -> .TI.
    abstract -> .AB.
    claims   -> .CLM.
    """

    field_codes = {
        "title": ".TI.",
        "abstract": ".AB.",
        "claims": ".CLM.",
    }

    results = {}

    for field in strategy.search_fields:
        field_code = field_codes.get(field.lower())

        if not field_code:
            continue

        concept_expressions = []

        for concept in strategy.concepts:
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

            # Apply the USPTO field restriction to this concept.
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


def build_uspto_proximity_expression(
    rule,
) -> str:
    """
    Build a USPTO proximity expression.

    Examples:
        reading ADJ non-invasive
        reading ADJ4 "non invasive"
        reading NEAR10 "non invasive"
    """

    left = _format_term(rule.left_term)
    right = _format_term(rule.right_term)

    if not left or not right:
        return ""

    operator = rule.operator.upper()

    if operator not in {"ADJ", "NEAR"}:
        raise ValueError(
            f"Unsupported proximity operator: {rule.operator}"
        )

    if rule.distance < 1:
        raise ValueError("Proximity distance must be >= 1")

    # USPTO uses ADJ by itself for adjacency and ADJn / NEARn
    # when a distance is specified.
    if operator == "ADJ" and rule.distance == 1:
        proximity_operator = "ADJ"
    else:
        proximity_operator = f"{operator}{rule.distance}"

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
