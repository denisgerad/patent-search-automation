import logging

from models.schemas import SearchStrategy

logger = logging.getLogger(__name__)


def _format_term(term: str) -> str:
    """Quote multi-word terms for phrase searching."""
    term = term.strip()

    if not term:
        return ""

    if " " in term:
        return f'"{term}"'

    return term


def _has_expression(value: str | None) -> bool:
    return bool(value and value.strip())


def _combine_roles(*expressions: str) -> str:
    """Combine non-empty role expressions using AND."""

    parts = [
        expression.strip()
        for expression in expressions
        if _has_expression(expression)
    ]

    return " AND ".join(
        f"({part})"
        for part in parts
    )


def _build_role_expression(terms: list[str]) -> str:
    """Build an OR expression for one search role."""

    cleaned = [
        _format_term(str(term).strip())
        for term in terms
        if str(term).strip()
    ]

    if not cleaned:
        return ""

    if len(cleaned) == 1:
        return cleaned[0]

    return f"({' OR '.join(cleaned)})"


def _build_role_variants(terms: list[str]) -> list[str]:
    """Build a deterministic, bounded set of search-role variants.

    The original terminology is retained. This only adds useful
    phrase/individual-token variants and does not invent synonyms.
    """
    variants: list[str] = []

    for raw_term in terms:
        term = str(raw_term).strip()

        if not term:
            continue

        if term not in variants:
            variants.append(term)

        words = [
            word.strip()
            for word in term.split()
            if word.strip()
        ]

        if len(words) > 1:
            for word in words:
                if len(word) >= 4 and word not in variants:
                    variants.append(word)

    return variants


def validate_core_terms(terms: list[str]) -> list[str]:
    """Keep core terms suitable for lexical patent search.

    Reject only obvious generic stand-alone terms that are not likely to
    describe the invention identity. Preserve distinctive arrangements that
    include generic technology as part of a more specific phrase.
    """
    generic_terms = {
        "camera",
        "sensor",
        "processor",
        "computer",
        "artificial intelligence",
        "ai",
        "machine learning",
        "image processing",
        "facial recognition",
        "face detection",
    }

    validated: list[str] = []
    for term in terms:
        cleaned = str(term).strip()
        if not cleaned:
            continue
        if cleaned.lower() in generic_terms:
            continue
        validated.append(cleaned)

    return validated


def _build_core_search_expression(core_terms: list[str]) -> str:
    """Build a core-only recall expression from the original core terms."""

    terms = validate_core_terms(core_terms)

    if not terms:
        return ""

    return " OR ".join(
        f'"{term}"' if " " in term else term
        for term in terms
    )


def build_search_paths_from_structure(
    structure: dict,
) -> list[dict]:
    """
    Build deterministic patent search paths from the AI invention
    structure.

    Core paths represent invention identity.
    Sensor/function paths provide implementation-level recall.
    """

    core_expr = _build_role_expression(
        validate_core_terms(structure.get("core_system", []))
    )

    sensor_expr = _build_role_expression(
        structure.get("sensors", [])
    )

    function_expr = _build_role_expression(
        structure.get("functions", [])
    )

    object_expr = _build_role_expression(
        structure.get("objects", [])
    )

    logger.info(
        "SEARCH ROLE EXPRESSIONS | "
        "core=%r | sensor=%r | function=%r | object=%r",
        core_expr,
        sensor_expr,
        function_expr,
        object_expr,
    )

    paths: list[dict] = []

    # =========================================================
    # TIER A — INVENTION IDENTITY
    # =========================================================

    if core_expr:
        paths.append({
            "name": "Core Only",
            "expression": core_expr,
            "quality_level": "recall",
            "recall_priority": "very_high",
            "precision_priority": "medium",
            "limit": 25,
        })

    expression = _combine_roles(
        core_expr,
        function_expr,
    )

    if expression:
        paths.append({
            "name": "Core + Function",
            "expression": expression,
            "quality_level": "broad",
            "recall_priority": "high",
            "precision_priority": "medium",
            "limit": 100,
        })

    expression = _combine_roles(
        core_expr,
        sensor_expr,
    )

    if expression:
        paths.append({
            "name": "Core + Sensor",
            "expression": expression,
            "quality_level": "balanced",
            "recall_priority": "medium",
            "precision_priority": "high",
            "limit": 100,
        })

    expression = _combine_roles(
        core_expr,
        function_expr,
        sensor_expr,
    )

    if expression:
        paths.append({
            "name": "Core + Function + Sensor",
            "expression": expression,
            "quality_level": "focused",
            "recall_priority": "low",
            "precision_priority": "high",
            "limit": 100,
        })

    # =========================================================
    # TIER B — IMPLEMENTATION DISCOVERY
    # =========================================================

    expression = _combine_roles(
        sensor_expr,
        function_expr,
    )

    if expression:
        paths.append({
            "name": "Sensor + Function",
            "expression": expression,
            "quality_level": "broad",
            "recall_priority": "high",
            "precision_priority": "medium",
            "limit": 100,
        })

    expression = _combine_roles(
        sensor_expr,
        object_expr,
        function_expr,
    )

    if expression:
        paths.append({
            "name": "Sensor + Object + Function",
            "expression": expression,
            "quality_level": "focused",
            "recall_priority": "medium",
            "precision_priority": "high",
            "limit": 100,
        })

    logger.info(
        "GENERATED SEARCH PATHS: %s",
        [path["name"] for path in paths],
    )

    return paths


def _field_expression(term_expression: str, fields: list[str]) -> str:
    """Expand one concept expression across the selected USPTO fields.

    Example:
        ("fraud score" OR "risk score")
    becomes:
        (("fraud score" OR "risk score").TI. OR ("fraud score" OR "risk score").AB. OR ("fraud score" OR "risk score").CLM.)
    """
    field_map = {
        "title": "TI",
        "abstract": "AB",
        "claims": "CLM",
    }

    expressions = []
    normalized = term_expression.strip()

    for field in fields:
        code = field_map.get(str(field).lower())
        if not code:
            continue

        if normalized.startswith("(") and normalized.endswith(")"):
            expressions.append(f"{normalized}.{code}.")
        else:
            expressions.append(f"({normalized}).{code}.")

    if not expressions:
        return ""

    return "(" + " OR ".join(expressions) + ")"


def build_boolean_search(strategy: SearchStrategy) -> str:
    """Build a field-aware Boolean search from critical concepts."""

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
            term.strip()
            for term in getattr(concept, "terms", [])
            if str(term).strip()
        ]

        if not terms:
            continue

        operator = str(getattr(concept, "operator", "OR")).upper()
        if operator not in {"AND", "OR"}:
            operator = "OR"

        expression = f" {operator} ".join(
            _format_term(term) for term in terms
        )

        if len(terms) > 1:
            expression = f"({expression})"

        concept_expressions.append(
            _field_expression(expression, strategy.search_fields)
        )

    if not concept_expressions:
        return ""

    boolean_query = " AND ".join(
        expression for expression in concept_expressions if expression
    )

    print("\n===== GENERATED BOOLEAN SEARCH =====")
    print(boolean_query)
    print("====================================\n")

    return boolean_query


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
