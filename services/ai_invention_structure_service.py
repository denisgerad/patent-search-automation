"""AI-assisted invention structure and patent search path generation."""

from __future__ import annotations

import json
import logging
import re

from services.claude_client import ClaudeClient

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """
You are an expert patent search strategist.

Analyze an invention disclosure and decompose it into five search roles:

1. application
2. core_system
3. sensors
4. functions
5. objects

The goal is patent prior-art search, not general brainstorming.

Definitions:

application:
The application/domain in which the invention operates.

core_system:
The main technical system, apparatus, method, or subsystem.

CORE SYSTEM — PATENT SEARCH TERMINOLOGY

The core_system is the most important part of the invention
composition.

It must identify the technical system, apparatus, subsystem,
architecture, or component arrangement that represents the invention.

However, the terminology must also be suitable for patent searching.

For each core system concept, provide terminology that is likely to
appear in patent claims, abstracts, titles, or descriptions.

DO NOT use abstract or marketing language.

DO NOT use generic enabling technologies unless they are themselves
the invention.

Avoid standalone terms such as:

- camera
- sensor
- processor
- computer
- AI
- artificial intelligence
- machine learning
- image processing
- facial recognition
- face detection

unless one of these is genuinely the inventive subject.

Instead identify the technical combination or arrangement.

For example, prefer terminology of the form:

"[distinctive component] + [technical relationship] + [technical operation]"

rather than a generic technology name.

The core terminology should satisfy this test:

"If I searched this phrase by itself in a patent database,
would I reasonably expect to retrieve patents describing this
specific technical invention or a closely related invention?"

If the answer is no, make the terminology more technically specific.

IMPORTANT:

Generate between 2 and 5 core concepts.

Each concept should be independently searchable.

For each concept, provide:

- a concise technical phrase
- important patent terminology/synonyms
- the distinctive technical relationship

Do not combine every invention detail into one extremely long phrase.

Do not use natural-language sentences as search terms.

Prefer short, searchable phrases such as:

- coordinated multi-camera illumination system
- synchronized stereo image registration
- adaptive light pattern projection and detection
- event-driven image correlation architecture

The core system should represent the invention identity,
not the generic hardware used to implement it.

sensors:
Sensors, sensing technologies, cameras, detectors, or input devices.

functions:
The specific technical functions performed by the invention.
Describe what the invention actually does in its stated application.
Prefer concise invention-specific phrases such as "lane detection",
"lane recognition", "lane tracking", "lane marking detection",
rather than generic verbs such as "detect", "identify", or "classify".
Do not replace the invention-specific function with generic search verbs.
If useful, provide closely related technical function phrases,
but keep them specific to the invention.

objects:
The thing being detected, measured, identified, controlled, or processed.

For each role provide technically useful patent-search synonyms.

Then construct several search paths by combining the roles.

Required search paths:

- Sensor + Function
- Core + Function + Sensor
- Sensor + Object + Function
- Core + Sensor + Object
- Application + Core + Sensor + Object

Rules:

- Do not invent technical features not supported by the invention.
- Keep terms suitable for patent searching.
- Prefer concise noun phrases.
- Functions should remain invention-specific technical phrases.
- Do not use generic wildcard verb stems for functions.
- Multi-word function phrases should be quoted in search expressions.
- Search paths should contain Boolean expressions.
- Use OR between synonyms within a role.
- Use AND between different roles.
- Multi-word phrases should be quoted.
- Wildcards may be used for function terms where appropriate.
- These are search suggestions and must be reviewed by the user.
- Do not automatically select or execute a search path.

Return JSON only.
"""


USER_TEMPLATE = """
Analyze this invention:

{invention}

Return JSON in exactly this general structure:

{{
  "application": [
    "..."
  ],
  "core_system": [
    "..."
  ],
  "sensors": [
    "..."
  ],
  "functions": [
    "..."
  ],
  "objects": [
    "..."
  ],
  "search_paths": [
    {{
      "name": "Sensor + Function",
      "roles": ["sensors", "functions"],
      "expression": "...",
      "purpose": "..."
    }}
  ]
}}
"""


def _extract_json(text: str) -> dict:
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(r"\s*```$", "", text)

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("Claude did not return valid JSON.")

    return json.loads(text[start:end + 1])


def _clean_terms(values) -> list[str]:
    if not isinstance(values, list):
        return []

    result = []

    for value in values:
        value = str(value).strip()

        if value and value not in result:
            result.append(value)

    return result


def _clean_paths(values) -> list[dict]:
    if not isinstance(values, list):
        return []

    result = []

    for item in values:
        if not isinstance(item, dict):
            continue

        name = str(item.get("name", "")).strip()
        expression = str(item.get("expression", "")).strip()
        purpose = str(item.get("purpose", "")).strip()

        roles = item.get("roles", [])

        if not isinstance(roles, list):
            roles = []

        roles = [
            str(role).strip()
            for role in roles
            if str(role).strip()
        ]

        if name and expression:
            result.append(
                {
                    "name": name,
                    "roles": roles,
                    "expression": expression,
                    "purpose": purpose,
                }
            )

    return result


def _quote_term(term: str) -> str:
    """Quote multi-word search terms."""

    term = term.strip()

    if not term:
        return ""

    if " " in term:
        return f'"{term}"'

    return term


def _role_expression(
    terms: list[str],
    wildcard: bool = False,
) -> str:
    """Build an OR expression from approved role terms."""

    cleaned = []

    for term in terms:
        term = term.strip()

        if not term:
            continue

        if wildcard:
            # Functions are normally verbs. Use the supplied term
            # as a searchable stem.
            if not term.endswith("*"):
                term = f"{term}*"

            cleaned.append(term)
        else:
            cleaned.append(_quote_term(term))

    if not cleaned:
        return ""

    if len(cleaned) == 1:
        return cleaned[0]

    return "(" + " OR ".join(cleaned) + ")"


def _combine_roles(*expressions: str) -> str:
    """Combine role expressions in a compact, explicit Boolean form."""

    parts = [
        expression.strip()
        for expression in expressions
        if expression and expression.strip()
    ]

    if not parts:
        return ""

    return " AND ".join(f"({part})" for part in parts)


def _build_broadened_role_expression(terms: list[str]) -> str:
    """Broaden a sparse role list without losing the role structure."""

    cleaned = [
        str(term).strip()
        for term in terms
        if str(term).strip()
    ]

    if not cleaned:
        return ""

    return " OR ".join(
        f'"{term}"' if " " in term else term
        for term in cleaned
    )


def build_search_paths_from_structure(
    structure: dict,
) -> list[dict]:
    """
    Build deterministic search paths from the reviewed invention structure.

    Each path represents a different point on the recall/precision spectrum.

    Recall:
        More general role combinations.

    Precision:
        More invention-specific role combinations.

    The user's edited terminology is authoritative.
    """

    application = _clean_terms(
        structure.get("application", [])
    )

    core_system = _clean_terms(
        structure.get("core_system", [])
    )

    sensors = _clean_terms(
        structure.get("sensors", [])
    )

    functions = _clean_terms(
        structure.get("functions", [])
    )

    objects = _clean_terms(
        structure.get("objects", [])
    )

    application_expr = _role_expression(
        application
    )

    core_expr = _role_expression(
        core_system
    )

    sensor_expr = _role_expression(
        sensors
    )

    function_expr = _role_expression(
        functions,
        wildcard=False,
    )

    object_expr = _role_expression(
        objects
    )

    logger.info(
        "SEARCH ROLE EXPRESSIONS | core=%r | sensor=%r | function=%r | object=%r",
        core_expr,
        sensor_expr,
        function_expr,
        object_expr,
    )

    paths = []

    # ------------------------------------------------------------------
    # 0. CORE ONLY (recall-oriented invention identity)
    # ------------------------------------------------------------------
    if core_expr:
        path = {
            "name": "Core Only",
            "roles": ["core_system"],
            "quality_level": "recall",
            "recall_priority": 5,
            "precision_priority": 2,
            "purpose": (
                "Search the invention identity alone to preserve recall "
                "before implementation-level paths are applied."
            ),
            "expression": core_expr,
            "limit": 25,
        }
        paths.append(path)
        logger.info(
            "SEARCH PATH: %s",
            json.dumps(path, indent=2, ensure_ascii=False),
        )

    # ------------------------------------------------------------------
    # 1. SENSOR + FUNCTION
    # ------------------------------------------------------------------
    # Technical search.
    #
    # Purpose:
    #   Good recall of patents using different system terminology
    #   but describing similar technical behaviour.
    # ------------------------------------------------------------------

    if sensor_expr and function_expr:
        path = {
            "name": "Sensor + Function",
            "roles": [
                "sensors",
                "functions",
            ],
            "quality_level": "broad",
            "recall_priority": 4,
            "precision_priority": 4,
            "purpose": (
                "Search the sensing technology together with "
                "the functions it performs."
            ),
            "expression": _combine_roles(
                sensor_expr,
                function_expr,
            ),
        }
        paths.append(path)
        logger.info(
            "SEARCH PATH: %s",
            json.dumps(path, indent=2, ensure_ascii=False),
        )

    # ------------------------------------------------------------------
    # 2. CORE + FUNCTION
    # ------------------------------------------------------------------
    # Broad invention-identity search.
    #
    # Purpose:
    #   High recall.
    #
    # This should normally retrieve a relatively large corpus.
    # ------------------------------------------------------------------

    if core_expr and function_expr:
        path = {
            "name": "Core + Function",
            "roles": [
                "core_system",
                "functions",
            ],
            "quality_level": "broad",
            "recall_priority": 5,
            "precision_priority": 2,
            "purpose": (
                "Broad search connecting the core system with "
                "its primary functions."
            ),
            "expression": _combine_roles(
                core_expr,
                function_expr,
            ),
        }
        paths.append(path)
        logger.info(
            "SEARCH PATH: %s",
            json.dumps(path, indent=2, ensure_ascii=False),
        )

    # ------------------------------------------------------------------
    # 3. CORE + SENSOR
    # ------------------------------------------------------------------
    # Architecture-oriented search.
    #
    # Purpose:
    #   Find documents describing a similar technical architecture.
    # ------------------------------------------------------------------

    if core_expr and sensor_expr:
        path = {
            "name": "Core + Sensor",
            "roles": [
                "core_system",
                "sensors",
            ],
            "quality_level": "balanced",
            "recall_priority": 3,
            "precision_priority": 6,
            "purpose": (
                "Search for the core system together with "
                "the sensing architecture."
            ),
            "expression": _combine_roles(
                core_expr,
                sensor_expr,
            ),
        }
        paths.append(path)
        logger.info(
            "SEARCH PATH: %s",
            json.dumps(path, indent=2, ensure_ascii=False),
        )

    # ------------------------------------------------------------------
    # 4. CORE + FUNCTION + SENSOR
    # ------------------------------------------------------------------
    # High precision search.
    #
    # Purpose:
    #   Strongest combination of invention identity, behaviour,
    #   and technical mechanism.
    # ------------------------------------------------------------------

    if core_expr and function_expr and sensor_expr:
        path = {
            "name": "Core + Function + Sensor",
            "roles": [
                "core_system",
                "functions",
                "sensors",
            ],
            "quality_level": "focused",
            "recall_priority": 2,
            "precision_priority": 10,
            "purpose": (
                "High-precision search requiring the core system, "
                "its function, and the sensing technology together."
            ),
            "expression": (
                f"{core_expr}\n"
                "AND\n"
                f"{function_expr}\n"
                "AND\n"
                f"{sensor_expr}"
            ),
        }
        paths.append(path)
        logger.info(
            "SEARCH PATH: %s",
            json.dumps(path, indent=2, ensure_ascii=False),
        )

    # ------------------------------------------------------------------
    # 5. SENSOR + OBJECT + FUNCTION
    # ------------------------------------------------------------------
    # Implementation/detail search.
    #
    # Purpose:
    #   Find patents describing the same technical operation even
    #   when the overall system terminology differs.
    # ------------------------------------------------------------------

    if sensor_expr and object_expr and function_expr:
        path = {
            "name": "Sensor + Object + Function",
            "roles": [
                "sensors",
                "objects",
                "functions",
            ],
            "quality_level": "focused",
            "recall_priority": 2,
            "precision_priority": 8,
            "purpose": (
                "Search the sensing technology in relation to "
                "specific objects and functions."
            ),
            "expression": (
                f"{sensor_expr}\n"
                "AND\n"
                f"{object_expr}\n"
                "AND\n"
                f"{function_expr}"
            ),
        }
        paths.append(path)
        logger.info(
            "SEARCH PATH: %s",
            json.dumps(path, indent=2, ensure_ascii=False),
        )
    return paths

def generate_invention_structure(
    invention: str,
    client: ClaudeClient,
) -> dict:
    """Generate structured invention roles and search paths."""

    invention = invention.strip()

    if not invention:
        raise ValueError("Invention description cannot be empty.")

    response = client.generate(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=USER_TEMPLATE.format(
            invention=invention,
        ),
    )

    data = _extract_json(response)

    structure = {
        "application": _clean_terms(
            data.get("application", [])
        ),
        "core_system": _clean_terms(
            data.get("core_system", [])
        ),
        "sensors": _clean_terms(
            data.get("sensors", [])
        ),
        "functions": _clean_terms(
            data.get("functions", [])
        ),
        "objects": _clean_terms(
            data.get("objects", [])
        ),
    }

    structure["search_paths"] = build_search_paths_from_structure(
        structure
    )

    logger.info(
        "GENERATED SEARCH PATHS:\n%s",
        json.dumps(
            structure["search_paths"],
            indent=2,
            ensure_ascii=False,
        ),
    )

    logger.info(
        "AI INVENTION STRUCTURE:\n%s",
        json.dumps(
            structure,
            indent=2,
            ensure_ascii=False,
        ),
    )

    return structure
    
