"""AI-assisted invention structure and patent search path generation."""

from __future__ import annotations

import json
import re

from services.claude_client import ClaudeClient


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

sensors:
Sensors, sensing technologies, cameras, detectors, or input devices.

functions:
Actions performed by the invention. Prefer patent-search verbs such as:
detect, determine, identify, recognize, measure, classify, estimate,
monitor, calculate, control.

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
- Functions should be verbs or verb stems suitable for wildcard searching.
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


def build_search_paths_from_structure(
    structure: dict,
) -> list[dict]:
    """
    Build deterministic search paths from the reviewed invention
    structure.

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
        wildcard=True,
    )

    object_expr = _role_expression(
        objects
    )

    paths = []

    def add_path(
        name: str,
        roles: list[str],
        expressions: list[str],
        purpose: str,
    ):
        expressions = [
            expression
            for expression in expressions
            if expression
        ]

        if not expressions:
            return

        paths.append(
            {
                "name": name,
                "roles": roles,
                "expression": "\nAND\n".join(
                    expressions
                ),
                "purpose": purpose,
            }
        )

    # 1. Sensor + Function
    add_path(
        "Sensor + Function",
        ["sensors", "functions"],
        [
            sensor_expr,
            function_expr,
        ],
        (
            "Find patents describing the sensing technology "
            "performing the relevant function."
        ),
    )

    # 2. Core + Function + Sensor
    add_path(
        "Core + Function + Sensor",
        ["core_system", "functions", "sensors"],
        [
            core_expr,
            function_expr,
            sensor_expr,
        ],
        (
            "Find patents combining the core system, "
            "its function, and the sensor technology."
        ),
    )

    # 3. Sensor + Object + Function
    add_path(
        "Sensor + Object + Function",
        ["sensors", "objects", "functions"],
        [
            sensor_expr,
            object_expr,
            function_expr,
        ],
        (
            "Find patents where the sensor performs the "
            "relevant function on the target object."
        ),
    )

    # 4. Core + Sensor + Object
    add_path(
        "Core + Sensor + Object",
        ["core_system", "sensors", "objects"],
        [
            core_expr,
            sensor_expr,
            object_expr,
        ],
        (
            "Find patents combining the core system, "
            "sensor technology, and target object."
        ),
    )

    # 5. Application + Core + Sensor + Object
    add_path(
        "Application + Core + Sensor + Object",
        [
            "application",
            "core_system",
            "sensors",
            "objects",
        ],
        [
            application_expr,
            core_expr,
            sensor_expr,
            object_expr,
        ],
        (
            "Find patents describing the complete "
            "application-specific technical combination."
        ),
    )

    # 6. Function + Object
    add_path(
        "Function + Object",
        ["functions", "objects"],
        [
            function_expr,
            object_expr,
        ],
        (
            "Find patents describing the relevant function "
            "performed on the target object."
        ),
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

    return structure
    
