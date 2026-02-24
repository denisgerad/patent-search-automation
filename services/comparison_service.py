"""
services/comparison_service.py

Sends the top-ranked patents to Claude for structured prior-art analysis.
The system prompt is loaded from prompts/comparison_system.txt.
To change the analysis instructions, edit that file — do not modify this service.
"""
import logging
import sys
from pathlib import Path
from typing import Any

# Ensure the project root is importable regardless of the launch directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.claude_client import ClaudeClient    # noqa: E402
from utils.prompt_loader import load_prompt      # noqa: E402

logger = logging.getLogger(__name__)

# Load the system prompt once at module import time.
# Template file: prompts/comparison_system.txt
# This is a static system prompt — no placeholders needed.
SYSTEM_PROMPT: str = load_prompt("comparison_system.txt")

# Maximum patents to include in a single comparison call.
# Claude's context window is large, but keep this bounded to control cost/latency.
# Change this value here if you need deeper analysis.
MAX_PATENTS_TO_COMPARE = 10


def _format_patents_for_prompt(patents: list[Any]) -> str:
    """
    Convert a list of RankedPatent objects (or dicts) into a text block
    suitable for inclusion in the user message sent to Claude.
    """
    lines: list[str] = []
    for item in patents:
        # Support both RankedPatent schema objects and plain dicts.
        if hasattr(item, "patent"):
            p = item.patent
            pid    = getattr(p, "patent_id", "unknown")
            title  = getattr(p, "patent_title", "N/A")
            abstract = getattr(p, "patent_abstract", None) or "N/A"
        else:
            pid    = item.get("patent_id", "unknown")
            title  = item.get("patent_title", "N/A")
            abstract = item.get("patent_abstract") or "N/A"

        lines.append(f"Patent ID: {pid}")
        lines.append(f"Title: {title}")
        lines.append(f"Abstract: {abstract}")
        lines.append("---")

    return "\n".join(lines)


def compare_patents(query: str, patents: list[Any], client: ClaudeClient) -> str:
    """
    Analyse the top patents against *query* and return a JSON string.

    The returned string is a JSON array as specified in prompts/comparison_system.txt,
    with fields: patent_id, overlap_level, key_overlapping_claims, differentiation_notes.

    Args:
        query:   The original (or expanded) search query.
        patents: Ranked patent objects; only the first MAX_PATENTS_TO_COMPARE are used.
        client:  An initialised ClaudeClient instance.

    Returns:
        Raw JSON string from Claude (parse in the caller where needed).
    """
    subset = patents[:MAX_PATENTS_TO_COMPARE]
    patent_text = _format_patents_for_prompt(subset)

    user_message = f"Query: {query}\n\nPatents:\n{patent_text}"

    logger.info(
        "Sending %d patent(s) to Claude for comparison (query='%s')",
        len(subset), query,
    )

    result = client.complete(system=SYSTEM_PROMPT, user=user_message)
    logger.debug("Comparison result received (%d chars)", len(result))
    return result
