"""
services/report_service.py

Generates a Markdown patent intelligence report using Claude.
The system prompt is loaded from prompts/report_system.txt.
To change the report structure or tone, edit that file — do not modify this service.
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
# Template file: prompts/report_system.txt
# This is a static system prompt — no placeholders needed.
SYSTEM_PROMPT: str = load_prompt("report_system.txt")


def generate(
    query: str,
    ranked_patents: list[Any],
    comparison: str,
    client: ClaudeClient,
) -> str:
    """
    Generate a Markdown patent intelligence report.

    The report follows the four-section structure defined in
    prompts/report_system.txt:
      - Executive Summary
      - Detailed Claim Analysis
      - Risk Assessment
      - Recommended Actions

    Args:
        query:           The original search query (used as context for the report).
        ranked_patents:  The ranked patent list (used only for metadata/count logging).
        comparison:      The JSON comparison string produced by comparison_service.
        client:          An initialised ClaudeClient instance.

    Returns:
        A Markdown string ready to be saved to data/processed/ or returned via the API.
    """
    user_message = (
        f"Query: {query}\n\n"
        f"Comparison Analysis:\n{comparison}"
    )

    logger.info(
        "Generating report for query='%s' based on %d patent(s)",
        query, len(ranked_patents),
    )

    report = client.complete(system=SYSTEM_PROMPT, user=user_message)
    logger.debug("Report generated (%d chars)", len(report))
    return report
