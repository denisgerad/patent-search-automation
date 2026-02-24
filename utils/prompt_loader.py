"""
utils/prompt_loader.py

Single place to load prompt templates from the prompts/ directory.
To change a prompt, edit the corresponding .txt file in prompts/ —
never hardcode prompt text in service files.
"""
from pathlib import Path

# All prompt templates live here.
# Change this constant if you ever relocate the prompts/ folder.
PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"


def load_prompt(filename: str) -> str:
    """
    Load a prompt template by filename.

    Args:
        filename: Name of the file inside prompts/, e.g. "query_expansion.txt".
                  A ".txt" extension is appended automatically if omitted.

    Returns:
        The file contents as a string (UTF-8).

    Raises:
        FileNotFoundError: If the prompt file does not exist in PROMPTS_DIR.
    """
    path = PROMPTS_DIR / filename
    # Auto-append .txt if no extension was given
    if not path.suffix:
        path = path.with_suffix(".txt")

    if not path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {path}\n"
            f"Expected location: {PROMPTS_DIR}/<filename>.txt"
        )

    return path.read_text(encoding="utf-8")
