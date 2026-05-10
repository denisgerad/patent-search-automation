"""
models/claude_client.py

Thin wrapper around the Anthropic SDK.
Responsibility: call the model, return a string.
No prompt logic lives here — prompts are owned by the calling service.
"""
import anthropic

from app.config import settings


class ClaudeClient:
    """Calls Anthropic Claude via the official SDK."""

    def __init__(self, model: str = "claude-sonnet-4-20250514", max_tokens: int = 4096):
        # Pass the key explicitly from settings so it works regardless of whether
        # ANTHROPIC_API_KEY is set as an OS environment variable.
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.model = model
        self.max_tokens = max_tokens

    def complete(self, system: str, user: str, max_tokens: int | None = None) -> str:
        """
        Send a system + user message and return the assistant's text.

        Args:
            system:     System prompt string (loaded from a .txt file by the caller).
            user:       User message string (assembled by the caller from live data).
            max_tokens: Override the instance default when needed.

        Returns:
            The model's reply as a plain string.
        """
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens or self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return msg.content[0].text
