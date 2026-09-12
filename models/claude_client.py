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

    _FALLBACK_MODELS = (
        "claude-sonnet-4-6",
        "claude-sonnet-5",
        "claude-opus-4-8",
    )

    def __init__(self, model: str | None = None, max_tokens: int = 4096):
        # Pass the key explicitly from settings so it works regardless of whether
        # ANTHROPIC_API_KEY is set as an OS environment variable.
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.model = (model or settings.anthropic_model or self._FALLBACK_MODELS[0]).strip()
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
        candidates = []
        seen = set()
        for name in [self.model, *self._FALLBACK_MODELS]:
            if name and name not in seen:
                candidates.append(name)
                seen.add(name)

        last_error = None
        for model_name in candidates:
            try:
                msg = self.client.messages.create(
                    model=model_name,
                    max_tokens=max_tokens or self.max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                self.model = model_name
                return msg.content[0].text
            except anthropic.NotFoundError as exc:
                last_error = exc
                continue

        if last_error is not None:
            raise last_error
        raise RuntimeError("No Anthropic model candidates were available.")
