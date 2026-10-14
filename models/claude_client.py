"""
Thin wrapper around the Anthropic SDK.
Prompts remain owned by the calling services.
"""

try:
    import anthropic
except ModuleNotFoundError:  # pragma: no cover - handled at runtime.
    anthropic = None

from app.config import settings


class ClaudeClient:
    """Calls Anthropic Claude and records API-reported token usage."""

    _FALLBACK_MODELS = (
        "claude-sonnet-4-6",
        "claude-sonnet-5",
        "claude-opus-4-8",
    )

    def __init__(self, model: str | None = None, max_tokens: int = 4096):
        if anthropic is None:
            raise ModuleNotFoundError(
                "The 'anthropic' package is required to use ClaudeClient. "
                "Install project dependencies first."
            )

        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.model = (
            model or settings.anthropic_model or self._FALLBACK_MODELS[0]
        ).strip()
        self.max_tokens = max_tokens

        self.usage_stage = "Unlabelled"
        self.usage_events: list[dict] = []

    def reset_usage(self) -> None:
        """Start a fresh usage ledger for one workflow run."""
        self.usage_events = []
        self.usage_stage = "Unlabelled"

    def set_usage_stage(self, stage: str) -> None:
        """Label subsequent API calls for reporting."""
        self.usage_stage = stage.strip() or "Unlabelled"

    def get_usage_summary(self) -> dict:
        """Return total token usage and a breakdown by workflow stage."""
        by_stage: dict[str, dict] = {}

        for event in self.usage_events:
            stage = event["stage"]
            totals = by_stage.setdefault(
                stage,
                {
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                },
            )
            totals["calls"] += 1
            totals["input_tokens"] += event["input_tokens"]
            totals["output_tokens"] += event["output_tokens"]
            totals["total_tokens"] += event["total_tokens"]

        input_tokens = sum(
            event["input_tokens"] for event in self.usage_events
        )
        output_tokens = sum(
            event["output_tokens"] for event in self.usage_events
        )

        return {
            "calls": len(self.usage_events),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "by_stage": by_stage,
            "events": list(self.usage_events),
        }

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
    ) -> str:
        """Compatibility wrapper for callers expecting generate()."""
        return self.complete(
            system=system_prompt,
            user=user_prompt,
            max_tokens=max_tokens,
        )

    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int | None = None,
    ) -> str:
        """Return the assistant's text and record successful-call usage."""
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

                usage = getattr(msg, "usage", None)
                input_tokens = int(
                    getattr(usage, "input_tokens", 0) or 0
                )
                output_tokens = int(
                    getattr(usage, "output_tokens", 0) or 0
                )

                self.usage_events.append(
                    {
                        "stage": self.usage_stage,
                        "model": model_name,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens,
                        "cache_read_input_tokens": int(
                            getattr(
                                usage, "cache_read_input_tokens", 0
                            ) or 0
                        ),
                        "cache_creation_input_tokens": int(
                            getattr(
                                usage, "cache_creation_input_tokens", 0
                            ) or 0
                        ),
                    }
                )

                return msg.content[0].text

            except anthropic.NotFoundError as exc:
                last_error = exc
                continue

        if last_error is not None:
            raise last_error

        raise RuntimeError("No Anthropic model candidates were available.")
