"""
models/mistral_client.py

Thin wrapper around Ollama for local Mistral inference.
Responsibility: call the model, return a string.
No prompt logic lives here — prompts are owned by the calling service.
"""
import ollama


class MistralClient:
    """Calls a locally-running Mistral model via Ollama."""

    def __init__(self, model: str = "mistral"):
        # Model name must match a model pulled with `ollama pull <model>`.
        # Override by passing a different name, e.g. "mistral:7b-instruct".
        self.model = model

    def generate(self, prompt: str) -> str:
        """Send a prompt and return the model's response as a plain string."""
        response = ollama.generate(model=self.model, prompt=prompt)
        return response["response"].strip()
