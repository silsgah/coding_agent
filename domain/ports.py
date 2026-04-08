"""
Domain ports (interfaces) — defines what the application layer needs
from infrastructure, without depending on any concrete implementation.
"""

from __future__ import annotations

from typing import Protocol, AsyncIterator, runtime_checkable


@runtime_checkable
class LLMPort(Protocol):
    """Abstract interface for any LLM backend (HF Transformers, vLLM, etc.)."""

    def load(self) -> None:
        """Load model weights. Must be called before generate()."""
        ...

    def is_ready(self) -> bool:
        """Return True if the model is loaded and ready for inference."""
        ...

    async def generate(
        self, prompt: str, max_tokens: int | None = None
    ) -> AsyncIterator[str]:
        """
        Stream generated text chunks for the given prompt.

        Yields:
            str: incremental text chunks as they are generated.
        """
        ...
