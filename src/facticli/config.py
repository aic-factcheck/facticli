from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InferenceConfig:
    """Shared inference provider settings for all pipeline stages."""

    inference_provider: str = "openai"
    model: str = "gpt-4.1-mini"
    base_url: str | None = None
    max_turns: int = 10
