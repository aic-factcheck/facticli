from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InferenceConfig:
    """Shared inference provider settings for all pipeline stages."""

    inference_provider: str = "openai-agents"
    model: str = "gpt-4.1-mini"
    gemini_model: str = "gemini-2.0-flash"
    max_turns: int = 10
