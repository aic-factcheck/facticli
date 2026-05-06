from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlparse

from agents import set_default_openai_api, set_default_openai_client, set_tracing_disabled
from openai import AsyncOpenAI

ApiMode = Literal["chat_completions", "responses"]


@dataclass(frozen=True)
class InferenceConfig:
    """OpenAI-compatible inference configuration.

    All inference backends are configured as OpenAI-compatible endpoints.
    Configuration is reduced to four knobs:

    - api_key: Authentication token
    - model: Model identifier
    - base_url: Optional endpoint override (None = OpenAI SDK default)
    - api_mode: Either "chat_completions" or "responses"
    """

    api_key: str
    model: str
    base_url: str | None
    api_mode: ApiMode


def _normalize_optional(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def infer_api_mode(base_url: str | None) -> ApiMode:
    """Choose the Agents SDK API mode for an OpenAI-compatible endpoint."""
    if not base_url:
        return "responses"

    parsed = urlparse(base_url)
    host = parsed.netloc.lower()
    if host == "api.openai.com" or host.endswith(".api.openai.com"):
        return "responses"
    return "chat_completions"


def load_inference_config(
    *,
    requested_model: str | None,
    base_url: str | None,
) -> InferenceConfig:
    """Load inference configuration from CLI args and environment.

    Resolution order:
    - API key: OPENAI_API_KEY
    - model: CLI --model > OPENAI_API_MODEL
    - base URL: CLI --base-url > OPENAI_API_BASE_URL > OpenAI SDK default
    """
    api_key = _normalize_optional(os.getenv("OPENAI_API_KEY"))
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    model = _normalize_optional(requested_model) or _normalize_optional(
        os.getenv("OPENAI_API_MODEL")
    )
    if not model:
        raise RuntimeError("OPENAI_API_MODEL is not set. Export it or pass --model.")

    resolved_base_url = _normalize_optional(base_url) or _normalize_optional(
        os.getenv("OPENAI_API_BASE_URL")
    )

    return InferenceConfig(
        api_key=api_key,
        model=model,
        base_url=resolved_base_url,
        api_mode=infer_api_mode(resolved_base_url),
    )


def configure_inference_client(config: InferenceConfig) -> None:
    """Configure the Agents SDK with the given inference configuration."""
    client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)

    set_default_openai_client(client, use_for_tracing=False)
    set_default_openai_api(config.api_mode)
    set_tracing_disabled(True)
