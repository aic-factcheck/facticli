from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from agents import set_default_openai_api, set_default_openai_client, set_tracing_disabled
from openai import AsyncOpenAI

ProviderName = Literal["openai", "gemini", "ollama"]


@dataclass(frozen=True)
class ProviderProfile:
    """Provider defaults used to bootstrap OpenAI-compatible inference clients."""

    name: ProviderName
    api_key_env: str
    model_env: str
    base_url_env: str | None
    default_model: str
    default_base_url: str | None
    default_api_mode: Literal["chat_completions", "responses"]


_PROFILE_BY_NAME: dict[str, ProviderProfile] = {
    "openai": ProviderProfile(
        name="openai",
        api_key_env="OPENAI_API_KEY",
        model_env="FACTICLI_MODEL",
        base_url_env="FACTICLI_BASE_URL",
        default_model="gpt-5.4",
        default_base_url=None,
        default_api_mode="responses",
    ),
    "openai-agents": ProviderProfile(
        name="openai",
        api_key_env="OPENAI_API_KEY",
        model_env="FACTICLI_MODEL",
        base_url_env="FACTICLI_BASE_URL",
        default_model="gpt-5.4",
        default_base_url=None,
        default_api_mode="responses",
    ),
    "gemini": ProviderProfile(
        name="gemini",
        api_key_env="GEMINI_API_KEY",
        model_env="FACTICLI_GEMINI_MODEL",
        base_url_env="FACTICLI_BASE_URL",
        default_model="gemini-3.1-pro",
        default_base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        default_api_mode="chat_completions",
    ),
    "ollama": ProviderProfile(
        name="ollama",
        api_key_env="OLLAMA_API_KEY",
        model_env="OLLAMA_MODEL",
        base_url_env="OLLAMA_BASE_URL",
        default_model="qwen3:latest",
        default_base_url=None,
        default_api_mode="chat_completions",
    ),
}


def resolve_provider_profile(inference_provider: str) -> ProviderProfile:
    """Map CLI provider alias to a concrete provider profile."""
    profile = _PROFILE_BY_NAME.get(inference_provider)
    if profile is None:
        raise ValueError(f"Unsupported inference provider: {inference_provider}")
    return profile


def resolve_model_name(inference_provider: str, requested_model: str | None) -> str:
    """Pick a model from explicit flag, then provider-specific environment defaults."""
    if requested_model and requested_model.strip():
        return requested_model.strip()

    profile = resolve_provider_profile(inference_provider)
    return os.getenv(profile.model_env, profile.default_model)


def configure_openai_compatible_client(
    *,
    inference_provider: str,
    base_url: str | None = None,
) -> None:
    """Configure the Agents SDK to use the selected OpenAI-compatible backend."""
    profile = resolve_provider_profile(inference_provider)
    api_key = os.getenv(profile.api_key_env)
    if not api_key:
        raise RuntimeError(f"{profile.api_key_env} is not set.")

    env_base_url = os.getenv(profile.base_url_env) if profile.base_url_env else None
    resolved_base_url = (
        base_url.strip()
        if base_url and base_url.strip()
        else env_base_url.strip()
        if env_base_url and env_base_url.strip()
        else profile.default_base_url
    )
    client = AsyncOpenAI(api_key=api_key, base_url=resolved_base_url)

    set_default_openai_client(client, use_for_tracing=False)
    set_default_openai_api(profile.default_api_mode)
    set_tracing_disabled(True)
