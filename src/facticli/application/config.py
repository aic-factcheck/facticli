from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InferenceConfig:
    inference_provider: str = "openai"
    model: str = "gpt-4.1-mini"
    base_url: str | None = None
    max_turns: int = 10


@dataclass(frozen=True)
class FactCheckRuntimeConfig(InferenceConfig):
    max_checks: int = 4
    max_parallel_research: int = 4
    max_feedback_rounds: int = 0
    max_follow_up_checks: int = 2
    search_context_size: str = "high"
    search_provider: str = "openai"
    search_results_per_query: int = 5
    max_search_queries_per_check: int = 5
    judge_max_turns: int = 12
    research_timeout_seconds: float = 120.0
    research_retry_attempts: int = 1


@dataclass(frozen=True)
class ClaimExtractionRuntimeConfig(InferenceConfig):
    max_claims: int = 12
