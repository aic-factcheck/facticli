from __future__ import annotations

from dataclasses import dataclass

from facticli.config import InferenceConfig


@dataclass(frozen=True)
class FactCheckRuntimeConfig(InferenceConfig):
    max_checks: int = 4
    max_parallel_research: int = 4
    search_context_size: str = "high"
    search_provider: str = "openai"
    search_results_per_query: int = 5
    max_search_queries_per_check: int = 5
    judge_extra_turns: int = 2
    research_timeout_seconds: float = 120.0
    research_retry_attempts: int = 1


@dataclass(frozen=True)
class ClaimExtractionRuntimeConfig(InferenceConfig):
    max_claims: int = 12
