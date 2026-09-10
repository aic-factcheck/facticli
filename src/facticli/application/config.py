from __future__ import annotations

from dataclasses import dataclass, field

# Stage names accepted by per-stage model / reasoning-effort routing.
STAGE_NAMES: tuple[str, ...] = ("plan", "research", "review", "judge", "single_agent", "extract_claims")
REASONING_EFFORTS: tuple[str, ...] = ("minimal", "low", "medium", "high")
STRATEGIES: tuple[str, ...] = ("pipeline", "single_agent")


@dataclass(frozen=True)
class InferenceConfig:
    """Shared inference knobs used by runtime service factories.

    ``stage_models`` / ``stage_efforts`` route individual stages to a different
    model or reasoning effort than the default (e.g. a small local model for
    research and a frontier model for the judge). Keys are ``STAGE_NAMES``.
    ``model_retry_attempts`` is the per-call cap for SDK-managed retries of
    transient provider errors; ``retry_budget`` caps retries across the run.
    """
    model: str | None = None
    base_url: str | None = None
    max_turns: int = 10
    stage_models: dict[str, str] = field(default_factory=dict)
    stage_efforts: dict[str, str] = field(default_factory=dict)
    model_retry_attempts: int = 3
    retry_budget: int | None = 12

    def model_for(self, stage: str, default: str) -> str:
        return self.stage_models.get(stage) or default

    def effort_for(self, stage: str) -> str | None:
        return self.stage_efforts.get(stage)


@dataclass(frozen=True)
class FactCheckRuntimeConfig(InferenceConfig):
    """Runtime configuration for full fact-check orchestration."""
    strategy: str = "pipeline"
    max_checks: int = 4
    max_parallel_research: int = 4
    max_feedback_rounds: int = 0
    max_follow_up_checks: int = 2
    search_context_size: str = "high"
    search_provider: str = "openai"
    search_results_per_query: int = 5
    max_search_queries_per_check: int = 5
    judge_max_turns: int = 12
    single_agent_max_turns: int = 24
    research_timeout_seconds: float = 120.0
    research_retry_attempts: int = 1
    token_budget: int | None = None
    verify_sources: bool = False
    citation_check_timeout_seconds: float = 8.0
    max_tool_output_chars: int = 12000
    max_chars_per_result_field: int = 1500
    block_fact_checkers: bool = False
    blocked_domains: tuple[str, ...] = ()
    knowledge_store_dir: str | None = None


@dataclass(frozen=True)
class ClaimExtractionRuntimeConfig(InferenceConfig):
    """Runtime configuration for standalone claim extraction flows."""
    max_claims: int = 12


def parse_stage_assignments(
    raw_values: list[str] | tuple[str, ...] | None,
    *,
    allowed_values: tuple[str, ...] | None = None,
    option_name: str,
) -> dict[str, str]:
    """Parse repeatable ``STAGE=VALUE`` CLI/config entries into a mapping.

    Raises ``ValueError`` on unknown stages, malformed entries, or (when
    ``allowed_values`` is given) values outside the allowed set.
    """
    assignments: dict[str, str] = {}
    for raw in raw_values or ():
        if "=" not in raw:
            raise ValueError(f"{option_name} expects STAGE=VALUE, got: {raw!r}")
        stage, value = raw.split("=", 1)
        stage = stage.strip().lower()
        value = value.strip()
        if stage not in STAGE_NAMES:
            raise ValueError(f"{option_name}: unknown stage {stage!r}; expected one of {', '.join(STAGE_NAMES)}")
        if not value:
            raise ValueError(f"{option_name}: empty value for stage {stage!r}")
        if allowed_values is not None and value.lower() not in allowed_values:
            raise ValueError(
                f"{option_name}: invalid value {value!r} for stage {stage!r}; expected one of {', '.join(allowed_values)}"
            )
        assignments[stage] = value.lower() if allowed_values is not None else value
    return assignments
