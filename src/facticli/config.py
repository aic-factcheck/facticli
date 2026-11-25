"""Runtime configuration for fact-check runs and claim extraction."""

from __future__ import annotations

from dataclasses import dataclass, field

# Stage names accepted by per-stage model / reasoning-effort routing.
STAGE_NAMES: tuple[str, ...] = ("plan", "research", "judge", "extract_claims")
# Superset of what reasoning models accept. Which values a given model honours
# is the provider's call; an unsupported one fails the request (bad_request).
REASONING_EFFORTS: tuple[str, ...] = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
# The search tools refuse calls beyond this multiple of the budget a researcher
# is told to plan for. Models run a call or two past a stated budget, and
# refusing those costs evidence (measurements in TODO.md), so the cap only
# stops runaway loops.
SEARCH_CAP_FACTOR = 2
SERVICE_TIERS: tuple[str, ...] = ("auto", "default", "flex", "priority")
FLEX_RESEARCH_TIMEOUT_SECONDS = 600.0


@dataclass(frozen=True)
class InferenceConfig:
    """Model, endpoint, and per-stage routing shared by every run.

    ``stage_models`` / ``stage_efforts`` route individual stages to a different
    model or reasoning effort than the default; keys are ``STAGE_NAMES``.
    ``model_retry_attempts`` is how often the OpenAI client retries a
    transient error (429, 5xx, timeout, connection) of one request.
    """
    model: str | None = None
    base_url: str | None = None
    max_turns: int = 10
    stage_models: dict[str, str] = field(default_factory=dict)
    stage_efforts: dict[str, str] = field(default_factory=dict)
    model_retry_attempts: int = 3
    # OpenAI processing tier sent with every model call ("flex": half price,
    # slower, occasionally HTTP 429 resource-unavailable, which is retried).
    service_tier: str | None = None

    def model_for(self, stage: str, default: str) -> str:
        return self.stage_models.get(stage) or default

    def effort_for(self, stage: str) -> str | None:
        return self.stage_efforts.get(stage)


@dataclass(frozen=True)
class FactCheckRuntimeConfig(InferenceConfig):
    """Configuration of one plan -> research -> judge run."""
    max_checks: int = 4
    max_parallel_research: int = 4
    search_context_size: str = "high"
    search_provider: str = "openai"
    search_results_per_query: int = 5
    max_search_queries_per_check: int = 5
    research_timeout_seconds: float = 120.0
    research_retry_attempts: int = 1
    max_tool_output_chars: int = 12000
    max_chars_per_result_field: int = 1500
    block_fact_checkers: bool = False
    blocked_domains: tuple[str, ...] = ()
    knowledge_store_dir: str | None = None
    tool_call_budget: int = 8
    search_cache_dir: str | None = None

    @property
    def effective_research_timeout_seconds(self) -> float:
        """Flex requests queue on the provider side, so one research check gets more wall time."""
        if self.service_tier == "flex":
            return max(self.research_timeout_seconds, FLEX_RESEARCH_TIMEOUT_SECONDS)
        return self.research_timeout_seconds


@dataclass(frozen=True)
class ClaimExtractionRuntimeConfig(InferenceConfig):
    """Configuration of standalone claim extraction."""
    max_claims: int = 12


def parse_stage_assignments(
    raw_values: list[str] | tuple[str, ...] | None,
    *,
    allowed_values: tuple[str, ...] | None = None,
    option_name: str,
) -> dict[str, str]:
    """Parse repeatable ``STAGE=VALUE`` entries into a mapping.

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
