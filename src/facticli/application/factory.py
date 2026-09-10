from __future__ import annotations

from facticli.adapters import (
    CompatibleClaimExtractionAdapter,
    CompatibleJudgeAdapter,
    CompatiblePlannerAdapter,
    CompatibleResearchAdapter,
    CompatibleReviewAdapter,
    CompatibleSingleAgentAdapter,
    configure_inference_client,
    load_inference_config,
)
from facticli.core.constraints import FACT_CHECK_DOMAINS

from .citations import CitationHealthChecker
from .config import STRATEGIES, ClaimExtractionRuntimeConfig, FactCheckRuntimeConfig
from .repository import RunArtifactRepository
from .services import ClaimExtractionService, FactCheckService, SingleAgentFactCheckService
from .stages import CitationCheckStage, ClaimExtractionStage, JudgeStage, PlanStage, ResearchStage, ReviewStage

FactCheckServiceLike = FactCheckService | SingleAgentFactCheckService


def resolve_stage_models(config: FactCheckRuntimeConfig | ClaimExtractionRuntimeConfig, default_model: str) -> dict[str, str]:
    """Model per stage after applying per-stage overrides to the default."""
    stages = ("plan", "research", "review", "judge", "single_agent", "extract_claims")
    return {stage: config.model_for(stage, default_model) for stage in stages}


def build_fact_check_service(
    config: FactCheckRuntimeConfig,
    artifact_repository: RunArtifactRepository | None = None,
) -> FactCheckServiceLike:
    """Composition root: wire adapters and stages for the configured strategy."""
    if config.strategy not in STRATEGIES:
        raise ValueError(f"Unsupported strategy: {config.strategy!r}; expected one of {', '.join(STRATEGIES)}")

    inference_config = load_inference_config(
        requested_model=config.model,
        base_url=config.base_url,
    )
    configure_inference_client(inference_config)

    models = resolve_stage_models(config, inference_config.model)
    blocked_domains = _resolve_blocked_domains(config)
    citation_stage = (
        CitationCheckStage(checker=CitationHealthChecker(timeout_seconds=config.citation_check_timeout_seconds))
        if config.verify_sources
        else None
    )
    common = dict(
        citation_stage=citation_stage,
        artifact_repository=artifact_repository,
        blocked_domains=blocked_domains,
        knowledge_store_dir=config.knowledge_store_dir,
        token_budget=config.token_budget,
        retry_budget=config.retry_budget,
    )

    if config.strategy == "single_agent":
        checker = CompatibleSingleAgentAdapter(
            model=models["single_agent"],
            max_turns=config.single_agent_max_turns,
            search_context_size=config.search_context_size,
            search_provider=config.search_provider,
            reasoning_effort=config.effort_for("single_agent"),
            retry_attempts=config.model_retry_attempts,
            max_chars_per_field=config.max_chars_per_result_field,
            max_total_chars=config.max_tool_output_chars,
        )
        return SingleAgentFactCheckService(
            checker=checker,
            max_checks=config.max_checks,
            max_search_queries_per_check=config.max_search_queries_per_check,
            stage_models={"single_agent": models["single_agent"]},
            **common,
        )

    planner = CompatiblePlannerAdapter(
        model=models["plan"],
        max_turns=config.max_turns,
        reasoning_effort=config.effort_for("plan"),
        retry_attempts=config.model_retry_attempts,
    )
    researcher = CompatibleResearchAdapter(
        model=models["research"],
        max_turns=config.max_turns,
        search_context_size=config.search_context_size,
        search_provider=config.search_provider,
        reasoning_effort=config.effort_for("research"),
        retry_attempts=config.model_retry_attempts,
        max_chars_per_field=config.max_chars_per_result_field,
        max_total_chars=config.max_tool_output_chars,
    )
    judge = CompatibleJudgeAdapter(
        model=models["judge"],
        max_turns=config.judge_max_turns,
        reasoning_effort=config.effort_for("judge"),
        retry_attempts=config.model_retry_attempts,
    )
    review = CompatibleReviewAdapter(
        model=models["review"],
        max_turns=config.max_turns,
        reasoning_effort=config.effort_for("review"),
        retry_attempts=config.model_retry_attempts,
    )

    return FactCheckService(
        plan_stage=PlanStage(
            planner=planner,
            max_checks=config.max_checks,
            max_search_queries_per_check=config.max_search_queries_per_check,
        ),
        research_stage=ResearchStage(
            researcher=researcher,
            max_parallel_research=config.max_parallel_research,
            research_timeout_seconds=config.research_timeout_seconds,
            research_retry_attempts=config.research_retry_attempts,
        ),
        judge_stage=JudgeStage(judge=judge),
        review_stage=ReviewStage(
            reviewer=review,
            max_follow_up_checks=config.max_follow_up_checks,
            max_search_queries_per_check=config.max_search_queries_per_check,
        ),
        max_feedback_rounds=config.max_feedback_rounds,
        max_follow_up_checks=config.max_follow_up_checks,
        max_search_queries_per_check=config.max_search_queries_per_check,
        stage_models={stage: models[stage] for stage in ("plan", "research", "review", "judge")},
        **common,
    )


def _resolve_blocked_domains(config: FactCheckRuntimeConfig) -> tuple[str, ...]:
    domains: list[str] = []
    if config.block_fact_checkers:
        domains.extend(FACT_CHECK_DOMAINS)
    domains.extend(config.blocked_domains)
    seen: set[str] = set()
    unique: list[str] = []
    for domain in domains:
        normalized = domain.strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(normalized)
    return tuple(unique)


def build_claim_extraction_service(config: ClaimExtractionRuntimeConfig) -> ClaimExtractionService:
    inference_config = load_inference_config(
        requested_model=config.model,
        base_url=config.base_url,
    )
    configure_inference_client(inference_config)

    backend = CompatibleClaimExtractionAdapter(
        model=config.model_for("extract_claims", inference_config.model),
        max_turns=config.max_turns,
        reasoning_effort=config.effort_for("extract_claims"),
        retry_attempts=config.model_retry_attempts,
    )

    return ClaimExtractionService(
        extraction_stage=ClaimExtractionStage(backend=backend, max_claims=config.max_claims)
    )
