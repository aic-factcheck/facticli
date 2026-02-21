from __future__ import annotations

from facticli.adapters import (
    CompatibleClaimExtractionAdapter,
    CompatibleJudgeAdapter,
    CompatiblePlannerAdapter,
    CompatibleResearchAdapter,
    configure_openai_compatible_client,
)

from .config import ClaimExtractionRuntimeConfig, FactCheckRuntimeConfig
from .repository import RunArtifactRepository
from .services import ClaimExtractionService, FactCheckService
from .stages import ClaimExtractionStage, JudgeStage, PlanStage, ResearchStage


def build_fact_check_service(
    config: FactCheckRuntimeConfig,
    artifact_repository: RunArtifactRepository | None = None,
) -> FactCheckService:
    configure_openai_compatible_client(
        inference_provider=config.inference_provider,
        base_url=config.base_url,
    )

    planner = CompatiblePlannerAdapter(model=config.model, max_turns=config.max_turns)
    researcher = CompatibleResearchAdapter(
        model=config.model,
        max_turns=config.max_turns,
        search_context_size=config.search_context_size,
        search_provider=config.search_provider,
    )
    judge = CompatibleJudgeAdapter(
        model=config.model,
        max_turns=config.max_turns,
        judge_extra_turns=config.judge_extra_turns,
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
        artifact_repository=artifact_repository,
    )


def build_claim_extraction_service(config: ClaimExtractionRuntimeConfig) -> ClaimExtractionService:
    configure_openai_compatible_client(
        inference_provider=config.inference_provider,
        base_url=config.base_url,
    )
    backend = CompatibleClaimExtractionAdapter(model=config.model, max_turns=config.max_turns)

    return ClaimExtractionService(
        extraction_stage=ClaimExtractionStage(backend=backend, max_claims=config.max_claims)
    )
