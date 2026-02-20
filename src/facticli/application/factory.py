from __future__ import annotations

from facticli.adapters import (
    BraveSearchRetriever,
    GeminiClaimExtractionAdapter,
    GeminiJudgeAdapter,
    GeminiPlannerAdapter,
    GeminiResearchAdapter,
    OpenAIClaimExtractionAdapter,
    OpenAIJudgeAdapter,
    OpenAIPlannerAdapter,
    OpenAIResearchAdapter,
)

from .config import ClaimExtractionRuntimeConfig, FactCheckRuntimeConfig
from .repository import RunArtifactRepository
from .services import ClaimExtractionService, FactCheckService
from .stages import ClaimExtractionStage, JudgeStage, PlanStage, ResearchStage


def build_fact_check_service(
    config: FactCheckRuntimeConfig,
    artifact_repository: RunArtifactRepository | None = None,
) -> FactCheckService:
    if config.inference_provider == "openai-agents":
        planner = OpenAIPlannerAdapter(model=config.model, max_turns=config.max_turns)
        researcher = OpenAIResearchAdapter(
            model=config.model,
            max_turns=config.max_turns,
            search_context_size=config.search_context_size,
            search_provider=config.search_provider,
        )
        judge = OpenAIJudgeAdapter(
            model=config.model,
            max_turns=config.max_turns,
            judge_extra_turns=config.judge_extra_turns,
        )
    elif config.inference_provider == "gemini":
        if config.search_provider != "brave":
            raise ValueError("Gemini inference currently supports search_provider='brave' only.")
        retriever = BraveSearchRetriever()
        planner = GeminiPlannerAdapter(model=config.gemini_model)
        researcher = GeminiResearchAdapter(
            model=config.gemini_model,
            retriever=retriever,
            results_per_query=config.search_results_per_query,
            max_search_queries_per_check=config.max_search_queries_per_check,
        )
        judge = GeminiJudgeAdapter(model=config.gemini_model)
    else:
        raise ValueError(f"Unsupported inference provider: {config.inference_provider}")

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
    if config.inference_provider == "openai-agents":
        backend = OpenAIClaimExtractionAdapter(model=config.model, max_turns=config.max_turns)
    elif config.inference_provider == "gemini":
        backend = GeminiClaimExtractionAdapter(model=config.gemini_model)
    else:
        raise ValueError(f"Unsupported inference provider: {config.inference_provider}")

    return ClaimExtractionService(
        extraction_stage=ClaimExtractionStage(backend=backend, max_claims=config.max_claims)
    )
