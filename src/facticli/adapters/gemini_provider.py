from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeVar

from pydantic import BaseModel

from facticli.application.interfaces import ClaimExtractionBackend, Judge, Planner, Researcher, Retriever
from facticli.core.contracts import (
    AspectFinding,
    ClaimExtractionResult,
    FactCheckReport,
    InvestigationPlan,
    VerificationCheck,
)
from facticli.core.normalize import normalize_query_list
from facticli.gemini_inference import GeminiStructuredClient
from facticli.skills import load_skill_prompt

TModel = TypeVar("TModel", bound=BaseModel)


@dataclass(frozen=True)
class _GeminiStructuredRunner:
    client: GeminiStructuredClient

    async def run_structured(
        self,
        *,
        skill_name: str,
        payload: dict[str, Any],
        output_model: type[TModel],
    ) -> TModel:
        instructions = load_skill_prompt(skill_name)
        return await self.client.generate_structured(
            instructions=instructions,
            payload=payload,
            output_model=output_model,
        )


class GeminiPlannerAdapter(Planner):
    def __init__(self, model: str):
        self._runner = _GeminiStructuredRunner(client=GeminiStructuredClient(model=model))

    async def plan(self, claim: str, max_checks: int) -> InvestigationPlan:
        return await self._runner.run_structured(
            skill_name="plan",
            payload={"claim": claim, "max_checks": max_checks},
            output_model=InvestigationPlan,
        )


class GeminiResearchAdapter(Researcher):
    def __init__(
        self,
        model: str,
        retriever: Retriever,
        results_per_query: int,
        max_search_queries_per_check: int,
    ):
        self._runner = _GeminiStructuredRunner(client=GeminiStructuredClient(model=model))
        self._retriever = retriever
        self._results_per_query = results_per_query
        self._max_search_queries_per_check = max_search_queries_per_check

    async def research(self, claim: str, check: VerificationCheck) -> AspectFinding:
        queries = normalize_query_list(
            check.search_queries,
            fallback=[check.question, claim],
            max_queries=self._max_search_queries_per_check,
        )
        search_results = await self._retriever.search(queries, self._results_per_query)
        payload = {
            "claim": claim,
            "check": check.model_dump(),
            "requirements": {
                "min_sources": 2,
                "must_use_search_tool": False,
                "preferred_provider": "brave",
            },
            "search_results": search_results,
        }
        finding = await self._runner.run_structured(
            skill_name="research_gemini",
            payload=payload,
            output_model=AspectFinding,
        )

        updates: dict[str, str] = {}
        if not finding.aspect_id.strip():
            updates["aspect_id"] = check.aspect_id
        if not finding.question.strip():
            updates["question"] = check.question
        return finding.model_copy(update=updates) if updates else finding


class GeminiJudgeAdapter(Judge):
    def __init__(self, model: str):
        self._runner = _GeminiStructuredRunner(client=GeminiStructuredClient(model=model))

    async def judge(
        self,
        claim: str,
        plan: InvestigationPlan,
        findings: list[AspectFinding],
    ) -> FactCheckReport:
        payload = {
            "claim": claim,
            "plan": plan.model_dump(),
            "findings": [finding.model_dump() for finding in findings],
        }
        return await self._runner.run_structured(
            skill_name="judge",
            payload=payload,
            output_model=FactCheckReport,
        )


class GeminiClaimExtractionAdapter(ClaimExtractionBackend):
    def __init__(self, model: str):
        self._runner = _GeminiStructuredRunner(client=GeminiStructuredClient(model=model))

    async def extract(self, input_text: str, max_claims: int) -> ClaimExtractionResult:
        payload = {
            "input_text": input_text,
            "requirements": {
                "max_claims": max_claims,
                "decontextualized": True,
                "atomic_claims": True,
                "maximize_checkworthy_coverage": True,
                "only_directly_mentioned_facts": True,
            },
        }
        return await self._runner.run_structured(
            skill_name="extract_claims",
            payload=payload,
            output_model=ClaimExtractionResult,
        )
