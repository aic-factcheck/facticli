from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from agents import Runner

from .agents import build_judge_agent, build_planner_agent, build_research_agent
from .brave_search import run_brave_web_search
from .gemini_inference import GeminiStructuredClient
from .skills import load_skill_prompt
from .types import (
    AspectFinding,
    EvidenceSignal,
    FactCheckReport,
    InvestigationPlan,
    SourceEvidence,
    VerificationCheck,
)


@dataclass(frozen=True)
class OrchestratorConfig:
    inference_provider: str = "openai-agents"
    model: str = "gpt-4.1-mini"
    gemini_model: str = "gemini-3-pro"
    max_checks: int = 4
    max_parallel_research: int = 4
    search_context_size: str = "high"
    search_provider: str = "openai"
    search_results_per_query: int = 5
    max_turns: int = 10


@dataclass
class FactCheckRun:
    claim: str
    plan: InvestigationPlan
    findings: list[AspectFinding]
    report: FactCheckReport


class FactCheckOrchestrator:
    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.gemini_client: GeminiStructuredClient | None = None
        self.planner_agent = None
        self.research_agent = None
        self.judge_agent = None

        if config.inference_provider == "openai-agents":
            self.planner_agent = build_planner_agent(model=config.model)
            self.research_agent = build_research_agent(
                model=config.model,
                search_context_size=config.search_context_size,
                search_provider=config.search_provider,
            )
            self.judge_agent = build_judge_agent(model=config.model)
        elif config.inference_provider == "gemini":
            self.gemini_client = GeminiStructuredClient(model=config.gemini_model)
        else:
            raise ValueError(f"Unsupported inference provider: {config.inference_provider}")

    async def check_claim(self, claim: str) -> FactCheckRun:
        normalized_claim = claim.strip()
        if not normalized_claim:
            raise ValueError("Claim is empty.")

        plan = await self._plan_claim(normalized_claim)
        findings = await self._run_parallel_research(normalized_claim, plan)
        report = await self._judge_claim(normalized_claim, plan, findings)

        if not report.findings:
            report.findings = findings
        report.sources = self._merge_sources(report.sources, findings)
        report.claim = normalized_claim

        return FactCheckRun(
            claim=normalized_claim,
            plan=plan,
            findings=findings,
            report=report,
        )

    async def _plan_claim(self, claim: str) -> InvestigationPlan:
        if self.config.inference_provider == "openai-agents":
            if self.planner_agent is None:
                raise RuntimeError("Planner agent is not initialized.")

            plan_input = (
                "Build a fact-checking plan for this claim.\n\n"
                f"Claim:\n{claim}\n\n"
                "Output at most 6 checks."
            )
            result = await Runner.run(
                self.planner_agent,
                plan_input,
                max_turns=self.config.max_turns,
            )
            plan = result.final_output_as(InvestigationPlan, raise_if_incorrect_type=True)
        else:
            plan = await self._run_gemini_structured(
                skill_name="plan",
                payload={
                    "claim": claim,
                    "max_checks": min(self.config.max_checks, 6),
                },
                output_model=InvestigationPlan,
            )

        checks = [check for check in plan.checks if check.question.strip()]
        checks = checks[: self.config.max_checks]
        if not checks:
            checks = [
                VerificationCheck(
                    aspect_id="claim_direct_check",
                    question=f"Is this claim accurate: {claim}",
                    rationale="Fallback direct verification when planning fails.",
                    search_queries=[claim],
                )
            ]

        plan.claim = claim
        plan.checks = checks
        return plan

    async def _run_parallel_research(
        self,
        claim: str,
        plan: InvestigationPlan,
    ) -> list[AspectFinding]:
        semaphore = asyncio.Semaphore(max(1, self.config.max_parallel_research))

        async def run_check(check: VerificationCheck) -> AspectFinding:
            async with semaphore:
                return await self._research_check(claim, check)

        tasks = [asyncio.create_task(run_check(check)) for check in plan.checks]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        findings: list[AspectFinding] = []
        for check, result in zip(plan.checks, results, strict=False):
            if isinstance(result, Exception):
                findings.append(
                    AspectFinding(
                        aspect_id=check.aspect_id,
                        question=check.question,
                        signal=EvidenceSignal.INSUFFICIENT,
                        summary=f"Research subroutine failed: {result}",
                        confidence=0.0,
                        sources=[],
                        caveats=["This check failed and needs rerun."],
                    )
                )
                continue

            findings.append(result)

        return findings

    async def _research_check(
        self,
        claim: str,
        check: VerificationCheck,
    ) -> AspectFinding:
        prompt_payload = {
            "claim": claim,
            "check": check.model_dump(),
            "requirements": {
                "min_sources": 2,
                "must_use_search_tool": True,
                "preferred_provider": self.config.search_provider,
            },
        }
        if self.config.inference_provider == "openai-agents":
            if self.research_agent is None:
                raise RuntimeError("Research agent is not initialized.")

            result = await Runner.run(
                self.research_agent,
                json.dumps(prompt_payload, indent=2),
                max_turns=self.config.max_turns,
            )
            finding = result.final_output_as(AspectFinding, raise_if_incorrect_type=True)
        else:
            if self.config.search_provider != "brave":
                raise RuntimeError(
                    "Gemini inference currently supports search_provider='brave' only."
                )

            search_queries = check.search_queries or [check.question, claim]
            search_payload = await self._run_brave_queries(search_queries)
            prompt_payload["search_results"] = search_payload

            finding = await self._run_gemini_structured(
                skill_name="research",
                payload=prompt_payload,
                output_model=AspectFinding,
            )

        if not finding.aspect_id.strip():
            finding.aspect_id = check.aspect_id
        if not finding.question.strip():
            finding.question = check.question
        return finding

    async def _judge_claim(
        self,
        claim: str,
        plan: InvestigationPlan,
        findings: list[AspectFinding],
    ) -> FactCheckReport:
        judge_payload = {
            "claim": claim,
            "plan": plan.model_dump(),
            "findings": [finding.model_dump() for finding in findings],
        }
        if self.config.inference_provider == "openai-agents":
            if self.judge_agent is None:
                raise RuntimeError("Judge agent is not initialized.")

            result = await Runner.run(
                self.judge_agent,
                json.dumps(judge_payload, indent=2),
                max_turns=self.config.max_turns + 2,
            )
            report = result.final_output_as(FactCheckReport, raise_if_incorrect_type=True)
        else:
            report = await self._run_gemini_structured(
                skill_name="judge",
                payload=judge_payload,
                output_model=FactCheckReport,
            )
        return report

    async def _run_gemini_structured(
        self,
        skill_name: str,
        payload: dict[str, Any],
        output_model: type[InvestigationPlan | AspectFinding | FactCheckReport],
    ) -> InvestigationPlan | AspectFinding | FactCheckReport:
        if self.gemini_client is None:
            raise RuntimeError("Gemini client is not initialized.")

        instructions = load_skill_prompt(skill_name)
        return await self.gemini_client.generate_structured(
            instructions=instructions,
            payload=payload,
            output_model=output_model,
        )

    async def _run_brave_queries(self, queries: list[str]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for query in queries:
            query_result = await asyncio.to_thread(
                run_brave_web_search,
                query,
                self.config.search_results_per_query,
            )
            results.append(query_result)
        return results

    def _merge_sources(
        self,
        report_sources: list[SourceEvidence],
        findings: list[AspectFinding],
    ) -> list[SourceEvidence]:
        combined: list[SourceEvidence] = []
        seen_urls: set[str] = set()

        for source in report_sources:
            normalized = source.url.strip().lower()
            if normalized and normalized not in seen_urls:
                seen_urls.add(normalized)
                combined.append(source)

        for finding in findings:
            for source in finding.sources:
                normalized = source.url.strip().lower()
                if normalized and normalized not in seen_urls:
                    seen_urls.add(normalized)
                    combined.append(source)

        return combined
