from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from typing import Any, TypeVar

from pydantic import BaseModel

from agents import Runner

from .agents import build_judge_agent, build_planner_agent, build_research_agent
from .brave_search import run_brave_web_search
from .config import InferenceConfig
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

_T = TypeVar("_T", bound=BaseModel)


@dataclass(frozen=True)
class OrchestratorConfig(InferenceConfig):
    max_checks: int = 4
    max_parallel_research: int = 4
    search_context_size: str = "high"
    search_provider: str = "openai"
    search_results_per_query: int = 5
    max_search_queries_per_check: int = 5
    # Extra turns given to the judge over other agents (synthesis is more complex).
    judge_extra_turns: int = 2
    # Per-research-task wall-clock timeout in seconds; 0 means no limit.
    research_timeout_seconds: float = 120.0
    # Number of retries for each check after the first failed attempt.
    research_retry_attempts: int = 1


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

        report = report.model_copy(update={
            "claim": normalized_claim,
            "findings": report.findings or findings,
            "sources": self._merge_sources(report.sources, findings),
        })

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
                f"Output at most {self.config.max_checks} checks."
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
                    "max_checks": self.config.max_checks,
                },
                output_model=InvestigationPlan,
            )

        checks = self._normalize_plan_checks(claim, plan.checks)
        if not checks:
            checks = [
                VerificationCheck(
                    aspect_id="claim_direct_check",
                    question=f"Is this claim accurate: {claim}",
                    rationale="Fallback direct verification when planning fails.",
                    search_queries=self._normalize_query_list([claim]),
                )
            ]

        return plan.model_copy(update={"claim": claim, "checks": checks})

    async def _run_parallel_research(
        self,
        claim: str,
        plan: InvestigationPlan,
    ) -> list[AspectFinding]:
        semaphore = asyncio.Semaphore(max(1, self.config.max_parallel_research))
        timeout = self.config.research_timeout_seconds or None
        max_attempts = 1 + max(0, self.config.research_retry_attempts)

        async def run_check(check: VerificationCheck) -> AspectFinding:
            async with semaphore:
                last_error: Exception | None = None
                for _attempt in range(max_attempts):
                    try:
                        coro = self._research_check(claim, check)
                        if timeout:
                            return await asyncio.wait_for(coro, timeout=timeout)
                        return await coro
                    except Exception as exc:  # pragma: no cover - exercised via gather exception path.
                        last_error = exc

                assert last_error is not None
                raise last_error

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
                        summary=(
                            f"Research subroutine failed after {max_attempts} attempt(s): "
                            f"{type(result).__name__}: {result}"
                        ),
                        confidence=0.0,
                        sources=[],
                        caveats=[
                            "This check failed and was downgraded to insufficient evidence."
                        ],
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

            search_queries = self._normalize_search_queries(check, claim)
            search_payload = await self._run_brave_queries(search_queries)
            prompt_payload["search_results"] = search_payload

            finding = await self._run_gemini_structured(
                skill_name="research_gemini",
                payload=prompt_payload,
                output_model=AspectFinding,
            )

        updates: dict[str, Any] = {}
        if not finding.aspect_id.strip():
            updates["aspect_id"] = check.aspect_id
        if not finding.question.strip():
            updates["question"] = check.question
        return finding.model_copy(update=updates) if updates else finding

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
                max_turns=self.config.max_turns + self.config.judge_extra_turns,
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
        output_model: type[_T],
    ) -> _T:
        if self.gemini_client is None:
            raise RuntimeError("Gemini client is not initialized.")

        instructions = load_skill_prompt(skill_name)
        return await self.gemini_client.generate_structured(
            instructions=instructions,
            payload=payload,
            output_model=output_model,
        )

    def _normalize_plan_checks(
        self,
        claim: str,
        checks: list[VerificationCheck],
    ) -> list[VerificationCheck]:
        normalized_checks: list[VerificationCheck] = []
        used_aspect_ids: set[str] = set()

        for index, check in enumerate(checks, start=1):
            question = check.question.strip()
            if not question:
                continue

            base_aspect_id = self._sanitize_aspect_id(check.aspect_id, fallback_index=index)
            aspect_id = base_aspect_id
            suffix = 2
            while aspect_id in used_aspect_ids:
                aspect_id = f"{base_aspect_id}_{suffix}"
                suffix += 1
            used_aspect_ids.add(aspect_id)

            normalized_checks.append(
                check.model_copy(
                    update={
                        "aspect_id": aspect_id,
                        "question": question,
                        "rationale": check.rationale.strip(),
                        "search_queries": self._normalize_query_list(
                            check.search_queries,
                            fallback=[question, claim],
                        ),
                    }
                )
            )

            if len(normalized_checks) >= self.config.max_checks:
                break

        return normalized_checks

    def _normalize_search_queries(self, check: VerificationCheck, claim: str) -> list[str]:
        return self._normalize_query_list(
            check.search_queries,
            fallback=[check.question, claim],
        )

    def _normalize_query_list(
        self,
        queries: list[str],
        fallback: list[str] | None = None,
    ) -> list[str]:
        candidates = [*queries, *(fallback or [])]
        normalized: list[str] = []
        seen: set[str] = set()

        for candidate in candidates:
            query = candidate.strip()
            if not query:
                continue
            key = query.casefold()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(query)
            if len(normalized) >= max(1, self.config.max_search_queries_per_check):
                break

        return normalized

    def _sanitize_aspect_id(self, raw_aspect_id: str, fallback_index: int) -> str:
        lowered = raw_aspect_id.strip().lower()
        cleaned = re.sub(r"[^a-z0-9_]+", "_", lowered).strip("_")
        return cleaned or f"check_{fallback_index}"

    async def _run_brave_queries(self, queries: list[str]) -> list[dict[str, Any]]:
        if not queries:
            return []

        tasks = [
            asyncio.to_thread(run_brave_web_search, query, self.config.search_results_per_query)
            for query in queries
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        payloads: list[dict[str, Any]] = []
        for query, result in zip(queries, results, strict=False):
            if isinstance(result, Exception):
                payloads.append(
                    {
                        "provider": "brave",
                        "query": query,
                        "result_count": 0,
                        "results": [],
                        "error": f"{type(result).__name__}: {result}",
                    }
                )
                continue
            payloads.append(result)

        return payloads

    def _merge_sources(
        self,
        report_sources: list[SourceEvidence],
        findings: list[AspectFinding],
    ) -> list[SourceEvidence]:
        combined: list[SourceEvidence] = []
        seen_urls: set[str] = set()

        for source in report_sources:
            normalized = self._normalize_source_url(source.url)
            if normalized and normalized not in seen_urls:
                seen_urls.add(normalized)
                combined.append(source)

        for finding in findings:
            for source in finding.sources:
                normalized = self._normalize_source_url(source.url)
                if normalized and normalized not in seen_urls:
                    seen_urls.add(normalized)
                    combined.append(source)

        return combined

    def _normalize_source_url(self, url: str) -> str:
        stripped = url.strip()
        if not stripped:
            return ""

        try:
            parts = urlsplit(stripped)
        except ValueError:
            return stripped.lower()

        filtered_query = [
            (key, value)
            for key, value in parse_qsl(parts.query, keep_blank_values=True)
            if not key.lower().startswith("utm_")
        ]
        normalized_query = urlencode(filtered_query, doseq=True)
        normalized_path = parts.path.rstrip("/")

        return urlunsplit(
            (
                parts.scheme.lower(),
                parts.netloc.lower(),
                normalized_path,
                normalized_query,
                "",
            )
        )
