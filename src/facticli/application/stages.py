from __future__ import annotations

import asyncio
from dataclasses import dataclass

from facticli.core.artifacts import RunArtifacts
from facticli.core.contracts import (
    AspectFinding,
    ClaimExtractionResult,
    EvidenceSignal,
    FactCheckReport,
    InvestigationPlan,
    SourceEvidence,
    VerificationCheck,
)
from facticli.core.normalize import normalize_plan_checks, normalize_query_list, normalize_source_url

from .interfaces import ClaimExtractionBackend, Judge, Planner, Researcher


@dataclass(frozen=True)
class PlanStage:
    planner: Planner
    max_checks: int
    max_search_queries_per_check: int

    async def execute(self, claim: str, artifacts: RunArtifacts) -> InvestigationPlan:
        plan_raw = await self.planner.plan(claim=claim, max_checks=self.max_checks)
        artifacts.plan_raw = plan_raw

        checks = normalize_plan_checks(
            claim=claim,
            checks=plan_raw.checks,
            max_checks=self.max_checks,
            max_search_queries_per_check=self.max_search_queries_per_check,
        )
        if not checks:
            checks = [
                VerificationCheck(
                    aspect_id="claim_direct_check",
                    question=f"Is this claim accurate: {claim}",
                    rationale="Fallback direct verification when planning fails.",
                    search_queries=normalize_query_list(
                        [claim], max_queries=self.max_search_queries_per_check
                    ),
                )
            ]

        plan = plan_raw.model_copy(update={"claim": claim, "checks": checks})
        artifacts.plan_normalized = plan
        return plan


@dataclass(frozen=True)
class ResearchStage:
    researcher: Researcher
    max_parallel_research: int
    research_timeout_seconds: float
    research_retry_attempts: int

    async def execute(
        self,
        claim: str,
        plan: InvestigationPlan,
        artifacts: RunArtifacts,
    ) -> list[AspectFinding]:
        semaphore = asyncio.Semaphore(max(1, self.max_parallel_research))
        timeout = self.research_timeout_seconds or None
        max_attempts = 1 + max(0, self.research_retry_attempts)

        async def run_check(check: VerificationCheck) -> AspectFinding:
            artifact = artifacts.get_or_create_check(check)
            async with semaphore:
                last_error: Exception | None = None
                for _attempt in range(max_attempts):
                    artifact.attempts += 1
                    try:
                        task = self.researcher.research(claim=claim, check=check)
                        if timeout:
                            finding = await asyncio.wait_for(task, timeout=timeout)
                        else:
                            finding = await task
                        artifact.finding = finding
                        return finding
                    except Exception as exc:  # pragma: no cover
                        last_error = exc
                        artifact.errors.append(f"{type(exc).__name__}: {exc}")

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


@dataclass(frozen=True)
class JudgeStage:
    judge: Judge

    async def execute(
        self,
        claim: str,
        plan: InvestigationPlan,
        findings: list[AspectFinding],
        artifacts: RunArtifacts,
    ) -> FactCheckReport:
        report_raw = await self.judge.judge(claim=claim, plan=plan, findings=findings)
        artifacts.report_raw = report_raw

        report_final = report_raw.model_copy(
            update={
                "claim": claim,
                "findings": report_raw.findings or findings,
                "sources": self._merge_sources(report_raw.sources, findings),
            }
        )
        artifacts.report_final = report_final
        return report_final

    def _merge_sources(
        self,
        report_sources: list[SourceEvidence],
        findings: list[AspectFinding],
    ) -> list[SourceEvidence]:
        combined: list[SourceEvidence] = []
        seen_urls: set[str] = set()

        for source in report_sources:
            normalized = normalize_source_url(source.url)
            if normalized and normalized not in seen_urls:
                seen_urls.add(normalized)
                combined.append(source)

        for finding in findings:
            for source in finding.sources:
                normalized = normalize_source_url(source.url)
                if normalized and normalized not in seen_urls:
                    seen_urls.add(normalized)
                    combined.append(source)

        return combined


@dataclass(frozen=True)
class ClaimExtractionStage:
    backend: ClaimExtractionBackend
    max_claims: int

    async def execute(self, input_text: str) -> ClaimExtractionResult:
        normalized_text = input_text.strip()
        if not normalized_text:
            raise ValueError("Input text is empty.")

        extraction = await self.backend.extract(input_text=normalized_text, max_claims=self.max_claims)
        extraction.input_text = normalized_text
        extraction.claims = extraction.claims[: self.max_claims]

        seen_ids: set[str] = set()
        for index, claim in enumerate(extraction.claims, start=1):
            if not claim.claim_id.strip():
                claim.claim_id = f"claim_{index}"
            if claim.claim_id in seen_ids:
                claim.claim_id = f"{claim.claim_id}_{index}"
            seen_ids.add(claim.claim_id)

        return extraction
