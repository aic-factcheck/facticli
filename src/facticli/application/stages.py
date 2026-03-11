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
    ReviewAction,
    ReviewDecision,
    SourceEvidence,
    VerificationCheck,
)
from facticli.core.normalize import normalize_plan_checks, normalize_query_list, normalize_source_url

from .interfaces import ClaimExtractionBackend, Judge, Planner, Researcher, Reviewer
from .progress import ProgressCallback, emit_progress


@dataclass(frozen=True)
class PlanStage:
    planner: Planner
    max_checks: int
    max_search_queries_per_check: int

    async def execute(
        self,
        claim: str,
        artifacts: RunArtifacts,
        progress_callback: ProgressCallback | None = None,
    ) -> InvestigationPlan:
        await emit_progress(progress_callback, "planning_started", {"claim": claim})
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
        await emit_progress(
            progress_callback,
            "planning_completed",
            {
                "claim": claim,
                "check_count": len(plan.checks),
                "checks": [
                    {
                        "aspect_id": check.aspect_id,
                        "question": check.question,
                    }
                    for check in plan.checks
                ],
            },
        )
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
        progress_callback: ProgressCallback | None = None,
    ) -> list[AspectFinding]:
        semaphore = asyncio.Semaphore(max(1, self.max_parallel_research))
        timeout = self.research_timeout_seconds or None
        max_attempts = 1 + max(0, self.research_retry_attempts)
        await emit_progress(
            progress_callback,
            "research_started",
            {"claim": claim, "check_count": len(plan.checks)},
        )

        async def run_check(index: int, check: VerificationCheck) -> tuple[int, VerificationCheck, AspectFinding | Exception]:
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
                        return index, check, finding
                    except Exception as exc:  # pragma: no cover
                        last_error = exc
                        artifact.errors.append(f"{type(exc).__name__}: {exc}")

                assert last_error is not None
                return index, check, last_error

        tasks = [
            asyncio.create_task(run_check(index, check)) for index, check in enumerate(plan.checks)
        ]

        ordered_findings: list[AspectFinding | None] = [None] * len(plan.checks)
        for task in asyncio.as_completed(tasks):
            index, check, outcome = await task
            if isinstance(outcome, Exception):
                finding = AspectFinding(
                    aspect_id=check.aspect_id,
                    question=check.question,
                    signal=EvidenceSignal.INSUFFICIENT,
                    summary=(
                        f"Research subroutine failed after {max_attempts} attempt(s): "
                        f"{type(outcome).__name__}: {outcome}"
                    ),
                    confidence=0.0,
                    sources=[],
                    caveats=[
                        "This check failed and was downgraded to insufficient evidence."
                    ],
                )
                ordered_findings[index] = finding
                await emit_progress(
                    progress_callback,
                    "research_check_failed",
                    {
                        "aspect_id": check.aspect_id,
                        "question": check.question,
                        "error": f"{type(outcome).__name__}: {outcome}",
                        "attempts": max_attempts,
                        "finding": finding.model_dump(),
                    },
                )
                continue

            ordered_findings[index] = outcome
            await emit_progress(
                progress_callback,
                "research_check_completed",
                {
                    "aspect_id": outcome.aspect_id,
                    "question": outcome.question,
                    "signal": outcome.signal.value,
                    "confidence": outcome.confidence,
                    "summary": outcome.summary,
                    "source_count": len(outcome.sources),
                },
            )

        findings = [finding for finding in ordered_findings if finding is not None]
        await emit_progress(
            progress_callback,
            "research_completed",
            {"finding_count": len(findings)},
        )
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
        progress_callback: ProgressCallback | None = None,
    ) -> FactCheckReport:
        await emit_progress(
            progress_callback,
            "judging_started",
            {"claim": claim, "finding_count": len(findings)},
        )
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
        await emit_progress(
            progress_callback,
            "judging_completed",
            {
                "verdict": report_final.verdict.value,
                "verdict_confidence": report_final.verdict_confidence,
                "source_count": len(report_final.sources),
            },
        )
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
class ReviewStage:
    reviewer: Reviewer
    max_follow_up_checks: int
    max_search_queries_per_check: int

    async def execute(
        self,
        claim: str,
        plan: InvestigationPlan,
        findings: list[AspectFinding],
        artifacts: RunArtifacts,
        *,
        round_index: int,
        progress_callback: ProgressCallback | None = None,
    ) -> ReviewDecision:
        await emit_progress(
            progress_callback,
            "review_started",
            {
                "claim": claim,
                "round_index": round_index,
                "finding_count": len(findings),
            },
        )
        review_artifact = artifacts.add_review_round(
            round_index=round_index,
            input_plan=plan,
            input_findings=findings,
        )
        decision_raw = await self.reviewer.review(claim=claim, plan=plan, findings=findings)

        normalized_action = decision_raw.action
        retry_aspect_ids = self._normalize_retry_aspect_ids(decision_raw.retry_aspect_ids, plan)
        follow_up_checks = normalize_plan_checks(
            claim=claim,
            checks=decision_raw.follow_up_checks,
            max_checks=self.max_follow_up_checks,
            max_search_queries_per_check=self.max_search_queries_per_check,
        )
        follow_up_checks = self._dedupe_follow_up_checks(follow_up_checks, plan)

        if normalized_action == ReviewAction.FOLLOW_UP and not retry_aspect_ids and not follow_up_checks:
            normalized_action = ReviewAction.FINALIZE

        decision_final = decision_raw.model_copy(
            update={
                "claim": claim,
                "action": normalized_action,
                "retry_aspect_ids": retry_aspect_ids,
                "follow_up_checks": follow_up_checks,
            }
        )
        review_artifact.decision = decision_final
        if follow_up_checks:
            review_artifact.follow_up_plan = InvestigationPlan(
                claim=claim,
                checks=follow_up_checks,
                assumptions=[],
            )

        await emit_progress(
            progress_callback,
            "review_completed",
            {
                "round_index": round_index,
                "action": decision_final.action.value,
                "retry_count": len(decision_final.retry_aspect_ids),
                "follow_up_count": len(decision_final.follow_up_checks),
            },
        )
        return decision_final

    def _normalize_retry_aspect_ids(
        self,
        retry_aspect_ids: list[str],
        plan: InvestigationPlan,
    ) -> list[str]:
        available = {check.aspect_id for check in plan.checks}
        normalized: list[str] = []
        seen: set[str] = set()
        for aspect_id in retry_aspect_ids:
            candidate = aspect_id.strip()
            if not candidate or candidate not in available or candidate in seen:
                continue
            seen.add(candidate)
            normalized.append(candidate)
        return normalized

    def _dedupe_follow_up_checks(
        self,
        follow_up_checks: list[VerificationCheck],
        plan: InvestigationPlan,
    ) -> list[VerificationCheck]:
        used_aspect_ids = {check.aspect_id for check in plan.checks}
        deduped: list[VerificationCheck] = []
        for check in follow_up_checks:
            aspect_id = check.aspect_id
            suffix = 2
            while aspect_id in used_aspect_ids:
                aspect_id = f"{check.aspect_id}_{suffix}"
                suffix += 1
            used_aspect_ids.add(aspect_id)
            deduped.append(check.model_copy(update={"aspect_id": aspect_id}))
        return deduped


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
