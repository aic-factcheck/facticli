from __future__ import annotations

from dataclasses import dataclass

from facticli.core.artifacts import RunArtifacts
from facticli.core.contracts import (
    AspectFinding,
    ClaimExtractionResult,
    FactCheckReport,
    InvestigationPlan,
    ReviewAction,
    VerificationCheck,
)
from facticli.core.normalize import normalize_plan_checks

from .progress import ProgressCallback, emit_progress
from .repository import RunArtifactRepository
from .stages import ClaimExtractionStage, JudgeStage, PlanStage, ResearchStage, ReviewStage


@dataclass
class FactCheckRun:
    claim: str
    plan: InvestigationPlan
    findings: list[AspectFinding]
    report: FactCheckReport
    artifacts: RunArtifacts


@dataclass(frozen=True)
class FactCheckService:
    plan_stage: PlanStage
    research_stage: ResearchStage
    judge_stage: JudgeStage
    review_stage: ReviewStage | None = None
    max_feedback_rounds: int = 0
    artifact_repository: RunArtifactRepository | None = None

    async def check_claim(
        self,
        claim: str,
        progress_callback: ProgressCallback | None = None,
    ) -> FactCheckRun:
        normalized_claim = claim.strip()
        if not normalized_claim:
            raise ValueError("Claim is empty.")

        artifacts = RunArtifacts(claim=claim, normalized_claim=normalized_claim)
        await emit_progress(progress_callback, "run_started", {"claim": normalized_claim})
        plan = await self.plan_stage.execute(
            claim=normalized_claim,
            artifacts=artifacts,
            progress_callback=progress_callback,
        )
        findings = await self.research_stage.execute(
            claim=normalized_claim,
            plan=plan,
            artifacts=artifacts,
            progress_callback=progress_callback,
        )
        plan, findings = await self._run_feedback_loop(
            claim=normalized_claim,
            plan=plan,
            findings=findings,
            artifacts=artifacts,
            progress_callback=progress_callback,
        )
        report = await self.judge_stage.execute(
            claim=normalized_claim,
            plan=plan,
            findings=findings,
            artifacts=artifacts,
            progress_callback=progress_callback,
        )

        if self.artifact_repository is not None:
            self.artifact_repository.save(artifacts)

        await emit_progress(
            progress_callback,
            "run_completed",
            {
                "claim": normalized_claim,
                "verdict": report.verdict.value,
                "verdict_confidence": report.verdict_confidence,
            },
        )

        return FactCheckRun(
            claim=normalized_claim,
            plan=plan,
            findings=findings,
            report=report,
            artifacts=artifacts,
        )

    async def _run_feedback_loop(
        self,
        *,
        claim: str,
        plan: InvestigationPlan,
        findings: list[AspectFinding],
        artifacts: RunArtifacts,
        progress_callback: ProgressCallback | None,
    ) -> tuple[InvestigationPlan, list[AspectFinding]]:
        if self.review_stage is None or self.max_feedback_rounds <= 0:
            return plan, findings

        current_plan = plan
        current_findings = findings

        for round_index in range(1, self.max_feedback_rounds + 1):
            decision = await self.review_stage.execute(
                claim=claim,
                plan=current_plan,
                findings=current_findings,
                artifacts=artifacts,
                round_index=round_index,
                progress_callback=progress_callback,
            )
            if decision.action != ReviewAction.FOLLOW_UP:
                break

            follow_up_plan = self._build_follow_up_plan(
                claim=claim,
                current_plan=current_plan,
                current_findings=current_findings,
                retry_aspect_ids=decision.retry_aspect_ids,
                follow_up_checks=decision.follow_up_checks,
            )
            if not follow_up_plan.checks:
                break

            if artifacts.review_rounds:
                artifacts.review_rounds[-1].follow_up_plan = follow_up_plan
            await emit_progress(
                progress_callback,
                "feedback_round_started",
                {
                    "round_index": round_index,
                    "check_count": len(follow_up_plan.checks),
                },
            )
            new_findings = await self.research_stage.execute(
                claim=claim,
                plan=follow_up_plan,
                artifacts=artifacts,
                progress_callback=progress_callback,
            )
            current_plan = self._merge_plan(current_plan, follow_up_plan)
            current_findings = self._merge_findings(current_findings, new_findings)
            await emit_progress(
                progress_callback,
                "feedback_round_completed",
                {
                    "round_index": round_index,
                    "finding_count": len(current_findings),
                },
            )

        return current_plan, current_findings

    def _build_follow_up_plan(
        self,
        *,
        claim: str,
        current_plan: InvestigationPlan,
        current_findings: list[AspectFinding],
        retry_aspect_ids: list[str],
        follow_up_checks: list[VerificationCheck],
    ) -> InvestigationPlan:
        findings_by_aspect = {finding.aspect_id: finding for finding in current_findings}
        retry_checks = [
            check
            for check in current_plan.checks
            if check.aspect_id in retry_aspect_ids and check.aspect_id in findings_by_aspect
        ]
        normalized_follow_up_checks = normalize_plan_checks(
            claim=claim,
            checks=follow_up_checks,
            max_checks=(
                getattr(self.review_stage, "max_follow_up_checks", len(follow_up_checks))
                if self.review_stage
                else len(follow_up_checks)
            ),
            max_search_queries_per_check=getattr(self.plan_stage, "max_search_queries_per_check", 5),
        )

        existing_aspect_ids = {check.aspect_id for check in current_plan.checks}
        unique_follow_up_checks = []
        for check in normalized_follow_up_checks:
            if check.aspect_id in existing_aspect_ids:
                continue
            unique_follow_up_checks.append(check)
            existing_aspect_ids.add(check.aspect_id)

        return InvestigationPlan(
            claim=claim,
            checks=[*retry_checks, *unique_follow_up_checks],
            assumptions=[],
        )

    def _merge_plan(
        self,
        current_plan: InvestigationPlan,
        follow_up_plan: InvestigationPlan,
    ) -> InvestigationPlan:
        existing_aspect_ids = {check.aspect_id for check in current_plan.checks}
        merged_checks = list(current_plan.checks)
        for check in follow_up_plan.checks:
            if check.aspect_id in existing_aspect_ids:
                continue
            merged_checks.append(check)
            existing_aspect_ids.add(check.aspect_id)
        return current_plan.model_copy(update={"checks": merged_checks})

    def _merge_findings(
        self,
        current_findings: list[AspectFinding],
        new_findings: list[AspectFinding],
    ) -> list[AspectFinding]:
        findings_by_aspect = {finding.aspect_id: finding for finding in current_findings}
        ordered_aspect_ids = [finding.aspect_id for finding in current_findings]

        for finding in new_findings:
            if finding.aspect_id not in findings_by_aspect:
                ordered_aspect_ids.append(finding.aspect_id)
            findings_by_aspect[finding.aspect_id] = finding

        return [findings_by_aspect[aspect_id] for aspect_id in ordered_aspect_ids]


@dataclass(frozen=True)
class ClaimExtractionService:
    extraction_stage: ClaimExtractionStage

    async def extract_claims(self, input_text: str) -> ClaimExtractionResult:
        return await self.extraction_stage.execute(input_text)
