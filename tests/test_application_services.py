from __future__ import annotations

import unittest

from facticli.application.progress import ProgressEvent, emit_progress
from facticli.application.services import FactCheckService
from facticli.core.artifacts import RunArtifacts
from facticli.core.contracts import (
    AspectFinding,
    EvidenceSignal,
    FactCheckReport,
    InvestigationPlan,
    ReviewAction,
    ReviewDecision,
    VeracityVerdict,
    VerificationCheck,
)


class _PlanStage:
    async def execute(self, claim: str, artifacts: RunArtifacts, progress_callback=None):
        plan = InvestigationPlan(
            claim=claim,
            checks=[
                VerificationCheck(
                    aspect_id="check_1",
                    question="Is this true?",
                    rationale="Direct check",
                    search_queries=[claim],
                )
            ],
            assumptions=[],
        )
        artifacts.plan_raw = plan
        artifacts.plan_normalized = plan
        await emit_progress(progress_callback, "planning_completed", {"check_count": 1, "checks": []})
        return plan


class _ResearchStage:
    async def execute(self, claim: str, plan: InvestigationPlan, artifacts: RunArtifacts, progress_callback=None):
        finding = AspectFinding(
            aspect_id="check_1",
            question="Is this true?",
            signal=EvidenceSignal.SUPPORTS,
            summary="Supported.",
            confidence=0.8,
            sources=[],
            caveats=[],
        )
        await emit_progress(progress_callback, "research_completed", {"finding_count": 1})
        return [finding]


class _JudgeStage:
    async def execute(
        self,
        claim: str,
        plan: InvestigationPlan,
        findings: list[AspectFinding],
        artifacts: RunArtifacts,
        progress_callback=None,
    ):
        report = FactCheckReport(
            claim=claim,
            verdict=VeracityVerdict.SUPPORTED,
            verdict_confidence=0.85,
            justification="Supported.",
            key_points=[],
            findings=findings,
            sources=[],
        )
        artifacts.report_raw = report
        artifacts.report_final = report
        await emit_progress(progress_callback, "judging_completed", {"verdict": report.verdict.value})
        return report


class _ReviewStage:
    def __init__(self) -> None:
        self.calls = 0

    async def execute(
        self,
        claim: str,
        plan: InvestigationPlan,
        findings: list[AspectFinding],
        artifacts: RunArtifacts,
        *,
        round_index: int,
        progress_callback=None,
    ):
        self.calls += 1
        await emit_progress(progress_callback, "review_completed", {"round_index": round_index})
        if self.calls == 1:
            return ReviewDecision(
                claim=claim,
                action=ReviewAction.FOLLOW_UP,
                rationale="Need an extra targeted check.",
                follow_up_checks=[
                    VerificationCheck(
                        aspect_id="check_2",
                        question="Can this be corroborated by a second source?",
                        rationale="Need corroboration.",
                        search_queries=["claim corroboration"],
                    )
                ],
                retry_aspect_ids=[],
            )
        return ReviewDecision(
            claim=claim,
            action=ReviewAction.FINALIZE,
            rationale="Enough evidence collected.",
            follow_up_checks=[],
            retry_aspect_ids=[],
        )


class ApplicationServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_fact_check_service_emits_run_lifecycle_events(self):
        service = FactCheckService(
            plan_stage=_PlanStage(),
            research_stage=_ResearchStage(),
            judge_stage=_JudgeStage(),
        )
        events: list[ProgressEvent] = []

        run = await service.check_claim("The Eiffel Tower was built in 1889.", progress_callback=events.append)

        self.assertEqual(run.report.verdict, VeracityVerdict.SUPPORTED)
        event_kinds = [event.kind for event in events]
        self.assertEqual(event_kinds[0], "run_started")
        self.assertEqual(event_kinds[-1], "run_completed")
        self.assertIn("planning_completed", event_kinds)
        self.assertIn("research_completed", event_kinds)
        self.assertIn("judging_completed", event_kinds)

    async def test_fact_check_service_runs_one_follow_up_round_when_enabled(self):
        class _LoopResearchStage:
            def __init__(self) -> None:
                self.calls: list[list[str]] = []

            async def execute(
                self,
                claim: str,
                plan: InvestigationPlan,
                artifacts: RunArtifacts,
                progress_callback=None,
            ):
                self.calls.append([check.aspect_id for check in plan.checks])
                return [
                    AspectFinding(
                        aspect_id=check.aspect_id,
                        question=check.question,
                        signal=EvidenceSignal.SUPPORTS,
                        summary=f"Supported {check.aspect_id}.",
                        confidence=0.8,
                        sources=[],
                        caveats=[],
                    )
                    for check in plan.checks
                ]

        review_stage = _ReviewStage()
        research_stage = _LoopResearchStage()
        service = FactCheckService(
            plan_stage=_PlanStage(),
            research_stage=research_stage,
            judge_stage=_JudgeStage(),
            review_stage=review_stage,
            max_feedback_rounds=1,
        )

        run = await service.check_claim("The Eiffel Tower was built in 1889.")

        self.assertEqual(review_stage.calls, 1)
        self.assertEqual(research_stage.calls, [["check_1"], ["check_2"]])
        self.assertEqual([check.aspect_id for check in run.plan.checks], ["check_1", "check_2"])
        self.assertEqual([finding.aspect_id for finding in run.findings], ["check_1", "check_2"])


if __name__ == "__main__":
    unittest.main()
