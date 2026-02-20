from __future__ import annotations

import unittest

from facticli.application.progress import ProgressEvent, emit_progress
from facticli.application.services import FactCheckService
from facticli.core.artifacts import RunArtifacts
from facticli.types import (
    AspectFinding,
    EvidenceSignal,
    FactCheckReport,
    InvestigationPlan,
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


if __name__ == "__main__":
    unittest.main()
