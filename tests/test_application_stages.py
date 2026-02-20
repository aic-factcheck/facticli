from __future__ import annotations

import unittest

from facticli.application.stages import ClaimExtractionStage, JudgeStage, PlanStage, ResearchStage
from facticli.application.progress import ProgressEvent
from facticli.core.artifacts import RunArtifacts
from facticli.types import (
    AspectFinding,
    CheckworthyClaim,
    ClaimExtractionResult,
    EvidenceSignal,
    FactCheckReport,
    InvestigationPlan,
    SourceEvidence,
    VeracityVerdict,
    VerificationCheck,
)


class _FakePlanner:
    async def plan(self, claim: str, max_checks: int) -> InvestigationPlan:
        return InvestigationPlan(
            claim="ignored",
            checks=[
                VerificationCheck(
                    aspect_id="Timeline 1",
                    question="  Was it completed in 1889? ",
                    rationale="  date  ",
                    search_queries=["", "Eiffel Tower 1889", "eiffel tower 1889"],
                ),
                VerificationCheck(
                    aspect_id="timeline_1",
                    question="Was it for the World's Fair?",
                    rationale="event",
                    search_queries=[],
                ),
            ],
        )


class _FakeResearcher:
    def __init__(self) -> None:
        self.calls: dict[str, int] = {}

    async def research(self, claim: str, check: VerificationCheck) -> AspectFinding:
        self.calls[check.aspect_id] = self.calls.get(check.aspect_id, 0) + 1
        if check.aspect_id == "bad_1":
            raise RuntimeError("simulated failure")
        return AspectFinding(
            aspect_id=check.aspect_id,
            question=check.question,
            signal=EvidenceSignal.SUPPORTS,
            summary=f"Supports {check.aspect_id}",
            confidence=0.8,
            sources=[],
        )


class _FakeJudge:
    async def judge(
        self,
        claim: str,
        plan: InvestigationPlan,
        findings: list[AspectFinding],
    ) -> FactCheckReport:
        return FactCheckReport(
            claim="placeholder",
            verdict=VeracityVerdict.SUPPORTED,
            verdict_confidence=0.75,
            justification="Supported by evidence.",
            key_points=[],
            findings=[],
            sources=[
                SourceEvidence(
                    title="Source A",
                    url="https://example.org/article?utm_source=test",
                    snippet="Evidence A.",
                )
            ],
        )


class _FakeExtractionBackend:
    async def extract(self, input_text: str, max_claims: int) -> ClaimExtractionResult:
        return ClaimExtractionResult(
            input_text="ignored",
            claims=[
                CheckworthyClaim(
                    claim_id="",
                    claim_text="Inflation fell below 3%.",
                    source_fragment="inflation fell below 3%",
                    checkworthy_reason="Numeric claim.",
                ),
                CheckworthyClaim(
                    claim_id="claim_1",
                    claim_text="Wages rose by 10%.",
                    source_fragment="wages rose by 10%",
                    checkworthy_reason="Numeric claim.",
                ),
                CheckworthyClaim(
                    claim_id="claim_3",
                    claim_text="This should be trimmed.",
                    source_fragment="trim",
                    checkworthy_reason="Extra claim.",
                ),
            ],
        )


class StageTests(unittest.IsolatedAsyncioTestCase):
    async def test_plan_stage_normalizes_checks(self):
        stage = PlanStage(planner=_FakePlanner(), max_checks=3, max_search_queries_per_check=4)
        artifacts = RunArtifacts(claim="raw", normalized_claim="raw")
        events: list[ProgressEvent] = []

        def progress_callback(event: ProgressEvent) -> None:
            events.append(event)

        plan = await stage.execute("Eiffel claim", artifacts, progress_callback=progress_callback)

        self.assertEqual(len(plan.checks), 2)
        self.assertEqual(plan.checks[0].aspect_id, "timeline_1")
        self.assertEqual(plan.checks[1].aspect_id, "timeline_1_2")
        self.assertEqual(plan.checks[0].question, "Was it completed in 1889?")
        self.assertIn("Eiffel claim", plan.checks[0].search_queries)
        self.assertIsNotNone(artifacts.plan_raw)
        self.assertIsNotNone(artifacts.plan_normalized)
        self.assertEqual(events[0].kind, "planning_started")
        self.assertEqual(events[-1].kind, "planning_completed")

    async def test_research_stage_retries_and_converts_failures(self):
        researcher = _FakeResearcher()
        stage = ResearchStage(
            researcher=researcher,
            max_parallel_research=2,
            research_timeout_seconds=0,
            research_retry_attempts=1,
        )
        artifacts = RunArtifacts(claim="raw", normalized_claim="raw")
        events: list[ProgressEvent] = []
        plan = InvestigationPlan(
            claim="claim",
            checks=[
                VerificationCheck(
                    aspect_id="ok_1",
                    question="Q1",
                    rationale="R1",
                    search_queries=["q1"],
                ),
                VerificationCheck(
                    aspect_id="bad_1",
                    question="Q2",
                    rationale="R2",
                    search_queries=["q2"],
                ),
            ],
            assumptions=[],
        )

        def progress_callback(event: ProgressEvent) -> None:
            events.append(event)

        findings = await stage.execute("claim", plan, artifacts, progress_callback=progress_callback)

        self.assertEqual(len(findings), 2)
        bad = [finding for finding in findings if finding.aspect_id == "bad_1"][0]
        self.assertEqual(bad.signal, EvidenceSignal.INSUFFICIENT)
        bad_artifact = [entry for entry in artifacts.research_checks if entry.check.aspect_id == "bad_1"][0]
        self.assertEqual(bad_artifact.attempts, 2)
        self.assertEqual(len(bad_artifact.errors), 2)
        event_kinds = [event.kind for event in events]
        self.assertIn("research_started", event_kinds)
        self.assertIn("research_check_completed", event_kinds)
        self.assertIn("research_check_failed", event_kinds)
        self.assertEqual(event_kinds[-1], "research_completed")

    async def test_judge_stage_backfills_findings_and_deduplicates_sources(self):
        stage = JudgeStage(judge=_FakeJudge())
        artifacts = RunArtifacts(claim="raw", normalized_claim="raw")
        events: list[ProgressEvent] = []
        findings = [
            AspectFinding(
                aspect_id="timeline_1",
                question="Was it completed in 1889?",
                signal=EvidenceSignal.SUPPORTS,
                summary="Supported.",
                confidence=0.9,
                sources=[
                    SourceEvidence(
                        title="Source A duplicate",
                        url="https://example.org/article",
                        snippet="Same source without tracking params.",
                    ),
                    SourceEvidence(
                        title="Source B",
                        url="https://example.org/other",
                        snippet="Additional corroboration.",
                    ),
                ],
                caveats=[],
            )
        ]

        report = await stage.execute(
            claim="The Eiffel Tower was built in 1889.",
            plan=InvestigationPlan(claim="x", checks=[], assumptions=[]),
            findings=findings,
            artifacts=artifacts,
            progress_callback=events.append,
        )

        self.assertEqual(len(report.findings), 1)
        self.assertEqual(len(report.sources), 2)
        self.assertIsNotNone(artifacts.report_raw)
        self.assertIsNotNone(artifacts.report_final)
        self.assertEqual(events[0].kind, "judging_started")
        self.assertEqual(events[-1].kind, "judging_completed")

    async def test_claim_extraction_stage_normalizes_output(self):
        stage = ClaimExtractionStage(backend=_FakeExtractionBackend(), max_claims=2)

        result = await stage.execute("  Transcript text  ")

        self.assertEqual(result.input_text, "Transcript text")
        self.assertEqual(len(result.claims), 2)
        self.assertEqual(result.claims[0].claim_id, "claim_1")
        self.assertEqual(result.claims[1].claim_id, "claim_1_2")


if __name__ == "__main__":
    unittest.main()
