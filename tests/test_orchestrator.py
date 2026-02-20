from __future__ import annotations

import json
import unittest
from unittest.mock import AsyncMock, patch

from facticli.orchestrator import FactCheckOrchestrator, OrchestratorConfig
from facticli.types import (
    AspectFinding,
    EvidenceSignal,
    FactCheckReport,
    InvestigationPlan,
    SourceEvidence,
    VerificationCheck,
    VeracityVerdict,
)


class _FakeRunResult:
    def __init__(self, output):
        self.output = output

    def final_output_as(self, _cls, raise_if_incorrect_type: bool = False):
        return self.output


class OrchestratorTests(unittest.IsolatedAsyncioTestCase):
    async def test_full_check_flow_openai_matches_notebook_scenario(self):
        planner_agent = object()
        research_agent = object()
        judge_agent = object()

        plan = InvestigationPlan(
            claim="unused",
            checks=[
                VerificationCheck(
                    aspect_id="timeline_1",
                    question="Was Eiffel Tower completed in 1889?",
                    rationale="Date check",
                    search_queries=["Eiffel Tower completion year"],
                ),
                VerificationCheck(
                    aspect_id="event_1",
                    question="Was it linked to the World's Fair?",
                    rationale="Event linkage check",
                    search_queries=["Eiffel Tower 1889 Exposition Universelle"],
                ),
            ],
        )

        async def fake_runner(agent, input, max_turns=10):  # noqa: A002
            if agent is planner_agent:
                return _FakeRunResult(plan)

            if agent is research_agent:
                payload = json.loads(input)
                check = payload["check"]
                return _FakeRunResult(
                    AspectFinding(
                        aspect_id=check["aspect_id"],
                        question=check["question"],
                        signal=EvidenceSignal.SUPPORTS,
                        summary=f"Evidence supports {check['aspect_id']}.",
                        confidence=0.82,
                        sources=[
                            SourceEvidence(
                                title=f"Source for {check['aspect_id']}",
                                url="https://example.org/shared-source",
                                snippet=f"Snippet for {check['aspect_id']}",
                            )
                        ],
                    )
                )

            if agent is judge_agent:
                payload = json.loads(input)
                return _FakeRunResult(
                    FactCheckReport(
                        claim=payload["claim"],
                        verdict=VeracityVerdict.SUPPORTED,
                        verdict_confidence=0.83,
                        justification="Both checks are supported by cited sources.",
                        key_points=["Timeline and event linkage are supported."],
                        findings=[],
                        sources=[],
                    )
                )

            raise AssertionError("Unexpected agent passed to Runner.run")

        with (
            patch("facticli.orchestrator.build_planner_agent", return_value=planner_agent),
            patch("facticli.orchestrator.build_research_agent", return_value=research_agent),
            patch("facticli.orchestrator.build_judge_agent", return_value=judge_agent),
            patch("facticli.orchestrator.Runner.run", new=AsyncMock(side_effect=fake_runner)),
        ):
            orchestrator = FactCheckOrchestrator(
                OrchestratorConfig(
                    inference_provider="openai-agents",
                    max_checks=4,
                    max_parallel_research=2,
                )
            )
            run = await orchestrator.check_claim("The Eiffel Tower was built in 1889 for the World's Fair.")

        self.assertEqual(run.report.claim, "The Eiffel Tower was built in 1889 for the World's Fair.")
        self.assertEqual(run.report.verdict, VeracityVerdict.SUPPORTED)
        self.assertEqual(len(run.findings), 2)
        # Judge returned no findings/sources, so orchestrator should backfill and dedupe.
        self.assertEqual(len(run.report.findings), 2)
        self.assertEqual(len(run.report.sources), 1)
        self.assertEqual(run.report.sources[0].url, "https://example.org/shared-source")

    async def test_parallel_research_converts_exceptions_into_insufficient_findings(self):
        with (
            patch("facticli.orchestrator.build_planner_agent", return_value=object()),
            patch("facticli.orchestrator.build_research_agent", return_value=object()),
            patch("facticli.orchestrator.build_judge_agent", return_value=object()),
        ):
            orchestrator = FactCheckOrchestrator(OrchestratorConfig(inference_provider="openai-agents"))

        plan = InvestigationPlan(
            claim="x",
            checks=[
                VerificationCheck(
                    aspect_id="ok_1",
                    question="Question 1",
                    rationale="r",
                    search_queries=["q1"],
                ),
                VerificationCheck(
                    aspect_id="bad_1",
                    question="Question 2",
                    rationale="r",
                    search_queries=["q2"],
                ),
            ],
        )

        async def fake_research(_claim: str, check: VerificationCheck) -> AspectFinding:
            if check.aspect_id == "bad_1":
                raise RuntimeError("simulated failure")
            return AspectFinding(
                aspect_id=check.aspect_id,
                question=check.question,
                signal=EvidenceSignal.SUPPORTS,
                summary="ok",
                confidence=0.9,
                sources=[],
            )

        orchestrator._research_check = fake_research  # type: ignore[method-assign]
        findings = await orchestrator._run_parallel_research("claim", plan)

        self.assertEqual(len(findings), 2)
        failure = [f for f in findings if f.aspect_id == "bad_1"][0]
        self.assertEqual(failure.signal, EvidenceSignal.INSUFFICIENT)
        self.assertIn("failed", failure.summary.lower())

    async def test_gemini_research_uses_brave_search_payload(self):
        with patch("facticli.orchestrator.GeminiStructuredClient", return_value=object()):
            orchestrator = FactCheckOrchestrator(
                OrchestratorConfig(
                    inference_provider="gemini",
                    search_provider="brave",
                )
            )

        check = VerificationCheck(
            aspect_id="timeline_1",
            question="Was it completed in 1889?",
            rationale="Date check",
            search_queries=["Eiffel Tower completion year"],
        )

        orchestrator._run_brave_queries = AsyncMock(
            return_value=[
                {
                    "provider": "brave",
                    "query": "Eiffel Tower completion year",
                    "result_count": 1,
                    "results": [{"title": "x", "url": "https://example.org", "description": "d"}],
                }
            ]
        )
        orchestrator._run_gemini_structured = AsyncMock(
            return_value=AspectFinding(
                aspect_id="",
                question="",
                signal=EvidenceSignal.SUPPORTS,
                summary="Supported by search results.",
                confidence=0.8,
                sources=[],
            )
        )

        finding = await orchestrator._research_check("claim text", check)
        self.assertEqual(finding.aspect_id, "timeline_1")
        self.assertEqual(finding.question, "Was it completed in 1889?")

        kwargs = orchestrator._run_gemini_structured.await_args.kwargs
        payload = kwargs["payload"]
        self.assertIn("search_results", payload)
        self.assertEqual(payload["search_results"][0]["provider"], "brave")


if __name__ == "__main__":
    unittest.main()

