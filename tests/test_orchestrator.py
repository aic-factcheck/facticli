from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from facticli.core.artifacts import RunArtifacts
from facticli.orchestrator import FactCheckOrchestrator, OrchestratorConfig
from facticli.types import FactCheckReport, InvestigationPlan, VeracityVerdict
from facticli.application.services import FactCheckRun


class OrchestratorWrapperTests(unittest.IsolatedAsyncioTestCase):
    async def test_check_claim_delegates_to_service(self):
        fake_run = FactCheckRun(
            claim="The Eiffel Tower was built in 1889.",
            plan=InvestigationPlan(claim="The Eiffel Tower was built in 1889.", checks=[], assumptions=[]),
            findings=[],
            report=FactCheckReport(
                claim="The Eiffel Tower was built in 1889.",
                verdict=VeracityVerdict.SUPPORTED,
                verdict_confidence=0.8,
                justification="Supported.",
                key_points=[],
                findings=[],
                sources=[],
            ),
            artifacts=RunArtifacts(
                claim="The Eiffel Tower was built in 1889.",
                normalized_claim="The Eiffel Tower was built in 1889.",
            ),
        )

        fake_service = AsyncMock()
        fake_service.check_claim = AsyncMock(return_value=fake_run)
        callback = lambda _event: None

        with patch("facticli.orchestrator.build_fact_check_service", return_value=fake_service):
            orchestrator = FactCheckOrchestrator(OrchestratorConfig(inference_provider="openai-agents"))
            run = await orchestrator.check_claim(
                "The Eiffel Tower was built in 1889.",
                progress_callback=callback,
            )

        self.assertEqual(run.report.verdict, VeracityVerdict.SUPPORTED)
        fake_service.check_claim.assert_awaited_once_with(
            "The Eiffel Tower was built in 1889.",
            progress_callback=callback,
        )

    async def test_latest_artifacts_returns_none_before_any_run(self):
        fake_service = AsyncMock()
        fake_service.check_claim = AsyncMock()

        with patch("facticli.orchestrator.build_fact_check_service", return_value=fake_service):
            orchestrator = FactCheckOrchestrator(OrchestratorConfig(inference_provider="openai-agents"))

        self.assertIsNone(orchestrator.latest_artifacts())


if __name__ == "__main__":
    unittest.main()
