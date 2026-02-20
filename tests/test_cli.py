from __future__ import annotations

import argparse
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import AsyncMock, patch

from facticli.application.progress import ProgressEvent
from facticli.cli import _load_extract_input_text, run_check_command, run_extract_claims_command
from facticli.core.artifacts import RunArtifacts
from facticli.orchestrator import FactCheckRun
from facticli.types import (
    CheckworthyClaim,
    ClaimExtractionResult,
    FactCheckReport,
    InvestigationPlan,
    SourceEvidence,
    VeracityVerdict,
)


class CLITests(unittest.IsolatedAsyncioTestCase):
    async def test_extract_claims_json_output(self):
        fake_result = ClaimExtractionResult(
            input_text="text",
            claims=[
                CheckworthyClaim(
                    claim_id="c1",
                    claim_text="Inflation fell below 3%.",
                    source_fragment="inflation fell below 3%",
                    checkworthy_reason="Numeric macro claim.",
                )
            ],
        )
        fake_extractor = AsyncMock()
        fake_extractor.extract = AsyncMock(return_value=fake_result)

        args = argparse.Namespace(
            inference_provider="openai-agents",
            model="gpt-4.1-mini",
            gemini_model="gemini-3-pro",
            max_claims=10,
            from_file=None,
            text="Inflation fell below 3%.",
            json=True,
        )

        with (
            patch.dict("os.environ", {"OPENAI_API_KEY": "dummy"}, clear=False),
            patch("facticli.cli.ClaimExtractor", return_value=fake_extractor),
        ):
            output = io.StringIO()
            with redirect_stdout(output):
                code = await run_extract_claims_command(args)

        self.assertEqual(code, 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["claims"][0]["claim_id"], "c1")

    async def test_check_command_rejects_gemini_without_brave_search(self):
        args = argparse.Namespace(
            claim="Some claim",
            inference_provider="gemini",
            model="gpt-4.1-mini",
            gemini_model="gemini-2.0-flash",
            max_checks=4,
            parallel=2,
            search_provider="openai",
            search_context_size="high",
            search_results_per_query=5,
            show_plan=False,
            json=False,
            include_artifacts=False,
        )

        with patch.dict("os.environ", {"GEMINI_API_KEY": "dummy"}, clear=False):
            code = await run_check_command(args)

        self.assertEqual(code, 2)

    async def test_check_command_json_with_artifacts(self):
        fake_run = FactCheckRun(
            claim="The first iPhone was released in 2007.",
            plan=InvestigationPlan(
                claim="The first iPhone was released in 2007.",
                checks=[],
                assumptions=[],
            ),
            findings=[],
            report=FactCheckReport(
                claim="The first iPhone was released in 2007.",
                verdict=VeracityVerdict.SUPPORTED,
                verdict_confidence=0.9,
                justification="Apple release records confirm 2007 launch.",
                key_points=["Launch date is widely documented."],
                findings=[],
                sources=[
                    SourceEvidence(
                        title="Apple Newsroom archive",
                        url="https://example.org/apple",
                        snippet="iPhone announced in 2007.",
                    )
                ],
            ),
            artifacts=RunArtifacts(
                claim="The first iPhone was released in 2007.",
                normalized_claim="The first iPhone was released in 2007.",
            ),
        )

        fake_orchestrator = AsyncMock()
        fake_orchestrator.check_claim = AsyncMock(return_value=fake_run)

        args = argparse.Namespace(
            claim="The first iPhone was released in 2007.",
            inference_provider="openai-agents",
            model="gpt-4.1-mini",
            gemini_model="gemini-2.0-flash",
            max_checks=4,
            parallel=2,
            search_provider="openai",
            search_context_size="high",
            search_results_per_query=5,
            show_plan=False,
            json=True,
            include_artifacts=True,
        )

        with (
            patch.dict("os.environ", {"OPENAI_API_KEY": "dummy"}, clear=False),
            patch("facticli.cli.FactCheckOrchestrator", return_value=fake_orchestrator),
        ):
            output = io.StringIO()
            with redirect_stdout(output):
                code = await run_check_command(args)

        self.assertEqual(code, 0)
        payload = json.loads(output.getvalue())
        self.assertIn("report", payload)
        self.assertIn("plan", payload)
        self.assertIn("findings", payload)
        self.assertIn("artifacts", payload)

    async def test_check_command_streams_progress_to_stderr(self):
        fake_run = FactCheckRun(
            claim="The Eiffel Tower was built in 1889.",
            plan=InvestigationPlan(claim="The Eiffel Tower was built in 1889.", checks=[], assumptions=[]),
            findings=[],
            report=FactCheckReport(
                claim="The Eiffel Tower was built in 1889.",
                verdict=VeracityVerdict.SUPPORTED,
                verdict_confidence=0.9,
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

        class _FakeOrchestrator:
            async def check_claim(self, _claim: str, progress_callback=None):
                if progress_callback is not None:
                    progress_callback(
                        ProgressEvent(
                            kind="planning_completed",
                            payload={
                                "check_count": 1,
                                "checks": [
                                    {
                                        "aspect_id": "timeline_1",
                                        "question": "Was it completed in 1889?",
                                    }
                                ],
                            },
                        )
                    )
                return fake_run

        args = argparse.Namespace(
            claim="The Eiffel Tower was built in 1889.",
            inference_provider="openai-agents",
            model="gpt-4.1-mini",
            gemini_model="gemini-2.0-flash",
            max_checks=4,
            parallel=2,
            search_provider="openai",
            search_context_size="high",
            search_results_per_query=5,
            show_plan=False,
            json=False,
            include_artifacts=False,
            stream_progress=True,
        )

        with (
            patch.dict("os.environ", {"OPENAI_API_KEY": "dummy"}, clear=False),
            patch("facticli.cli.FactCheckOrchestrator", return_value=_FakeOrchestrator()),
        ):
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = await run_check_command(args)

        self.assertEqual(code, 0)
        self.assertIn("[progress] Plan ready", stderr.getvalue())
        self.assertIn("[timeline_1] Was it completed in 1889?", stderr.getvalue())


class LoadExtractInputTextTests(unittest.TestCase):
    def test_load_extract_input_text_from_argument(self):
        args = argparse.Namespace(text="Some text", from_file=None)
        self.assertEqual(_load_extract_input_text(args), "Some text")

    def test_load_extract_input_text_from_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "input.txt"
            path.write_text("Transcript text", encoding="utf-8")

            args = argparse.Namespace(text=None, from_file=str(path))
            self.assertEqual(_load_extract_input_text(args), "Transcript text")

    def test_load_extract_input_text_requires_input(self):
        args = argparse.Namespace(text=None, from_file=None)
        with self.assertRaises(ValueError):
            _load_extract_input_text(args)

    def test_load_extract_input_text_rejects_ambiguous_input(self):
        args = argparse.Namespace(text="Inline text", from_file="input.txt")
        with self.assertRaises(ValueError):
            _load_extract_input_text(args)

    def test_load_extract_input_text_rejects_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = argparse.Namespace(text=None, from_file=tmpdir)
            with self.assertRaises(ValueError):
                _load_extract_input_text(args)


if __name__ == "__main__":
    unittest.main()
