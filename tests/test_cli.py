from __future__ import annotations

import argparse
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import AsyncMock, patch

from facticli.cli import _load_extract_input_text, build_parser, run_check_command, run_extract_claims_command
from facticli.core.artifacts import RunArtifacts
from facticli.pipeline import FactCheckRun
from facticli.core.contracts import (
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
        args = build_parser().parse_args(
            ["extract-claims", "Inflation fell below 3%.", "--model", "gpt-5.4", "--max-claims", "10", "--json"]
        )

        fake_service = AsyncMock()
        fake_service.extract_claims = AsyncMock(return_value=fake_result)

        with (
            patch.dict("os.environ", {"OPENAI_API_KEY": "dummy", "OPENAI_API_MODEL": "gpt-5.4"}, clear=False),
            patch("facticli.cli.build_claim_extractor", return_value=fake_service),
        ):
            output = io.StringIO()
            with redirect_stdout(output):
                code = await run_extract_claims_command(args)

        self.assertEqual(code, 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["claims"][0]["claim_id"], "c1")

    async def test_check_command_rejects_missing_openai_api_key(self):
        args = build_parser().parse_args(
            ["check", "Some claim", "--model", "gpt-5.4", "--parallel", "2", "--search-provider", "openai"]
        )

        with patch.dict("os.environ", {}, clear=True):
            code = await run_check_command(args)

        self.assertEqual(code, 2)

    async def test_check_command_rejects_knowledge_store_it_cannot_point_at(self):
        with patch.dict("os.environ", {"FACTICLI_SEARCH_PROVIDER": "knowledge_store"}, clear=False):
            args = build_parser().parse_args(["check", "Some claim", "--model", "m"])
        self.assertEqual(args.search_provider, "knowledge_store")  # argparse does not check defaults
        stderr = io.StringIO()
        with patch.dict("os.environ", {"OPENAI_API_KEY": "k"}, clear=False), redirect_stderr(stderr):
            code = await run_check_command(args)
        self.assertEqual(code, 2)
        self.assertIn("--knowledge-store-dir", stderr.getvalue())
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            build_parser().parse_args(["check", "Some claim", "--search-provider", "knowledge_store"])

    async def test_extract_claims_command_rejects_missing_model(self):
        args = build_parser().parse_args(["extract-claims", "Inflation fell below 3%.", "--max-claims", "10", "--json"])

        with patch.dict("os.environ", {"OPENAI_API_KEY": "dummy"}, clear=True):
            code = await run_extract_claims_command(args)

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

        fake_service = AsyncMock()
        fake_service.check_claim = AsyncMock(return_value=fake_run)

        args = build_parser().parse_args(
            [
                "check", "The first iPhone was released in 2007.", "--model", "gpt-5.4", "--parallel", "2",
                "--search-provider", "openai", "--json", "--include-artifacts",
            ]
        )

        with (
            patch.dict("os.environ", {"OPENAI_API_KEY": "dummy", "OPENAI_API_MODEL": "gpt-5.4"}, clear=False),
            patch("facticli.cli.build_fact_checker", return_value=fake_service),
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

    async def test_check_command_maps_search_flags_onto_the_runtime_config(self):
        args = build_parser().parse_args(
            [
                "check", "claim", "--search-provider", "brave", "--tool-call-budget", "5", "--search-cache-dir", "cache",
                "--stage-effort", "judge=xhigh", "--search-results", "9", "--json",
            ]
        )
        captured: dict[str, object] = {}

        def fake_build(config, artifacts_dir=None):
            captured["config"] = config
            raise ValueError("stop here")

        with (
            patch.dict("os.environ", {"OPENAI_API_KEY": "dummy", "OPENAI_API_MODEL": "m", "BRAVE_SEARCH_API_KEY": "k"}, clear=False),
            patch("facticli.cli.build_fact_checker", side_effect=fake_build),
            redirect_stderr(io.StringIO()),
        ):
            code = await run_check_command(args)

        self.assertEqual(code, 2)
        config = captured["config"]
        self.assertEqual((config.tool_call_budget, config.search_cache_dir, config.search_results_per_query), (5, "cache", 9))
        self.assertEqual(config.stage_efforts, {"judge": "xhigh"})
        defaults = build_parser().parse_args(["check", "claim"])
        self.assertEqual((defaults.tool_call_budget, defaults.search_cache_dir), (8, None))

    def test_removed_flags_are_gone(self):
        for flag in ("--strategy", "--feedback-rounds", "--token-budget", "--retry-budget", "--verify-sources"):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                build_parser().parse_args(["check", "claim", flag, "1"])
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            build_parser().parse_args(["--profile", "x", "check", "claim"])

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

        class _FakeService:
            async def check_claim(self, _claim: str, progress=None):
                if progress is not None:
                    progress("[progress] Plan ready with 1 check(s):")
                    progress("  - [timeline_1] Was it completed in 1889?")
                return fake_run

        args = build_parser().parse_args(
            [
                "check", "The Eiffel Tower was built in 1889.", "--model", "gpt-5.4", "--parallel", "2",
                "--search-provider", "openai", "--stream-progress",
            ]
        )

        with (
            patch.dict("os.environ", {"OPENAI_API_KEY": "dummy", "OPENAI_API_MODEL": "gpt-5.4"}, clear=False),
            patch("facticli.cli.build_fact_checker", return_value=_FakeService()),
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
