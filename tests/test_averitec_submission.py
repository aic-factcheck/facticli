from __future__ import annotations

import asyncio
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import openai

from facticli.averitec_submission import (
    EXIT_ABORTED,
    _run,
    build_failed_submission_row,
    build_parser,
    build_submission_evidence,
    build_submission_row,
)
from facticli.core.artifacts import RunArtifacts
from facticli.core.contracts import (
    AspectFinding,
    EvidenceSignal,
    FactCheckReport,
    FindingStatus,
    SourceEvidence,
    SourceGrounding,
    VeracityVerdict,
)


def _make_report() -> FactCheckReport:
    primary_source = SourceEvidence(
        title="Primary Source",
        url="https://example.com/source-a",
        snippet="Primary snippet",
    )
    secondary_source = SourceEvidence(
        title="Secondary Source",
        url="https://example.com/source-b",
        snippet="Secondary snippet",
    )
    finding = AspectFinding(
        aspect_id="aspect_1",
        question="Did the event happen in 2020?",
        signal=EvidenceSignal.SUPPORTS,
        summary="Multiple records confirm the event happened in 2020.",
        confidence=0.8,
        sources=[primary_source],
        caveats=[],
    )
    return FactCheckReport(
        claim="An event happened in 2020.",
        verdict=VeracityVerdict.SUPPORTED,
        verdict_confidence=0.8,
        justification="Available sources support the claim.",
        key_points=[],
        findings=[finding],
        sources=[primary_source, secondary_source],
    )


class AveritecSubmissionMappingTests(unittest.TestCase):
    def test_build_submission_row_populates_expected_fields(self):
        report = _make_report()
        row = build_submission_row(
            record={"claim": "An event happened in 2020.", "id": 42},
            row_index=7,
            claim_field="claim",
            claim_id_field=None,
            report=report,
            max_evidence=10,
            empty_question=False,
        )

        self.assertEqual(row["claim_id"], 42)
        self.assertEqual(row["claim"], "An event happened in 2020.")
        self.assertEqual(row["pred_label"], "Supported")
        self.assertEqual(row["evidence"][0]["question"], "Did the event happen in 2020?")
        self.assertIn("happened in 2020", row["evidence"][0]["answer"])
        self.assertEqual(row["evidence"][0]["url"], "https://example.com/source-a")
        self.assertEqual(row["evidence"][0]["scraped_text"], "Primary snippet")

    def test_build_submission_row_uses_row_index_when_no_claim_id(self):
        report = _make_report()
        row = build_submission_row(
            record={"claim": "An event happened in 2020."},
            row_index=123,
            claim_field="claim",
            claim_id_field=None,
            report=report,
            max_evidence=10,
            empty_question=False,
        )
        self.assertEqual(row["claim_id"], 123)

    def test_build_submission_evidence_can_force_empty_questions(self):
        report = _make_report()
        evidence = build_submission_evidence(
            report,
            max_evidence=10,
            empty_question=True,
        )
        self.assertGreaterEqual(len(evidence), 1)
        self.assertTrue(all(item["question"] == "" for item in evidence))

    def test_build_submission_evidence_deduplicates_and_respects_limit(self):
        report = _make_report()
        evidence = build_submission_evidence(
            report,
            max_evidence=1,
            empty_question=False,
        )
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0]["url"], "https://example.com/source-a")

    def test_evidence_rows_each_state_a_different_fact(self):
        def source(name: str) -> SourceEvidence:
            return SourceEvidence(title=name, url=f"https://example.com/{name}", snippet=f"Snippet  from {name}")

        findings = [
            AspectFinding(
                aspect_id="first", question="Q1?", signal=EvidenceSignal.SUPPORTS, summary="Summary one.", confidence=0.8,
                sources=[source("a1"), source("a2"), source("a3"), source("a4")],
            ),
            AspectFinding(
                aspect_id="second", question="Q2?", signal=EvidenceSignal.REFUTES, summary="Summary two.", confidence=0.7,
                sources=[source("b1")],
            ),
            AspectFinding(
                aspect_id="failed", question="Q3?", signal=EvidenceSignal.INSUFFICIENT, summary="Research failed.", confidence=0.0,
                status=FindingStatus.FAILED,
            ),
        ]
        report = FactCheckReport(
            claim="c", verdict=VeracityVerdict.CONFLICTING, verdict_confidence=0.5, justification="j", findings=findings,
            sources=[source("a1"), source("extra")],
        )

        # Every finding is heard once before any is heard twice, so a source-rich finding cannot crowd the others out.
        evidence = build_submission_evidence(report, max_evidence=5, empty_question=False)
        self.assertEqual(
            [(row["question"], row["answer"]) for row in evidence],
            [
                ("Q1?", "Summary one."),
                ("Q2?", "Summary two."),
                ("Q1?", "Snippet from a1"),
                ("Q2?", "Snippet from b1"),
                ("Q1?", "Snippet from a2"),
            ],
        )
        self.assertEqual(evidence[2]["url"], "https://example.com/a1")

        everything = build_submission_evidence(report, max_evidence=20, empty_question=True)
        answers = [row["answer"] for row in everything]
        self.assertEqual(len(answers), len(set(answers)))  # the report-level copy of a1 adds nothing
        self.assertEqual(answers[-1], "Snippet from extra")  # report-level sources fill what is left
        self.assertNotIn("Research failed.", answers)  # a failed check is not evidence

    def test_build_failed_submission_row_uses_not_enough_evidence(self):
        row = build_failed_submission_row(
            record={"claim": "An event happened in 2020."},
            row_index=8,
            claim_field="claim",
            claim_id_field=None,
        )
        self.assertEqual(row["claim_id"], 8)
        self.assertEqual(row["pred_label"], "Not Enough Evidence")
        self.assertEqual(row["evidence"], [])


class _BatchService:
    """Succeeds on the first claim, then fails every later one with the given error."""

    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls = 0

    async def check_claim(self, claim: str, claim_id=None, claim_date=None):
        self.calls += 1
        if self.calls > 1:
            raise self.error
        report = _make_report()
        report.sources[0].grounding = SourceGrounding.QUOTE_MATCHED
        return SimpleNamespace(report=report, artifacts=RunArtifacts(claim=claim, normalized_claim=claim, report_final=report))


class BatchRunTests(unittest.TestCase):
    def _run_batch(self, service, records: list[dict] | None = None, extra_args: tuple[str, ...] = ()):
        """Run the batch CLI over four claims; return (exit code, rows, manifest, stderr)."""
        records = records or [{"claim": f"Claim {index}."} for index in range(4)]
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "claims.json"
            input_path.write_text(json.dumps(records), encoding="utf-8")
            output_path = Path(tmp) / "submission.json"
            manifest_path = Path(tmp) / "submission.runinfo.json"
            args = build_parser().parse_args(["--input", str(input_path), "--output", str(output_path), *extra_args])
            stderr = io.StringIO()
            with (
                patch.dict("os.environ", {"OPENAI_API_KEY": "k", "OPENAI_API_MODEL": "m"}, clear=False),
                patch("facticli.averitec_submission.build_fact_checker", side_effect=[service] if service else ValueError("bad config")),
                redirect_stderr(stderr),
            ):
                code = asyncio.run(_run(args))
            rows = json.loads(output_path.read_text(encoding="utf-8")) if output_path.exists() else None
            manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
        return code, rows, manifest, stderr.getvalue()

    def test_a_fatal_error_stops_the_batch_instead_of_writing_fallback_labels(self):
        response = httpx.Response(429, request=httpx.Request("POST", "https://api.example"))
        quota = openai.RateLimitError("You exceeded your current quota", response=response, body={"code": "insufficient_quota"})
        service = _BatchService(quota)
        code, rows, manifest, stderr = self._run_batch(service)

        self.assertEqual(code, EXIT_ABORTED)
        # Only the claim that really ran is written, so --resume retries the rest.
        self.assertEqual([(row["claim_id"], row["pred_label"]) for row in rows], [(0, "Supported")])
        self.assertEqual(service.calls, 2)  # claims after the fatal error were not attempted
        self.assertTrue(manifest["aborted"].startswith("quota: RateLimitError"))
        self.assertEqual(manifest["claims"][0]["grounding"], {"quote_matched": 1})
        self.assertIn("rerun with --resume", stderr)

    def test_ordinary_failures_get_fallback_rows_and_the_batch_goes_on(self):
        service = _BatchService(RuntimeError("flaky"))
        code, rows, manifest, _ = self._run_batch(service, extra_args=("--tool-call-budget", "5", "--search-results", "9"))
        self.assertEqual(code, 0)
        self.assertEqual([row["pred_label"] for row in rows], ["Supported", *["Not Enough Evidence"] * 3])
        self.assertEqual(service.calls, 4)
        self.assertNotIn("aborted", manifest)
        self.assertEqual((manifest["settings"]["tool_call_budget"], manifest["settings"]["search_results"]), (5, 9))

    def test_fail_fast_stops_but_keeps_the_rows_already_finished(self):
        service = _BatchService(RuntimeError("flaky"))
        code, rows, manifest, stderr = self._run_batch(service, extra_args=("--fail-fast",))
        self.assertEqual(code, 1)
        self.assertEqual([(row["claim_id"], row["pred_label"]) for row in rows], [(0, "Supported")])
        self.assertEqual(service.calls, 2)
        self.assertIn("--fail-fast", manifest["aborted"])
        self.assertIn("rerun with --resume", stderr)

    def test_problems_known_before_the_first_claim_exit_without_spending(self):
        records = [{"claim": "Claim 0.", "claim_date": "25-8-2020"}, {"claim": "Claim 1.", "claim_date": "8/25/2020"}]
        service = _BatchService(RuntimeError("unused"))
        code, rows, _, stderr = self._run_batch(service, records, ("--claim-date-field", "claim_date"))
        self.assertEqual((code, rows, service.calls), (2, None, 0))
        self.assertIn("Unparseable claim date in field 'claim_date' of 1 row(s), first at row 1", stderr)

        code, rows, _, stderr = self._run_batch(None)  # the service cannot be built from these settings
        self.assertEqual((code, rows), (2, None))
        self.assertIn("bad config", stderr)


if __name__ == "__main__":
    unittest.main()
