from __future__ import annotations

import unittest

from facticli.averitec_submission import (
    build_failed_submission_row,
    build_submission_evidence,
    build_submission_row,
)
from facticli.types import (
    AspectFinding,
    EvidenceSignal,
    FactCheckReport,
    SourceEvidence,
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


if __name__ == "__main__":
    unittest.main()
