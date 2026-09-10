from __future__ import annotations

import unittest

from facticli.application.services import FactCheckRun
from facticli.core.artifacts import RunArtifacts
from facticli.core.contracts import (
    AspectFinding,
    EvidenceSignal,
    FactCheckReport,
    FindingStatus,
    InvestigationPlan,
    SourceEvidence,
    SourceTier,
    UrlStatus,
    VeracityVerdict,
    VerificationCheck,
)
from facticli.core.usage import BudgetStatus, UsageSummary
from facticli.render import format_run_text


class RenderTests(unittest.TestCase):
    def _run(self) -> FactCheckRun:
        source = SourceEvidence(
            title="Reuters", url="https://www.reuters.com/world/x", snippet="quote", published_at="2020-01-02",
            tier=SourceTier.NEWS, url_status=UrlStatus.ARCHIVED, archived_url="https://web.archive.org/web/x",
        )
        findings = [
            AspectFinding(aspect_id="a", question="Qa", signal=EvidenceSignal.SUPPORTS, summary="sa", confidence=0.9, sources=[source]),
            AspectFinding(
                aspect_id="b", question="Qb", signal=EvidenceSignal.INSUFFICIENT, summary="f", confidence=0.0,
                status=FindingStatus.BUDGET_EXHAUSTED, failure_reason="token budget exhausted",
            ),
        ]
        report = FactCheckReport(
            claim="c", verdict=VeracityVerdict.SUPPORTED, verdict_confidence=0.8, justification="j [S1]",
            key_points=["k"], counter_argument="It could be refuted because ...", evidence_gaps=["missing primary"],
            findings=findings, sources=[source],
        )
        artifacts = RunArtifacts(
            claim="c", normalized_claim="c", strategy="pipeline", stage_models={"plan": "m", "judge": "big"},
            duration_seconds=12.3,
            usage_summary=UsageSummary(requests=5, input_tokens=1000, cached_input_tokens=250, output_tokens=200, total_tokens=1200, cache_hit_rate=0.25, per_stage={"plan": {"total_tokens": 300, "calls": 1, "llm_seconds": 2.0}}),
            budget=BudgetStatus(token_budget=1000, tokens_used=1200, token_budget_exhausted=True, retry_budget=12, retries_used=1, skipped_stages=["research:b"]),
            citation_check={"archived": 1},
        )
        plan = InvestigationPlan(claim="c", checks=[VerificationCheck(aspect_id="a", question="Qa", rationale="r", search_queries=["q"], acceptance_criteria=["done when x"])])
        return FactCheckRun(claim="c", plan=plan, findings=findings, report=report, artifacts=artifacts)

    def test_text_output_covers_new_fields(self):
        text = format_run_text(self._run(), show_plan=True, show_usage=True)
        self.assertIn("Counter-argument Considered", text)
        self.assertIn("Evidence Gaps", text)
        self.assertIn("done when: done when x", text)
        self.assertIn("[b] skipped (budget) | not an observation", text)
        self.assertIn("[S1] Reuters (news, 2020-01-02, archived)", text)
        self.assertIn("archived: https://web.archive.org/web/x", text)
        self.assertIn("Run Stats", text)
        self.assertIn("models: plan=m, judge=big", text)
        self.assertIn("cache hit rate 25%", text)
        self.assertIn("token budget 1200/1000 (exhausted)", text)
        self.assertIn("skipped: research:b", text)
        self.assertIn("citations: archived=1", text)

    def test_usage_hidden_by_default(self):
        text = format_run_text(self._run())
        self.assertNotIn("Run Stats", text)
        self.assertNotIn("Plan", text.split("Findings")[0].split("Justification")[1])


if __name__ == "__main__":
    unittest.main()
