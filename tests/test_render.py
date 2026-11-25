from __future__ import annotations

import unittest

from facticli.pipeline import FactCheckRun
from facticli.core.artifacts import RunArtifacts
from facticli.core.contracts import (
    AspectFinding,
    EvidenceSignal,
    FactCheckReport,
    FindingStatus,
    InvestigationPlan,
    SourceEvidence,
    SourceGrounding,
    SourceTier,
    StopReason,
    VeracityVerdict,
    VerificationCheck,
)
from facticli.core.retrieval import RetrievalSummary
from facticli.core.usage import UsageSummary
from facticli.render import format_run_text


class RenderTests(unittest.TestCase):
    def _run(self) -> FactCheckRun:
        source = SourceEvidence(
            title="Reuters", url="https://www.reuters.com/world/x", snippet="quote", published_at="2020-01-02",
            tier=SourceTier.NEWS,
        )
        findings = [
            AspectFinding(aspect_id="a", question="Qa", signal=EvidenceSignal.SUPPORTS, summary="sa", confidence=0.9, sources=[source]),
            AspectFinding(
                aspect_id="b", question="Qb", signal=EvidenceSignal.INSUFFICIENT, summary="f", confidence=0.0,
                status=FindingStatus.FAILED, failure_reason="timeout: TimeoutError",
            ),
        ]
        report = FactCheckReport(
            claim="c", verdict=VeracityVerdict.SUPPORTED, verdict_confidence=0.8, justification="j [S1]",
            key_points=["k"], counter_argument="It could be refuted because ...", evidence_gaps=["missing primary"],
            findings=findings, sources=[source],
        )
        artifacts = RunArtifacts(
            claim="c", normalized_claim="c", stage_models={"plan": "m", "judge": "big"},
            duration_seconds=12.3,
            usage_summary=UsageSummary(requests=5, input_tokens=1000, cached_input_tokens=250, output_tokens=200, total_tokens=1200, cache_hit_rate=0.25, per_stage={"plan": {"total_tokens": 300, "calls": 1, "llm_seconds": 2.0}}),
        )
        plan = InvestigationPlan(claim="c", checks=[VerificationCheck(aspect_id="a", question="Qa", rationale="r", search_queries=["q"], acceptance_criteria=["done when x"])])
        return FactCheckRun(claim="c", plan=plan, findings=findings, report=report, artifacts=artifacts)

    def test_text_output_covers_new_fields(self):
        text = format_run_text(self._run(), show_plan=True, show_usage=True)
        self.assertIn("Counter-argument Considered", text)
        self.assertIn("Evidence Gaps", text)
        self.assertIn("done when: done when x", text)
        self.assertIn("[b] failed | not an observation", text)
        self.assertIn("reason: timeout: TimeoutError", text)
        self.assertIn("[S1] Reuters (news, 2020-01-02)", text)
        self.assertIn("Run Stats", text)
        self.assertIn("models: plan=m, judge=big", text)
        self.assertIn("cache hit rate 25%", text)

    def test_provenance_limits_and_wasted_tokens_are_visible(self):
        run = self._run()
        quoted = run.report.sources[0].model_copy(update={"grounding": SourceGrounding.QUOTE_MATCHED})
        unconfirmed = quoted.model_copy(update={"title": "AP", "url": "https://apnews.com/x", "grounding": SourceGrounding.URL_MATCHED})
        invented = quoted.model_copy(update={"title": "Blog", "url": "https://blog.example/x", "grounding": SourceGrounding.UNMATCHED})
        run.report.sources[:] = [quoted, unconfirmed, invented]
        run.report.findings[0] = run.report.findings[0].model_copy(update={"stop_reason": StopReason.TURN_LIMIT})
        run.artifacts.unresolved_citations = [7]
        run.artifacts.retrieval_summary = RetrievalSummary(searches=5, pages_opened=1, cache_hits=2, refused=1, errors=1, distinct_urls=12)
        run.artifacts.usage_summary = UsageSummary(
            requests=6, input_tokens=1000, cached_input_tokens=250, cache_write_tokens=400, output_tokens=200, total_tokens=1200,
            cache_hit_rate=0.25, failed_attempts=1, failed_attempt_tokens=300, wrap_up_calls=1,
            per_stage={"research": {"total_tokens": 900, "calls": 3, "llm_seconds": 4.0, "failed_attempts": 1, "failed_attempt_tokens": 300}},
        )
        text = format_run_text(run, show_usage=True)
        self.assertIn("[S1] Reuters (news, 2020-01-02, quote found in retrieved text)", text)
        self.assertIn("[S2] AP (news, 2020-01-02, url retrieved, quote not matched)", text)
        self.assertIn("[S3] Blog (news, 2020-01-02, NOT in search results)", text)
        self.assertIn("[a] supports | confidence 0.90 | cut short at the turn limit", text)
        self.assertIn("250 cached, 400 cache writes,", text)
        self.assertIn("research: 900 tokens, 3 call(s), 4.0s, 1 failed attempt(s) costing 300 tokens", text)
        self.assertIn("wrap-up calls at the turn limit: 1", text)
        self.assertIn(
            "retrieval: 5 search(es), 1 page(s) opened, 12 distinct URL(s), 2 served from cache, 1 refused at the cap, 1 failed", text
        )
        # A dangling citation is a problem with the report, so it shows without --show-usage.
        self.assertIn("cited above but absent from this list: S7", format_run_text(run))

    def test_usage_hidden_by_default(self):
        text = format_run_text(self._run())
        self.assertNotIn("Run Stats", text)
        self.assertNotIn("Plan", text.split("Findings")[0].split("Justification")[1])


if __name__ == "__main__":
    unittest.main()
