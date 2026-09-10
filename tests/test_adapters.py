from __future__ import annotations

import unittest

from agents import ModelRetryNormalizedError, RetryPolicyContext

from facticli.adapters.openai_provider import build_model_settings
from facticli.adapters.payloads import (
    FAILED_CHECK_NOTICE,
    budget_payload,
    index_sources,
    judge_payload,
    research_payload,
    review_payload,
    single_agent_payload,
)
from facticli.adapters.retry_policy import build_retry_settings, classify_retry_context, facticli_retry_policy
from facticli.core.constraints import ResearchConstraints
from facticli.core.contracts import (
    AspectFinding,
    EvidenceSignal,
    FindingStatus,
    InvestigationPlan,
    SourceEvidence,
    SourceTier,
    VerificationCheck,
)
from facticli.core.errors import ErrorKind
from facticli.core.usage import UsageLog, activate_usage_log, deactivate_usage_log


def _check(aspect_id: str, criteria: list[str] | None = None) -> VerificationCheck:
    return VerificationCheck(
        aspect_id=aspect_id,
        question=f"Question {aspect_id}?",
        rationale="r",
        search_queries=["q"],
        acceptance_criteria=criteria or [],
    )


def _source(url: str, tier: SourceTier | None = None) -> SourceEvidence:
    return SourceEvidence(title=url, url=url, snippet=f"snippet from {url}", published_at="2020-01-01", tier=tier)


class PayloadTests(unittest.TestCase):
    def test_judge_payload_groups_by_signal_and_indexes_sources(self):
        plan = InvestigationPlan(claim="c", checks=[_check("a", ["done when x"]), _check("b"), _check("c")])
        shared = _source("https://www.bbc.com/news/1?utm_source=x", SourceTier.NEWS)
        findings = [
            AspectFinding(
                aspect_id="a", question="Qa", signal=EvidenceSignal.SUPPORTS, summary="sa", confidence=0.9,
                sources=[shared, _source("https://data.gov/x", SourceTier.PRIMARY)],
            ),
            AspectFinding(
                aspect_id="b", question="Qb", signal=EvidenceSignal.REFUTES, summary="sb", confidence=0.6,
                sources=[SourceEvidence(title="dup", url="https://www.bbc.com/news/1/", snippet="same page", tier=SourceTier.NEWS)],
            ),
            AspectFinding(
                aspect_id="c", question="Qc", signal=EvidenceSignal.INSUFFICIENT, summary="failed", confidence=0.0,
                status=FindingStatus.FAILED, failure_reason="timeout: TimeoutError",
            ),
        ]
        payload = judge_payload(claim="c", plan=plan, findings=findings)

        self.assertEqual(payload["evidence_overview"]["checks_completed"], 2)
        self.assertEqual(payload["evidence_overview"]["checks_failed"], 1)
        self.assertEqual(payload["evidence_overview"]["signal_counts"]["supports"], 1)
        self.assertEqual(payload["evidence_overview"]["signal_counts"]["refutes"], 1)
        self.assertEqual(payload["evidence_overview"]["signal_counts"]["insufficient"], 0)
        self.assertEqual(payload["evidence_overview"]["source_tier_counts"], {"news": 1, "primary": 1})
        # Shared source deduplicated across findings with a stable id and cited_by both.
        self.assertEqual([row["id"] for row in payload["sources"]], ["S1", "S2"])
        self.assertEqual(payload["sources"][0]["cited_by"], ["a", "b"])
        self.assertEqual(payload["findings_by_signal"]["supports"][0]["source_ids"], ["S1", "S2"])
        self.assertEqual(payload["findings_by_signal"]["supports"][0]["acceptance_criteria"], ["done when x"])
        self.assertEqual(payload["findings_by_signal"]["refutes"][0]["source_ids"], ["S1"])
        self.assertEqual(payload["failed_checks"][0]["aspect_id"], "c")
        self.assertEqual(payload["failed_checks_notice"], FAILED_CHECK_NOTICE)
        self.assertIn("[S2]", payload["citation_rule"])

    def test_judge_payload_without_failures_omits_notice(self):
        plan = InvestigationPlan(claim="c", checks=[_check("a")])
        findings = [AspectFinding(aspect_id="a", question="Q", signal=EvidenceSignal.MIXED, summary="s", confidence=0.5)]
        payload = judge_payload(claim="c", plan=plan, findings=findings)
        self.assertNotIn("failed_checks", payload)
        self.assertEqual(index_sources(findings), ([], {}))

    def test_review_payload_pairs_checks_with_findings(self):
        plan = InvestigationPlan(claim="c", checks=[_check("a", ["crit"]), _check("missing")])
        findings = [
            AspectFinding(
                aspect_id="a", question="Q", signal=EvidenceSignal.SUPPORTS, summary="s", confidence=0.8,
                sources=[_source("https://x.org", SourceTier.OTHER)],
            )
        ]
        payload = review_payload(claim="c", plan=plan, findings=findings, max_follow_up_checks=3, round_index=2)
        self.assertEqual(payload["limits"], {"max_follow_up_checks": 3})
        self.assertEqual(payload["round_index"], 2)
        self.assertEqual(payload["checks"][0]["status"], "completed")
        self.assertEqual(payload["checks"][0]["acceptance_criteria"], ["crit"])
        self.assertEqual(payload["checks"][0]["finding"]["sources"][0]["tier"], "other")
        self.assertEqual(payload["checks"][1]["status"], "missing")
        self.assertIsNone(payload["checks"][1]["finding"])

    def test_research_and_single_agent_payloads_carry_constraints_and_budget(self):
        constraints = ResearchConstraints(claim_date="2020-10-31", blocked_domains=["politifact.com"])
        log = UsageLog(token_budget=5000)
        payload = research_payload(
            claim="c", check=_check("a"), search_provider="brave", constraints=constraints, usage_log=log, tool_call_budget=6
        )
        self.assertEqual(payload["constraints"]["evidence_cutoff_date"], "2020-10-31")
        self.assertEqual(payload["budget"]["tokens_remaining"], 5000)
        self.assertEqual(payload["requirements"]["tool_call_budget"], 6)
        self.assertIn("acceptance_criteria", payload["check"])

        bare = research_payload(claim="c", check=_check("a"), search_provider="openai", constraints=None, usage_log=None, tool_call_budget=8)
        self.assertNotIn("constraints", bare)
        self.assertNotIn("budget", bare)
        self.assertIsNone(budget_payload(UsageLog()))

        single = single_agent_payload(claim="c", max_checks=3, search_provider="openai", constraints=constraints, usage_log=None)
        self.assertEqual(single["requirements"]["max_aspects"], 3)
        self.assertIn("constraints", single)


class ModelSettingsTests(unittest.TestCase):
    def test_build_model_settings(self):
        settings = build_model_settings(parallel_tool_calls=True, reasoning_effort="low", retry_attempts=3)
        self.assertTrue(settings.parallel_tool_calls)
        self.assertEqual(settings.reasoning.effort, "low")
        self.assertEqual(settings.retry.max_retries, 3)
        self.assertIs(settings.retry.policy, facticli_retry_policy)
        self.assertIsNone(settings.temperature)

        plain = build_model_settings(parallel_tool_calls=False)
        self.assertIsNone(plain.reasoning)
        self.assertIsNone(plain.retry)
        self.assertIsNone(build_retry_settings(0))


class RetryPolicyTests(unittest.TestCase):
    def _context(self, error: Exception, **normalized) -> RetryPolicyContext:
        return RetryPolicyContext(
            error=error,
            attempt=1,
            max_retries=3,
            stream=False,
            normalized=ModelRetryNormalizedError(**normalized),
        )

    def test_retries_transient_and_refuses_permanent_errors(self):
        rate_limited = self._context(RuntimeError("429"), status_code=429, retry_after=2.5)
        decision = facticli_retry_policy(rate_limited)
        self.assertTrue(decision.retry)
        self.assertEqual(decision.delay, 2.5)
        self.assertEqual(classify_retry_context(rate_limited), ErrorKind.RATE_LIMIT)

        self.assertTrue(facticli_retry_policy(self._context(RuntimeError("net"), is_network_error=True)).retry)
        self.assertTrue(facticli_retry_policy(self._context(RuntimeError("t"), is_timeout=True)).retry)
        self.assertTrue(facticli_retry_policy(self._context(RuntimeError("503"), status_code=503)).retry)

        self.assertFalse(facticli_retry_policy(self._context(RuntimeError("401"), status_code=401)).retry)
        self.assertFalse(facticli_retry_policy(self._context(RuntimeError("400"), status_code=400)).retry)
        overflow = self._context(RuntimeError("x"), status_code=400, message="maximum context length exceeded")
        decision = facticli_retry_policy(overflow)
        self.assertFalse(decision.retry)
        self.assertIn("context_overflow", decision.reason)

    def test_retries_draw_from_run_budget(self):
        log, token = activate_usage_log(retry_budget=1)
        try:
            context = self._context(RuntimeError("503"), status_code=503)
            self.assertTrue(facticli_retry_policy(context).retry)
            second = facticli_retry_policy(context)
            self.assertFalse(second.retry)
            self.assertIn("budget", second.reason)
            self.assertEqual(log.retries_used, 1)
        finally:
            deactivate_usage_log(token)


if __name__ == "__main__":
    unittest.main()
