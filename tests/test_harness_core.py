from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace

import httpx
import openai
from agents import AgentOutputSchema, MaxTurnsExceeded, ModelBehaviorError

from facticli.core.contracts import (
    AspectFinding,
    EvidenceSignal,
    FactCheckReport,
    FindingStatus,
    InvestigationPlan,
    ReviewDecision,
    SourceEvidence,
    SourceTier,
)
from facticli.core.errors import (
    BudgetExhaustedError,
    ErrorKind,
    classify_exception,
    classify_status_code,
    describe_exception,
    is_retryable,
)
from facticli.core.source_quality import assign_source_tiers, classify_source_tier, tier_counts
from facticli.core.truncation import UNTRUSTED_CONTENT_NOTICE, truncate_middle, truncate_search_results
from facticli.core.usage import (
    activate_usage_log,
    deactivate_usage_log,
    get_usage_log,
    record_stage_usage,
    summarize_usage,
)


class ContractSchemaTests(unittest.TestCase):
    def test_harness_assigned_fields_stay_out_of_model_schema(self):
        finding_props = AgentOutputSchema(AspectFinding).json_schema()["properties"]
        self.assertNotIn("status", finding_props)
        self.assertNotIn("failure_reason", finding_props)
        source_props = AgentOutputSchema(AspectFinding).json_schema()["$defs"]["SourceEvidence"]["properties"]
        for hidden in ("tier", "url_status", "archived_url"):
            self.assertNotIn(hidden, source_props)
        self.assertIn("acceptance_criteria", AgentOutputSchema(InvestigationPlan).json_schema()["$defs"]["VerificationCheck"]["properties"])
        report_props = AgentOutputSchema(FactCheckReport).json_schema()["properties"]
        self.assertIn("counter_argument", report_props)
        self.assertIn("evidence_gaps", report_props)
        self.assertIn("gaps", AgentOutputSchema(ReviewDecision).json_schema()["properties"])

    def test_defaults_keep_old_payloads_valid(self):
        finding = AspectFinding.model_validate(
            {"aspect_id": "a", "question": "q", "signal": "supports", "summary": "s", "confidence": 0.5}
        )
        self.assertEqual(finding.status, FindingStatus.COMPLETED)
        self.assertTrue(finding.is_observation)
        source = SourceEvidence.model_validate({"title": "t", "url": "https://x.org", "snippet": "s"})
        self.assertIsNone(source.tier)
        report = FactCheckReport.model_validate(
            {"claim": "c", "verdict": "Supported", "verdict_confidence": 0.7, "justification": "j"}
        )
        self.assertEqual(report.counter_argument, "")


class SourceTierTests(unittest.TestCase):
    def test_tier_table(self):
        cases = {
            "https://www.czso.cz/csu/czso/home": SourceTier.PRIMARY,
            "https://data.gov.uk/dataset/x": SourceTier.PRIMARY,
            "https://ec.europa.eu/eurostat/web": SourceTier.PRIMARY,
            "https://www.nature.com/articles/1": SourceTier.PRIMARY,
            "https://something.gov.au/report": SourceTier.PRIMARY,
            "https://en.wikipedia.org/wiki/Prague": SourceTier.REFERENCE,
            "https://web.archive.org/web/2020/https://x": SourceTier.REFERENCE,
            "https://www.reuters.com/world/europe/story": SourceTier.NEWS,
            "https://irozhlas.cz/zpravy": SourceTier.NEWS,
            "https://www.reuters.com/fact-check/story": SourceTier.FACT_CHECKER,
            "https://www.politifact.com/factchecks/2020/": SourceTier.FACT_CHECKER,
            "https://demagog.cz/vyrok/1": SourceTier.FACT_CHECKER,
            "https://x.com/user/status/1": SourceTier.USER_GENERATED,
            "https://someone.substack.com/p/post": SourceTier.USER_GENERATED,
            "https://www.reddit.com/r/x/comments/1": SourceTier.USER_GENERATED,
            "https://random-blog.example/post": SourceTier.OTHER,
            "not a url": SourceTier.OTHER,
        }
        for url, expected in cases.items():
            self.assertEqual(classify_source_tier(url), expected, url)

    def test_overrides_and_assignment(self):
        self.assertEqual(
            classify_source_tier("https://mypaper.example/a", {"mypaper.example": SourceTier.NEWS}),
            SourceTier.NEWS,
        )
        sources = [
            SourceEvidence(title="a", url="https://www.bbc.com/news/1", snippet="s"),
            SourceEvidence(title="b", url="https://x.org", snippet="s", tier=SourceTier.PRIMARY),
        ]
        tiered = assign_source_tiers(sources)
        self.assertEqual([s.tier for s in tiered], [SourceTier.NEWS, SourceTier.PRIMARY])
        self.assertEqual(tier_counts(tiered), {"news": 1, "primary": 1})
        # Originals are untouched.
        self.assertIsNone(sources[0].tier)


class ErrorTaxonomyTests(unittest.TestCase):
    def _response(self, status: int) -> httpx.Response:
        return httpx.Response(status, request=httpx.Request("GET", "https://api.example"))

    def test_classification(self):
        req = httpx.Request("GET", "https://api.example")
        self.assertEqual(classify_exception(openai.RateLimitError("rl", response=self._response(429), body=None)), ErrorKind.RATE_LIMIT)
        self.assertEqual(
            classify_exception(
                openai.BadRequestError("maximum context length is 128000 tokens", response=self._response(400), body=None)
            ),
            ErrorKind.CONTEXT_OVERFLOW,
        )
        self.assertEqual(classify_exception(openai.BadRequestError("bad", response=self._response(400), body=None)), ErrorKind.BAD_REQUEST)
        self.assertEqual(classify_exception(openai.AuthenticationError("no", response=self._response(401), body=None)), ErrorKind.AUTH)
        self.assertEqual(classify_exception(openai.InternalServerError("boom", response=self._response(503), body=None)), ErrorKind.SERVER)
        self.assertEqual(classify_exception(openai.APITimeoutError(request=req)), ErrorKind.TIMEOUT)
        self.assertEqual(classify_exception(openai.APIConnectionError(request=req)), ErrorKind.NETWORK)
        self.assertEqual(classify_exception(asyncio.TimeoutError()), ErrorKind.TIMEOUT)
        self.assertEqual(classify_exception(ModelBehaviorError("bad json")), ErrorKind.SCHEMA)
        self.assertEqual(classify_exception(MaxTurnsExceeded("x")), ErrorKind.MAX_TURNS)
        self.assertEqual(classify_exception(BudgetExhaustedError("x")), ErrorKind.BUDGET)
        self.assertEqual(classify_exception(RuntimeError("weird")), ErrorKind.UNKNOWN)
        self.assertEqual(classify_exception(SimpleNamespace), ErrorKind.UNKNOWN)

    def test_status_codes_and_retryability(self):
        self.assertEqual(classify_status_code(408), ErrorKind.TIMEOUT)
        self.assertEqual(classify_status_code(502), ErrorKind.SERVER)
        self.assertEqual(classify_status_code(404), ErrorKind.NOT_FOUND)
        self.assertEqual(classify_status_code(413, "request too large"), ErrorKind.CONTEXT_OVERFLOW)
        self.assertTrue(is_retryable(ErrorKind.RATE_LIMIT))
        self.assertTrue(is_retryable(ErrorKind.SERVER))
        self.assertFalse(is_retryable(ErrorKind.AUTH))
        self.assertFalse(is_retryable(ErrorKind.CONTEXT_OVERFLOW))
        self.assertFalse(is_retryable(ErrorKind.SCHEMA))
        self.assertTrue(describe_exception(RuntimeError("x")).startswith("unknown: RuntimeError: x"))


class UsageLedgerTests(unittest.TestCase):
    def test_budget_ledger_and_cached_tokens(self):
        log, token = activate_usage_log(token_budget=1000, retry_budget=2)
        try:
            self.assertIs(get_usage_log(), log)
            record_stage_usage(
                stage="plan",
                model="m",
                usage=SimpleNamespace(
                    requests=1,
                    input_tokens=600,
                    output_tokens=100,
                    total_tokens=700,
                    input_tokens_details=SimpleNamespace(cached_tokens=300),
                    output_tokens_details=SimpleNamespace(reasoning_tokens=40),
                ),
                duration_seconds=1.0,
            )
            self.assertEqual(log.tokens_remaining(), 300)
            self.assertFalse(log.token_budget_exhausted())
            record_stage_usage(
                stage="research",
                model="m",
                usage=SimpleNamespace(requests=1, input_tokens=300, output_tokens=100, total_tokens=400),
                duration_seconds=1.0,
            )
            self.assertTrue(log.token_budget_exhausted())
            self.assertEqual(log.tokens_remaining(), 0)
            self.assertTrue(log.try_consume_retry())
            self.assertTrue(log.try_consume_retry())
            self.assertFalse(log.try_consume_retry())
            log.record_skipped_stage("review_round_1")
        finally:
            deactivate_usage_log(token)
        self.assertIsNone(get_usage_log())

        summary = summarize_usage(log.events)
        self.assertEqual(summary.cached_input_tokens, 300)
        self.assertEqual(summary.reasoning_tokens, 40)
        self.assertAlmostEqual(summary.cache_hit_rate, 300 / 900, places=4)
        status = log.budget_status()
        self.assertTrue(status.token_budget_exhausted)
        self.assertTrue(status.retry_budget_exhausted)
        self.assertEqual(status.retries_used, 2)
        self.assertEqual(status.skipped_stages, ["review_round_1"])

    def test_unbudgeted_log_never_exhausts(self):
        log, token = activate_usage_log()
        try:
            self.assertIsNone(log.tokens_remaining())
            self.assertFalse(log.token_budget_exhausted())
            for _ in range(50):
                self.assertTrue(log.try_consume_retry())
        finally:
            deactivate_usage_log(token)


class TruncationTests(unittest.TestCase):
    def test_truncate_middle_keeps_head_and_tail(self):
        text = "H" * 50 + "M" * 100 + "T" * 50
        out = truncate_middle(text, 80)
        self.assertLessEqual(len(out), 80)
        self.assertTrue(out.startswith("HHHH"))
        self.assertTrue(out.endswith("TTTT"))
        self.assertIn("chars omitted", out)
        self.assertEqual(truncate_middle("short", 80), "short")

    def test_truncate_search_results_bounds_fields_and_count(self):
        results = [
            {"title": f"t{i}", "url": f"https://x/{i}", "description": "d" * 500, "extra_snippets": ["e" * 500]}
            for i in range(10)
        ]
        bounded, header = truncate_search_results(
            results, max_results=10, max_chars_per_field=100, max_total_chars=800
        )
        self.assertLess(len(bounded), 10)
        self.assertTrue(all(len(item["description"]) <= 100 for item in bounded))
        self.assertEqual(header["truncation"]["original_result_count"], 10)
        self.assertGreater(header["truncation"]["truncated_text_fields"], 0)

        untouched, header = truncate_search_results(
            results[:2], max_results=5, max_chars_per_field=1000, max_total_chars=100000
        )
        self.assertEqual(len(untouched), 2)
        self.assertEqual(header, {})
        self.assertIn("never", UNTRUSTED_CONTENT_NOTICE)


if __name__ == "__main__":
    unittest.main()
