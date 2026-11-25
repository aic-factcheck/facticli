from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace

import httpx
import openai
from agents import AgentOutputSchema, MaxTurnsExceeded, ModelBehaviorError

from facticli.core.contracts import (
    AspectFinding,
    FactCheckReport,
    FindingStatus,
    InvestigationPlan,
    SourceEvidence,
    SourceTier,
)
from facticli.core.errors import (
    FATAL_KINDS,
    ErrorKind,
    SearchAuthError,
    classify_exception,
    classify_status_code,
    describe_exception,
    is_fatal,
)
from facticli.core.source_quality import assign_source_tiers, classify_source_tier
from facticli.core.truncation import UNTRUSTED_CONTENT_NOTICE, truncate_middle, truncate_search_results
from facticli.core.usage import (
    activate_usage_log,
    apply_usage,
    deactivate_usage_log,
    get_usage_log,
    summarize_usage,
    track_usage,
)


class ContractSchemaTests(unittest.TestCase):
    def test_harness_assigned_fields_stay_out_of_model_schema(self):
        finding_props = AgentOutputSchema(AspectFinding).json_schema()["properties"]
        self.assertNotIn("status", finding_props)
        self.assertNotIn("failure_reason", finding_props)
        self.assertNotIn("stop_reason", finding_props)
        source_props = AgentOutputSchema(AspectFinding).json_schema()["$defs"]["SourceEvidence"]["properties"]
        for hidden in ("tier", "grounding"):
            self.assertNotIn(hidden, source_props)
        self.assertIn("acceptance_criteria", AgentOutputSchema(InvestigationPlan).json_schema()["$defs"]["VerificationCheck"]["properties"])
        report_props = AgentOutputSchema(FactCheckReport).json_schema()["properties"]
        self.assertIn("counter_argument", report_props)
        self.assertIn("evidence_gaps", report_props)

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
            # The most specific listed domain wins over a parent entry or a suffix rule.
            "https://dictionary.cambridge.org/dictionary/english/x": SourceTier.REFERENCE,
            "https://www.cambridge.org/core/journals/x": SourceTier.PRIMARY,
            "https://www.loc.gov/item/x": SourceTier.REFERENCE,
            "https://web.archive.org/web/2020/https://www.politifact.com/factchecks/x": SourceTier.FACT_CHECKER,
        }
        for url, expected in cases.items():
            self.assertEqual(classify_source_tier(url), expected, url)

    def test_assignment(self):
        sources = [
            SourceEvidence(title="a", url="https://www.bbc.com/news/1", snippet="s"),
            SourceEvidence(title="b", url="https://x.org", snippet="s", tier=SourceTier.PRIMARY),
        ]
        tiered = assign_source_tiers(sources)
        self.assertEqual([s.tier for s in tiered], [SourceTier.NEWS, SourceTier.PRIMARY])
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
        self.assertEqual(classify_exception(RuntimeError("weird")), ErrorKind.UNKNOWN)
        self.assertEqual(classify_exception(SimpleNamespace), ErrorKind.UNKNOWN)

    def test_fatal_kinds_are_told_apart_from_their_transient_lookalikes(self):
        from agents import ModelRefusalError, UserError

        wording = "You exceeded your current quota, please check your plan and billing details."
        exhausted = openai.RateLimitError(wording, response=self._response(429), body={"code": "insufficient_quota"})
        # Per-minute throttling is worded the same way by some providers; only the structured code is trusted.
        throttled = openai.RateLimitError(wording, response=self._response(429), body=None)
        self.assertEqual(classify_exception(exhausted), ErrorKind.QUOTA)
        self.assertEqual(classify_exception(throttled), ErrorKind.RATE_LIMIT)
        self.assertEqual(classify_status_code(429, '{"code": "insufficient_quota"}'), ErrorKind.QUOTA)
        self.assertEqual(classify_status_code(429, "Quota exceeded for requests per minute"), ErrorKind.RATE_LIMIT)
        self.assertEqual(classify_status_code(402), ErrorKind.QUOTA)

        # A 403 can be about one request (moderation, model access); only a 401 condemns the key.
        denied = openai.PermissionDeniedError("input was flagged", response=self._response(403), body=None)
        self.assertEqual(classify_exception(denied), ErrorKind.FORBIDDEN)
        self.assertEqual((classify_status_code(403), classify_status_code(401)), (ErrorKind.FORBIDDEN, ErrorKind.AUTH))
        self.assertEqual(classify_exception(ModelRefusalError("I can't help with that")), ErrorKind.REFUSAL)

        self.assertEqual(FATAL_KINDS, {ErrorKind.AUTH, ErrorKind.QUOTA, ErrorKind.NOT_FOUND})
        for kind in (ErrorKind.RATE_LIMIT, ErrorKind.FORBIDDEN, ErrorKind.REFUSAL, ErrorKind.BAD_REQUEST):
            self.assertFalse(is_fatal(kind), kind)

        # The SDK wraps what a tool raised; the tool's own error decides the kind.
        wrapped = UserError("Error running tool brave_web_search: rejected")
        wrapped.__cause__ = SearchAuthError("rejected")
        self.assertEqual(classify_exception(wrapped), ErrorKind.AUTH)
        self.assertEqual(classify_exception(UserError("misconfigured agent")), ErrorKind.UNKNOWN)

    def test_status_codes(self):
        self.assertEqual(classify_status_code(408), ErrorKind.TIMEOUT)
        self.assertEqual(classify_status_code(502), ErrorKind.SERVER)
        self.assertEqual(classify_status_code(404), ErrorKind.NOT_FOUND)
        self.assertEqual(classify_status_code(413, "request too large"), ErrorKind.CONTEXT_OVERFLOW)
        self.assertTrue(describe_exception(RuntimeError("x")).startswith("unknown: RuntimeError: x"))


class UsageLedgerTests(unittest.TestCase):
    def test_ledger_counts_failed_runs_and_cached_tokens(self):
        log, token = activate_usage_log()
        try:
            self.assertIs(get_usage_log(), log)
            with track_usage(stage="plan", model="m") as event:
                apply_usage(
                    event,
                    SimpleNamespace(
                        requests=1,
                        input_tokens=600,
                        output_tokens=100,
                        total_tokens=700,
                        input_tokens_details=SimpleNamespace(cached_tokens=300, cache_write_tokens=200),
                        output_tokens_details=SimpleNamespace(reasoning_tokens=40),
                    ),
                )
            self.assertEqual(log.tokens_used, 700)
            with self.assertRaises(RuntimeError), track_usage(stage="research", model="m") as event:
                apply_usage(event, SimpleNamespace(requests=1, input_tokens=300, output_tokens=100, total_tokens=400))
                raise RuntimeError("the run failed after its first response")
            # Tokens spent by a failed run count like any others.
            self.assertEqual(log.tokens_used, 1100)
        finally:
            deactivate_usage_log(token)
        self.assertIsNone(get_usage_log())

        summary = summarize_usage(log.events)
        self.assertEqual(summary.cached_input_tokens, 300)
        self.assertEqual(summary.reasoning_tokens, 40)
        self.assertEqual(summary.cache_write_tokens, 200)
        self.assertEqual((summary.failed_attempts, summary.failed_attempt_tokens), (1, 400))
        self.assertEqual(summary.per_stage["research"]["failed_attempts"], 1)
        self.assertEqual(summary.per_stage["plan"]["failed_attempts"], 0)
        self.assertAlmostEqual(summary.cache_hit_rate, 300 / 900, places=4)


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
