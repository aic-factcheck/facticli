from __future__ import annotations

import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agents import MaxTurnsExceeded, ModelBehaviorError, set_tracing_disabled
from agents.testing import ModelStep, ScriptedModel, assistant_message, function_call
from agents.usage import Usage
from openai.types.responses import ResponseFunctionWebSearch
from openai.types.responses.response_function_web_search import ActionOpenPage, ActionSearch, ActionSearchSource

import facticli.brave_search as brave
from facticli.config import FactCheckRuntimeConfig
from facticli.core.artifacts import RunArtifacts
from facticli.core.constraints import ResearchConstraints, activate_constraints, deactivate_constraints
from facticli.core.contracts import (
    AspectFinding,
    ClaimExtractionResult,
    FactCheckReport,
    FindingStatus,
    InvestigationPlan,
    SourceGrounding,
    StopReason,
    VerificationCheck,
)
from facticli.core.errors import ErrorKind, SearchAuthError, classify_exception
from facticli.core.retrieval import RetrievalJournal, activate_journal, deactivate_journal, summarize_retrieval
from facticli.core.usage import activate_usage_log, deactivate_usage_log, summarize_usage
from facticli.llm import HOSTED_SEARCH_INCLUDE, StageRunner, build_search_tools, record_hosted_search_calls
from facticli.payloads import judge_payload
from facticli.pipeline import ClaimExtractor, FactChecker

# The real stage runners driven through ``Runner.run`` by a scripted model: what the
# harness does around a model call, without a network.

set_tracing_disabled(True)

CHECK = VerificationCheck(aspect_id="height_1", question="How tall?", rationale="r", search_queries=["q"], acceptance_criteria=["c"])
PER_CALL = Usage(requests=1, input_tokens=100, output_tokens=10, total_tokens=110)
FINDING = {
    "aspect_id": "height_1",
    "question": "How tall?",
    "signal": "supports",
    "summary": "It is 330 metres tall.",
    "confidence": 0.7,
    "sources": [
        {
            "title": "Operator",
            "url": "https://www.example.org/height/",
            "snippet": "The tower is 330 metres tall according to the operator.",
            "publisher": None,
            "published_at": "2026-01-02",
        },
        {
            "title": "Invented",
            "url": "https://invented.example.com/x",
            "snippet": "A passage that no tool call ever returned to the model.",
            "publisher": None,
            "published_at": None,
        },
    ],
    "caveats": [],
}
REPORT = {
    "claim": "c",
    "verdict": "Supported",
    "verdict_confidence": 0.8,
    "justification": "Backed by the operator [S1].",
    "key_points": [],
    "counter_argument": "none",
    "evidence_gaps": [],
    "findings": [],
    "sources": [],
}
SEARCH_TOOLING = dict(search_context_size="low", max_chars_per_field=1500, max_total_chars=12000, search_results_per_query=5)


async def _fake_brave(params, api_key):
    return [
        {
            "title": "Operator",
            "url": "https://example.org/height",
            "description": "The tower is 330 metres tall according to the operator.",
            "age": None,
            "page_age": "2026-01-02T00:00:00",
            "extra_snippets": [],
        }
    ]


def _search(index: int):
    return [
        function_call(
            "brave_web_search",
            {"query": f"query {index}", "count": None, "country": "us", "search_lang": "en"},
            call_id=f"call_{index}",
        )
    ]


def _final(payload: dict | str = FINDING):
    return [assistant_message(payload if isinstance(payload, str) else json.dumps(payload))]


def _stage_payload(model: ScriptedModel) -> dict:
    """The JSON payload a stage sent as its first user message."""
    return json.loads(model.calls[0].input[0]["content"])


def _runner(stage: str, output_type: type, model, *, max_turns: int = 2, tools=()) -> StageRunner:
    return StageRunner(name=stage, stage=stage, output_type=output_type, model=model, max_turns=max_turns, tools=tools)


def _checker(*, provider: str = "brave", researcher=None, planner=None, judge=None, retries: int = 0, search_cap: int | None = 4):
    config = FactCheckRuntimeConfig(
        search_provider=provider,
        tool_call_budget=8,
        research_timeout_seconds=0,
        research_retry_attempts=retries,
        max_parallel_research=2,
    )
    checker = FactChecker(config, planner=planner, researcher=researcher, judge=judge)
    checker.search_cap = search_cap
    return checker


def _researcher(steps, *, max_turns: int = 5, provider: str = "brave"):
    model = ScriptedModel(steps, default_usage=PER_CALL)
    runner = _runner("research", AspectFinding, model, max_turns=max_turns, tools=build_search_tools(search_provider=provider, **SEARCH_TOOLING))
    return model, _checker(provider=provider, researcher=runner)


def _stage(checker: FactChecker, *, retries: int = 0, search_cap: int | None = 4) -> FactChecker:
    checker.config = FactCheckRuntimeConfig(
        search_provider=checker.config.search_provider,
        tool_call_budget=8,
        research_timeout_seconds=0,
        research_retry_attempts=retries,
    )
    checker.search_cap = search_cap
    return checker


class _Harness(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        for item in (
            patch.dict("os.environ", {"BRAVE_SEARCH_API_KEY": "test-key"}, clear=False),
            patch.object(brave, "_request_brave", _fake_brave),
        ):
            item.start()
            self.addCleanup(item.stop)

    async def _research(self, adapter, *, search_cap: int | None = None):
        """Run one research call under a ledger and a journal; return (finding, log, journal, error)."""
        log, usage_token = activate_usage_log()
        journal = RetrievalJournal(search_cap=search_cap)
        journal_token = activate_journal(journal)
        try:
            return await adapter._research_once("claim", CHECK), log, journal, None
        except Exception as exc:
            return None, log, journal, exc
        finally:
            deactivate_journal(journal_token)
            deactivate_usage_log(usage_token)


class UsageAccountingTests(_Harness):
    async def test_a_failed_run_still_pays_into_the_ledger(self):
        _, adapter = _researcher([_search(1), _final("not json")])
        _, log, _, error = await self._research(adapter)
        self.assertIsInstance(error, ModelBehaviorError)
        (event,) = log.events
        self.assertEqual((event.completed, event.requests, event.total_tokens), (False, 2, 220))
        self.assertEqual(log.tokens_used, 220)  # the token budget sees it
        summary = summarize_usage(log.events)
        self.assertEqual((summary.failed_attempts, summary.failed_attempt_tokens), (1, 220))
        self.assertEqual(summary.per_stage["research"]["failed_attempt_tokens"], 220)

    async def test_a_run_cancelled_by_a_timeout_is_billed_for_what_it_spent(self):
        async def slow(_call):
            await asyncio.sleep(5)
            return _final()

        _, adapter = _researcher([_search(1), ModelStep.respond(slow)])
        log, usage_token = activate_usage_log()
        try:
            with self.assertRaises(asyncio.TimeoutError):
                await asyncio.wait_for(adapter._research_once("claim", CHECK), timeout=0.2)
        finally:
            deactivate_usage_log(usage_token)
        (event,) = log.events
        self.assertEqual((event.completed, event.total_tokens), (False, 110))  # the first response was already paid for
        self.assertGreater(event.duration_seconds, 0.1)

    async def test_cache_write_tokens_are_recorded(self):
        usage = Usage(
            requests=1,
            input_tokens=1000,
            output_tokens=20,
            total_tokens=1020,
            input_tokens_details={"cached_tokens": 300, "cache_write_tokens": 600},
        )
        judge = _runner("judge", FactCheckReport, ScriptedModel([_final(REPORT)], default_usage=usage), max_turns=3)
        log, usage_token = activate_usage_log()
        try:
            await judge.run(judge_payload(claim="c", plan=InvestigationPlan(claim="c"), findings=[]))
        finally:
            deactivate_usage_log(usage_token)
        summary = summarize_usage(log.events)
        self.assertEqual((summary.cached_input_tokens, summary.cache_write_tokens, summary.cache_hit_rate), (300, 600, 0.3))
        self.assertEqual((summary.failed_attempts, summary.wrap_up_calls), (0, 0))


class WrapUpTests(_Harness):
    async def test_turn_limit_becomes_a_marked_finding(self):
        model, adapter = _researcher([_search(1), _search(2), _final()], max_turns=2)
        finding, log, journal, error = await self._research(adapter)
        self.assertIsNone(error)
        self.assertEqual((finding.status, finding.stop_reason), (FindingStatus.COMPLETED, StopReason.TURN_LIMIT))

        wrap_up_call = model.calls[-1]
        self.assertEqual(wrap_up_call.model_settings.tool_choice, "none")
        self.assertIn("turn limit", wrap_up_call.input[-1]["content"])
        self.assertIn("Do not fill gaps from memory", wrap_up_call.input[-1]["content"])
        # The wrap-up call sees the evidence already retrieved and the same tool schemas, not a fresh prompt.
        self.assertTrue(any(item.get("type") == "function_call_output" for item in wrap_up_call.input if isinstance(item, dict)))
        self.assertEqual([tool.name for tool in wrap_up_call.tools], ["brave_web_search"])

        self.assertEqual([(event.wrap_up, event.completed) for event in log.events], [(False, True), (True, True)])
        summary = summarize_usage(log.events)
        self.assertEqual((summary.wrap_up_calls, summary.total_tokens), (1, 330))
        # The wrap-up ran inside the primary run; its time is not counted twice.
        self.assertAlmostEqual(summary.llm_seconds, log.events[0].duration_seconds)
        self.assertEqual(len(journal.records), 2)

    async def test_a_failed_wrap_up_surfaces_the_turn_limit_error(self):
        _, adapter = _researcher([_search(1), _search(2), _final("garbage")], max_turns=2)
        _, log, _, error = await self._research(adapter)
        self.assertIsInstance(error, MaxTurnsExceeded)
        self.assertEqual([(event.wrap_up, event.completed) for event in log.events], [(False, False), (True, False)])
        self.assertEqual(log.tokens_used, 330)

    async def test_search_cap_refusal_marks_the_finding(self):
        model, adapter = _researcher([_search(1), _search(2), _search(3), _final()], max_turns=6)
        finding, _, journal, error = await self._research(adapter, search_cap=2)
        self.assertIsNone(error)
        self.assertEqual(finding.stop_reason, StopReason.SEARCH_CAP)
        self.assertEqual([record.refused for record in journal.records], [False, False, True])
        self.assertEqual(json.loads(model.calls[-1].input[-1]["output"])["error"], "search_limit_reached")

    async def test_a_malformed_tool_call_is_answered_not_fatal(self):
        # Providers that do not enforce tool schemas can send arguments the tool cannot parse.
        malformed = [function_call("brave_web_search", {"query": "second", "count": "ten"}, call_id="call_bad")]
        model, adapter = _researcher([_search(1), malformed, _final()])
        finding, _, journal, error = await self._research(adapter)
        self.assertIsNone(error)
        self.assertEqual(finding.signal.value, "supports")
        self.assertIn("An error occurred while running the tool", model.calls[-1].input[-1]["output"])
        self.assertEqual(len(journal.records), 1)  # the evidence already retrieved is kept


class StageInputTests(_Harness):
    async def test_research_and_judge_payloads_lead_with_the_reference_date(self):
        planner_model = ScriptedModel([_final({"claim": "c", "checks": [], "assumptions": []})])
        research_model, researcher = _researcher([_final()])
        judge_model = ScriptedModel([_final(REPORT)])
        constraints_token = activate_constraints(ResearchConstraints(reference_date="2020-10-31"))
        try:
            checker = _checker(planner=_runner("plan", InvestigationPlan, planner_model), judge=_runner("judge", FactCheckReport, judge_model))
            artifacts = RunArtifacts(claim="c", normalized_claim="c")
            await checker._plan("c", artifacts, lambda _line: None)
            await self._research(researcher)
            await checker._judge("c", InvestigationPlan(claim="c"), [], artifacts, lambda _line: None)
        finally:
            deactivate_constraints(constraints_token)
        for model in (research_model, judge_model):
            self.assertEqual(next(iter(_stage_payload(model).items())), ("reference_date", "2020-10-31"))
        # The planner is not told: it turned the date into checks on titles held "as of" it.
        self.assertNotIn("reference_date", _stage_payload(planner_model))

    async def test_labels_the_judge_assigns_itself_are_dropped(self):
        claimed = json.loads(json.dumps(REPORT))
        claimed["sources"] = [{"title": "t", "url": "https://a.org/1", "snippet": "s", "grounding": "quote_matched", "tier": "primary"}]
        claimed["findings"] = [{**FINDING, "status": "failed", "stop_reason": "turn_limit"}]
        checker = _checker(judge=_runner("judge", FactCheckReport, ScriptedModel([_final(claimed)])))
        artifacts = RunArtifacts(claim="c", normalized_claim="c")
        report = await checker._judge("c", InvestigationPlan(claim="c"), [], artifacts, lambda _line: None)
        raw = artifacts.report_raw
        self.assertEqual((raw.sources[0].grounding, raw.sources[0].tier), (None, None))
        self.assertEqual((report.findings[0].status, report.findings[0].stop_reason), (FindingStatus.COMPLETED, None))
        # No researcher cited it, so the harness labels the judge-only source itself.
        self.assertEqual(report.sources[0].grounding, SourceGrounding.UNMATCHED)

    async def test_claim_extraction_sends_what_it_always_sent(self):
        # The hosted extraction API runs outside a fact-check run: no ledger, no journal, no constraints.
        result = {"input_text": "t", "detected_language": "en", "claims": [], "coverage_notes": [], "excluded_nonfactual": []}
        model = ScriptedModel([_final(result)])
        extractor = ClaimExtractor(_runner("extract_claims", ClaimExtractionResult, model, max_turns=4), max_claims=7)
        extracted = await extractor.extract_claims("Some text.")
        self.assertEqual(extracted.detected_language, "en")
        (call,) = model.calls
        payload = _stage_payload(model)
        self.assertEqual(list(payload), ["input_text", "requirements"])
        self.assertEqual((payload["input_text"], payload["requirements"]["max_claims"]), ("Some text.", 7))
        self.assertEqual(call.tools, [])
        self.assertIsNone(call.model_settings.response_include)

    async def test_hosted_search_asks_for_the_consulted_sources(self):
        hosted_model, hosted = _researcher([_final()], provider="openai")
        local_model, local = _researcher([_final()])
        await self._research(hosted)
        await self._research(local)
        self.assertEqual(hosted_model.calls[0].model_settings.response_include, [HOSTED_SEARCH_INCLUDE])
        self.assertIsNone(local_model.calls[0].model_settings.response_include)


class HostedSearchRunTests(_Harness):
    @staticmethod
    def _hosted_search(*actions):
        return [
            ResponseFunctionWebSearch(id=f"ws_{index}", status="completed", type="web_search_call", action=action)
            for index, action in enumerate(actions)
        ]

    async def test_consulted_urls_are_journaled_as_responses_arrive(self):
        searched = ActionSearch(
            type="search",
            query="eiffel height",
            sources=[ActionSearchSource(type="url", url="https://example.org/height?utm_source=openai")],
        )
        opened = ActionOpenPage(type="open_page", url="https://example.org/history")
        _, adapter = _researcher([[*self._hosted_search(searched, opened), *_final()]], provider="openai")
        finding, artifact = await ResearchStageTests._execute(self, _stage(adapter))
        # URL-level provenance only: hosted search returns no text to compare a quote with.
        self.assertEqual([source.grounding for source in finding.sources], [SourceGrounding.URL_MATCHED, SourceGrounding.UNMATCHED])
        self.assertEqual(
            [(record.provider, record.kind, record.urls) for record in artifact.retrieval],
            [("openai", "search", ["https://example.org/height?utm_source=openai"]), ("openai", "open_page", ["https://example.org/history"])],
        )

    async def test_a_failed_attempt_still_leaves_its_hosted_receipts(self):
        searched = ActionSearch(type="search", query="q", sources=[ActionSearchSource(type="url", url="https://example.org/height")])
        _, adapter = _researcher([[*self._hosted_search(searched), *_final("not json")]], provider="openai")
        finding, artifact = await ResearchStageTests._execute(self, _stage(adapter))
        self.assertEqual(finding.status, FindingStatus.FAILED)
        self.assertEqual([record.urls for record in artifact.retrieval], [["https://example.org/height"]])


class HostedSearchJournalTests(unittest.TestCase):
    @staticmethod
    def _response(*actions):
        items = [SimpleNamespace(type="web_search_call", status="completed", action=action) for action in actions]
        return SimpleNamespace(output=[*items, SimpleNamespace(type="message")])

    def test_reported_sources_are_url_provenance(self):
        journal = RetrievalJournal()
        journal.complete = True
        sources = [SimpleNamespace(url="https://a.org/1"), SimpleNamespace(url="https://b.org/2")]
        record_hosted_search_calls(
            self._response(
                SimpleNamespace(type="search", queries=["eiffel height", "tour eiffel"], query=None, sources=sources),
                SimpleNamespace(type="open_page", url="https://c.org/3"),
            ),
            journal,
        )
        self.assertTrue(journal.complete)
        self.assertEqual([(record.provider, record.query) for record in journal.records], [("openai", "eiffel height | tour eiffel"), ("openai", None)])
        self.assertEqual(journal.sightings("https://a.org/1?utm_source=openai"), [""])
        self.assertEqual(journal.sightings("https://c.org/3"), [""])
        summary = summarize_retrieval(journal.records)
        self.assertEqual((summary.searches, summary.pages_opened), (1, 1))

    def test_a_search_without_a_source_list_leaves_the_journal_incomplete(self):
        journal = RetrievalJournal()
        journal.complete = True
        record_hosted_search_calls(self._response(SimpleNamespace(type="search", queries=None, query="q", sources=None)), journal)
        self.assertFalse(journal.complete)
        self.assertEqual(journal.records[0].query, "q")


class ResearchStageTests(_Harness):
    async def _execute(self, checker: FactChecker):
        artifacts = RunArtifacts(claim="c", normalized_claim="c")
        finding = await checker.research_check("c", CHECK, artifacts)
        return finding, artifacts.research_checks[0]

    async def test_sources_are_grounded_and_receipts_kept(self):
        _, adapter = _researcher([_search(1), _final()])
        finding, artifact = await self._execute(_stage(adapter))
        self.assertEqual([source.grounding for source in finding.sources], [SourceGrounding.QUOTE_MATCHED, SourceGrounding.UNMATCHED])
        self.assertIsNotNone(finding.sources[0].tier)
        self.assertEqual(
            [record.model_dump(exclude_defaults=True) for record in artifact.retrieval],
            [{"provider": "brave", "query": "query 1", "urls": ["https://example.org/height"]}],  # no retrieved text
        )

    async def test_citations_without_any_search_are_unmatched_and_model_labels_are_dropped(self):
        claimed = json.loads(json.dumps(FINDING))
        claimed["sources"][0]["grounding"] = "quote_matched"  # a label only the harness may assign
        claimed["status"] = "failed"
        _, adapter = _researcher([_final(claimed)])  # answers straight away, no tool call
        finding, _ = await self._execute(_stage(adapter))
        self.assertEqual(finding.status, FindingStatus.COMPLETED)
        self.assertEqual([source.grounding for source in finding.sources], [SourceGrounding.UNMATCHED, SourceGrounding.UNMATCHED])

    async def test_each_attempt_gets_a_fresh_cap_and_all_receipts_are_kept(self):
        # The first attempt uses its whole search allowance and fails on output; the retry may search again.
        _, adapter = _researcher([_search(1), _final("not json"), _search(2), _final()])
        finding, artifact = await self._execute(_stage(adapter, retries=1, search_cap=1))
        self.assertEqual((finding.status, finding.stop_reason), (FindingStatus.COMPLETED, None))
        self.assertEqual((artifact.attempts, artifact.error_kinds), (2, ["schema"]))
        self.assertEqual([(record.query, record.refused) for record in artifact.retrieval], [("query 1", False), ("query 2", False)])

    async def test_a_rejected_search_key_aborts_the_run(self):
        async def rejected(params, api_key):
            raise SearchAuthError("Brave Search rejected the API key (HTTP 422).")

        _, adapter = _researcher([_search(1), _final()])
        artifacts = RunArtifacts(claim="c", normalized_claim="c")
        with patch.object(brave, "_request_brave", rejected):
            with self.assertRaises(Exception) as raised:
                await _stage(adapter, retries=2).research_check("c", CHECK, artifacts)
        self.assertEqual(classify_exception(raised.exception), ErrorKind.AUTH)
        self.assertEqual(artifacts.research_checks[0].attempts, 1)  # every retry would fail the same way


if __name__ == "__main__":
    unittest.main()
