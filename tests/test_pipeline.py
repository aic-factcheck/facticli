from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import openai

from facticli.config import ClaimExtractionRuntimeConfig, FactCheckRuntimeConfig
from facticli.core.artifacts import ResearchCheckArtifact, RunArtifacts
from facticli.core.constraints import (
    ResearchConstraints,
    activate_constraints,
    deactivate_constraints,
    is_blocked_url,
    normalize_claim_date,
    parse_date_loose,
    unwrap_archive_url,
    url_domain,
    violates_date_cutoff,
)
from facticli.core.contracts import (
    AspectFinding,
    CheckworthyClaim,
    ClaimExtractionResult,
    EvidenceSignal,
    FactCheckReport,
    FindingStatus,
    InvestigationPlan,
    SourceEvidence,
    SourceGrounding,
    SourceTier,
    VeracityVerdict,
    VerificationCheck,
)
from facticli.core.retrieval import RetrievalRecord, get_journal
from facticli.core.usage import activate_usage_log, apply_usage, deactivate_usage_log, summarize_usage, track_usage
from facticli.llm import Endpoint
from facticli.payloads import index_sources
from facticli.pipeline import (
    ClaimExtractor,
    FactChecker,
    apply_source_constraints,
    build_claim_extractor,
    build_fact_checker,
    merge_sources,
    save_artifacts,
)


def _check(aspect_id: str) -> VerificationCheck:
    return VerificationCheck(aspect_id=aspect_id, question=f"Q {aspect_id}", rationale="r", search_queries=["q"])


def _plan(*aspect_ids: str) -> InvestigationPlan:
    return InvestigationPlan(claim="claim", checks=[_check(aspect_id) for aspect_id in aspect_ids])


def _finding(aspect_id: str, urls: list[str] | None = None, signal: EvidenceSignal = EvidenceSignal.SUPPORTS) -> AspectFinding:
    return AspectFinding(
        aspect_id=aspect_id,
        question=f"Q {aspect_id}",
        signal=signal,
        summary=f"summary {aspect_id}",
        confidence=0.8,
        sources=[SourceEvidence(title=url, url=url, snippet="snip") for url in (urls or [])],
    )


def _report(**overrides) -> FactCheckReport:
    fields = dict(claim="x", verdict=VeracityVerdict.SUPPORTED, verdict_confidence=0.8, justification="j [S1]", counter_argument="c")
    return FactCheckReport(**{**fields, **overrides})


class _Stage:
    """Fake model stage: ``respond(payload)`` returns the output or raises."""

    def __init__(self, respond) -> None:
        self.respond = respond
        self.payloads: list[dict] = []

    async def run(self, payload: dict):
        self.payloads.append(payload)
        result = self.respond(payload)
        if asyncio.iscoroutine(result):
            result = await result
        return result, False


class _Researcher(_Stage):
    def __init__(self, *, fail: set[str] | None = None, error: Exception | None = None) -> None:
        self.fail = fail or set()
        self.error = error or RuntimeError("boom")
        self.calls: list[str] = []
        super().__init__(self._respond)

    def _respond(self, payload: dict) -> AspectFinding:
        aspect_id = payload["check"]["aspect_id"]
        self.calls.append(aspect_id)
        if aspect_id in self.fail:
            raise self.error
        return _finding(aspect_id, ["https://www.reuters.com/world/x", "https://x.com/u/1"])


def _checker(*, planner=None, researcher=None, judge=None, retries: int = 1, parallel: int = 2, **config) -> FactChecker:
    return FactChecker(
        FactCheckRuntimeConfig(
            max_parallel_research=parallel, research_timeout_seconds=0, research_retry_attempts=retries, **config
        ),
        planner=planner or _Stage(lambda payload: _plan("a")),
        researcher=researcher or _Researcher(),
        judge=judge or _Stage(lambda payload: _report()),
    )


async def _research(checker: FactChecker, plan: InvestigationPlan, artifacts: RunArtifacts, lines: list[str] | None = None):
    say = lines.append if lines is not None else (lambda _line: None)
    return await checker._research("c", plan, artifacts, say)


class PipelineRunTests(unittest.IsolatedAsyncioTestCase):
    async def test_full_run_plans_researches_judges_and_reports_progress(self):
        planner = _Stage(
            lambda payload: InvestigationPlan(
                claim="ignored",
                checks=[
                    VerificationCheck(
                        aspect_id="Timeline 1",
                        question="  Was it completed in 1889? ",
                        rationale="  date  ",
                        search_queries=["", "Eiffel Tower 1889", "eiffel tower 1889"],
                    ),
                    VerificationCheck(aspect_id="timeline_1", question="For the World's Fair?", rationale="e"),
                ],
            )
        )
        researcher = _Researcher()
        lines: list[str] = []
        checker = _checker(planner=planner, researcher=researcher, max_checks=3)
        run = await checker.check_claim("  Eiffel claim  ", progress=lines.append, claim_date="25-8-2020")

        self.assertEqual([check.aspect_id for check in run.plan.checks], ["timeline_1", "timeline_1_2"])
        self.assertEqual(run.plan.checks[0].question, "Was it completed in 1889?")
        self.assertIn("Eiffel claim", run.plan.checks[0].search_queries)
        self.assertEqual(planner.payloads[0]["requirements"]["max_checks"], 3)
        self.assertNotIn("reference_date", planner.payloads[0])
        self.assertEqual(researcher.payloads[0]["reference_date"], "2020-08-25")
        self.assertEqual(run.claim, "Eiffel claim")
        self.assertEqual(run.report.claim, "Eiffel claim")
        self.assertEqual(lines[0], "[progress] Starting fact-check: Eiffel claim")
        self.assertIn("[progress] Plan ready with 2 check(s):", lines)
        self.assertEqual(lines[-1], "[progress] Fact-check run completed.")
        self.assertIsNotNone(run.artifacts.usage_summary)
        self.assertIsNone(get_journal())  # the journal does not outlive the run

    async def test_an_empty_plan_falls_back_to_one_direct_check(self):
        checker = _checker(planner=_Stage(lambda payload: InvestigationPlan(claim="c", checks=[])))
        run = await checker.check_claim("The claim.")
        self.assertEqual([check.aspect_id for check in run.plan.checks], ["claim_direct_check"])

    async def test_the_claim_is_evaluated_as_of_its_date_or_else_today(self):
        checker = _checker()
        dated = (await checker.check_claim("The claim.", claim_date="25-8-2020")).artifacts.constraints
        self.assertEqual((dated.claim_date, dated.reference_date), ("2020-08-25", "2020-08-25"))
        undated = (await checker.check_claim("The claim.", claim_date="  ")).artifacts.constraints
        self.assertIsNone(undated.claim_date)
        self.assertRegex(undated.reference_date, r"^\d{4}-\d{2}-\d{2}$")

    async def test_unparseable_claim_date_stops_the_run_instead_of_dropping_the_cutoff(self):
        checker = _checker()
        for bad in ("8-25-2020", "25.8.2020", "last week"):
            with self.assertRaisesRegex(ValueError, "Unparseable claim date"):
                await checker.check_claim("The claim.", claim_date=bad)

    async def test_blocked_domains_reach_the_constraints_and_artifacts_are_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            checker = _checker(block_fact_checkers=True, blocked_domains=("Example.org", "politifact.com"), stage_efforts={"judge": "high"})
            checker.artifacts_dir = tmp
            run = await checker.check_claim("The claim.", claim_id=7)
            files = list(Path(tmp).glob("run_*_7_*.json"))
            self.assertEqual(len(files), 1)
        blocked = run.artifacts.constraints.blocked_domains
        self.assertIn("example.org", blocked)
        self.assertEqual(blocked.count("politifact.com"), 1)
        self.assertEqual(run.artifacts.stage_efforts, {"judge": "high"})


class ResearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_sources_get_tiers_and_failures_carry_status(self):
        researcher = _Researcher(fail={"b"})
        artifacts = RunArtifacts(claim="c", normalized_claim="c")
        lines: list[str] = []
        findings = await _research(_checker(researcher=researcher), _plan("a", "b"), artifacts, lines)

        good, bad = findings
        self.assertEqual([s.tier for s in good.sources], [SourceTier.NEWS, SourceTier.USER_GENERATED])
        self.assertEqual((bad.status, bad.signal, bad.is_observation), (FindingStatus.FAILED, EvidenceSignal.INSUFFICIENT, False))
        self.assertTrue(bad.failure_reason.startswith("unknown: RuntimeError"))
        self.assertIn("[progress] [b] failed: unknown: RuntimeError: boom", lines)
        self.assertEqual(researcher.calls.count("b"), 2)
        bad_artifact = artifacts.research_checks[1]
        self.assertEqual((bad_artifact.attempts, bad_artifact.error_kinds), (2, ["unknown", "unknown"]))

    async def test_non_retryable_errors_stop_retrying(self):
        response = httpx.Response(400, request=httpx.Request("GET", "https://api"))
        researcher = _Researcher(fail={"a"}, error=openai.BadRequestError("unsupported value", response=response, body=None))
        artifacts = RunArtifacts(claim="c", normalized_claim="c")
        findings = await _research(_checker(researcher=researcher, retries=3, parallel=1), _plan("a"), artifacts)
        self.assertTrue(findings[0].failure_reason.startswith("bad_request: BadRequestError"))
        self.assertEqual(researcher.calls, ["a"])
        self.assertEqual(artifacts.research_checks[0].error_kinds, ["bad_request"])

    async def test_a_fatal_error_aborts_the_run_and_cancels_sibling_checks(self):
        cancelled: list[str] = []

        async def dead_key(payload: dict):
            aspect_id = payload["check"]["aspect_id"]
            if aspect_id == "a":
                response = httpx.Response(401, request=httpx.Request("GET", "https://api"))
                raise openai.AuthenticationError("denied", response=response, body=None)
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                cancelled.append(aspect_id)
                raise
            return _finding(aspect_id)

        artifacts = RunArtifacts(claim="c", normalized_claim="c")
        checker = _checker(researcher=_Stage(dead_key), retries=3, parallel=3)
        with self.assertRaises(openai.AuthenticationError):
            await _research(checker, _plan("a", "b", "c"), artifacts)
        # Every later request would fail the same way: no retry, and nobody keeps spending.
        self.assertEqual(artifacts.research_checks[0].attempts, 1)
        self.assertEqual(sorted(cancelled), ["b", "c"])

    async def test_a_slow_check_times_out_into_a_failed_finding(self):
        async def slow(payload: dict):
            await asyncio.sleep(5)

        checker = _checker(researcher=_Stage(slow), retries=0)
        checker.config = FactCheckRuntimeConfig(research_timeout_seconds=0.05, research_retry_attempts=0)
        artifacts = RunArtifacts(claim="c", normalized_claim="c")
        (finding,) = await _research(checker, _plan("a"), artifacts)
        self.assertEqual(finding.status, FindingStatus.FAILED)
        self.assertEqual(artifacts.research_checks[0].error_kinds, ["timeout"])

    async def test_a_researcher_that_does_not_journal_leaves_sources_unassessed(self):
        artifacts = RunArtifacts(claim="c", normalized_claim="c")
        findings = await _research(_checker(retries=0), _plan("a"), artifacts)
        self.assertTrue(all(source.grounding is None for source in findings[0].sources))
        self.assertEqual(artifacts.research_checks[0].retrieval, [])

    async def test_journaled_results_ground_the_citations_under_the_search_cap(self):
        quote = "The council approved the budget on Monday evening."
        caps: list[int | None] = []

        def journaling(payload: dict) -> AspectFinding:
            journal = get_journal()
            journal.complete = True
            caps.append(journal.search_cap)
            journal.add(RetrievalRecord(provider="brave", query="q", urls=["https://www.bbc.com/news/1"]), [quote])
            finding = _finding(payload["check"]["aspect_id"])
            finding.sources = [
                SourceEvidence(title="bbc", url="https://www.bbc.com/news/1", snippet=quote),
                SourceEvidence(title="made up", url="https://example.com/nope", snippet=quote),
            ]
            return finding

        checker = _checker(researcher=_Stage(journaling), tool_call_budget=5)
        artifacts = RunArtifacts(claim="c", normalized_claim="c")
        (finding,) = await _research(checker, _plan("a"), artifacts)
        self.assertEqual(caps, [10])  # twice the budget the researcher is told about
        self.assertEqual([s.grounding for s in finding.sources], [SourceGrounding.QUOTE_MATCHED, SourceGrounding.UNMATCHED])
        self.assertEqual(len(artifacts.research_checks[0].retrieval), 1)

    async def test_flex_tier_lengthens_the_research_timeout(self):
        self.assertEqual(FactCheckRuntimeConfig(service_tier="flex").effective_research_timeout_seconds, 600.0)
        self.assertEqual(FactCheckRuntimeConfig().effective_research_timeout_seconds, 120.0)


class JudgeTests(unittest.IsolatedAsyncioTestCase):
    async def _judge(self, report: FactCheckReport, findings: list[AspectFinding], plan: InvestigationPlan):
        artifacts = RunArtifacts(claim="c", normalized_claim="c")
        judged = await _checker(judge=_Stage(lambda payload: report))._judge("c", plan, findings, artifacts, lambda _line: None)
        return judged, artifacts

    async def test_harness_findings_are_authoritative_and_sources_tiered(self):
        report = _report(
            key_points=[" a ", "a", ""],
            counter_argument="Could be refuted if ...",
            evidence_gaps=["gap"],
            # The model echoes findings without harness-only fields; they must not overwrite ours.
            findings=[_finding("a"), _finding("b")],
            sources=[SourceEvidence(title="extra", url="https://www.czso.cz/x", snippet="s")],
        )
        findings = [
            _finding("a", ["https://www.reuters.com/world/x"]),
            AspectFinding(
                aspect_id="b", question="Q b", signal=EvidenceSignal.INSUFFICIENT, summary="failed", confidence=0.0,
                status=FindingStatus.FAILED, failure_reason="timeout",
            ),
        ]
        findings[0].sources[0].tier = SourceTier.NEWS
        judged, artifacts = await self._judge(report, findings, _plan("a", "b"))

        self.assertEqual(judged.findings[1].status, FindingStatus.FAILED)
        self.assertEqual(judged.key_points, ["a"])
        self.assertEqual(judged.claim, "c")
        self.assertEqual([s.url for s in judged.sources], ["https://www.reuters.com/world/x", "https://www.czso.cz/x"])
        self.assertEqual([s.tier for s in judged.sources], [SourceTier.NEWS, SourceTier.PRIMARY])
        self.assertIsNotNone(artifacts.report_raw)
        self.assertIs(artifacts.report_final, judged)

    async def test_cited_source_ids_must_exist_in_the_source_table(self):
        report = _report(justification="Confirmed [S1][S2]. Also [S5].", key_points=["Point [S2]"], counter_argument="None [S9].")
        findings = [_finding("a", ["https://www.reuters.com/x", "https://www.bbc.com/y"]), _finding("b", ["https://www.bbc.com/y"])]
        _, artifacts = await self._judge(report, findings, _plan("a", "b"))
        self.assertEqual(artifacts.unresolved_citations, [5, 9])  # the table has two rows

    async def test_a_source_only_the_judge_lists_is_labelled_untraceable(self):
        report = _report(
            justification="Confirmed [S1][S2].",
            sources=[
                SourceEvidence(title="copied", url="https://www.reuters.com/x", snippet="s"),
                SourceEvidence(title="miscopied", url="https://www.reuters.com/y", snippet="s"),
            ],
        )
        traced = _finding("a", ["https://www.reuters.com/x"])
        traced.sources[0].grounding = SourceGrounding.QUOTE_MATCHED
        judged, artifacts = await self._judge(report, [traced], _plan("a"))
        self.assertEqual(
            [(source.url, source.grounding) for source in judged.sources],
            [("https://www.reuters.com/x", SourceGrounding.QUOTE_MATCHED), ("https://www.reuters.com/y", SourceGrounding.UNMATCHED)],
        )
        # The reader sees it as S2 with its label, so citing it is not a dangling id.
        self.assertEqual(artifacts.unresolved_citations, [])

    def test_a_url_retrieved_by_any_check_counts_as_retrieved(self):
        url = "https://en.wikipedia.org/wiki/X"
        from_memory = _finding("a", [url])
        from_memory.sources[0].grounding = SourceGrounding.UNMATCHED
        retrieved = _finding("b", [url])
        retrieved.sources[0].grounding = SourceGrounding.QUOTE_MATCHED
        for order in ([from_memory, retrieved], [retrieved, from_memory]):
            self.assertEqual([source.grounding for source in merge_sources([], order)], [SourceGrounding.QUOTE_MATCHED])
            self.assertNotIn("grounding", index_sources(order)[0][0])
        self.assertEqual(index_sources([from_memory, from_memory])[0][0]["grounding"], "unmatched")

    def test_merge_sources_prefers_tiered_copy(self):
        tiered = SourceEvidence(title="t", url="https://a.org/x", snippet="s", tier=SourceTier.OTHER)
        untiered = SourceEvidence(title="t", url="https://a.org/x/", snippet="s")
        self.assertEqual(len(merge_sources([untiered], [_finding("a")])), 1)
        finding = AspectFinding(aspect_id="a", question="q", signal=EvidenceSignal.SUPPORTS, summary="s", confidence=0.5, sources=[untiered])
        self.assertEqual(merge_sources([tiered], [finding])[0].tier, SourceTier.OTHER)


class ClaimExtractionTests(unittest.IsolatedAsyncioTestCase):
    async def test_extraction_output_is_normalized(self):
        result = ClaimExtractionResult(
            input_text="ignored",
            detected_language=" CS ",
            claims=[
                CheckworthyClaim(claim_id="", claim_text="A", source_fragment="a", checkworthy_reason="r"),
                CheckworthyClaim(claim_id="claim_1", claim_text="B", source_fragment="b", checkworthy_reason="r"),
                CheckworthyClaim(claim_id="x", claim_text="C", source_fragment="c", checkworthy_reason="r"),
            ],
        )
        stage = _Stage(lambda payload: result)
        extracted = await ClaimExtractor(stage, max_claims=2).extract_claims("  Transcript text  ")
        self.assertEqual(extracted.input_text, "Transcript text")
        self.assertEqual(extracted.detected_language, "cs")
        self.assertEqual([claim.claim_id for claim in extracted.claims], ["claim_1", "claim_1_2"])
        self.assertEqual(stage.payloads[0]["requirements"]["max_claims"], 2)

    async def test_empty_input_is_refused(self):
        with self.assertRaises(ValueError):
            await ClaimExtractor(_Stage(lambda payload: None), max_claims=2).extract_claims("   ")


class BuildTests(unittest.TestCase):
    ENDPOINT = Endpoint(api_key="key", model="default-model", base_url=None, api_mode="responses")

    def _build(self, config: FactCheckRuntimeConfig, endpoint: Endpoint | None = None):
        with (
            patch("facticli.pipeline.resolve_endpoint", return_value=endpoint or self.ENDPOINT) as resolve,
            patch("facticli.pipeline.configure_client") as configure,
        ):
            checker = build_fact_checker(config)
        return checker, resolve, configure

    def test_stage_routing_tools_and_client_retries(self):
        checker, resolve, configure = self._build(
            FactCheckRuntimeConfig(
                model="m", search_provider="exa", stage_models={"judge": "big"}, model_retry_attempts=5, service_tier="flex"
            )
        )
        resolve.assert_called_once_with(requested_model="m", base_url=None)
        configure.assert_called_once_with(self.ENDPOINT, max_retries=5)
        self.assertEqual(checker.stage_models, {"plan": "default-model", "research": "default-model", "judge": "big"})
        self.assertEqual(checker.judge.model_label, "big")
        research_agent = checker.researcher._agent
        self.assertEqual([tool.name for tool in research_agent.tools], ["exa_web_search"])
        self.assertEqual(research_agent.model_settings.extra_args, {"service_tier": "flex"})
        self.assertEqual(checker.planner._agent.tools, [])

    def test_researcher_plans_for_the_budget_and_the_tools_stop_at_twice_that(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("facticli.pipeline.build_search_tools", wraps=__import__("facticli.llm").llm.build_search_tools) as tools:
                checker, _, _ = self._build(
                    FactCheckRuntimeConfig(search_provider="brave", tool_call_budget=5, search_results_per_query=9, search_cache_dir=tmp)
                )
        self.assertEqual((checker.config.tool_call_budget, checker.search_cap), (5, 10))
        self.assertEqual(tools.call_args.kwargs["search_results_per_query"], 9)
        self.assertEqual(str(tools.call_args.kwargs["cache"].directory), tmp)

    def test_search_cache_needs_a_provider_it_can_record(self):
        for provider in ("openai", "knowledge_store"):
            with self.assertRaisesRegex(ValueError, "--search-provider brave"):
                self._build(FactCheckRuntimeConfig(search_provider=provider, search_cache_dir="cache"))

    def test_claim_extractor_uses_the_same_endpoint_resolution(self):
        with (
            patch("facticli.pipeline.resolve_endpoint", return_value=self.ENDPOINT) as resolve,
            patch("facticli.pipeline.configure_client") as configure,
        ):
            extractor = build_claim_extractor(ClaimExtractionRuntimeConfig(model="m", max_claims=4))
        resolve.assert_called_once_with(requested_model="m", base_url=None)
        configure.assert_called_once_with(self.ENDPOINT, max_retries=3)
        self.assertEqual((extractor.stage.model_label, extractor.max_claims), ("default-model", 4))


class ConstraintTests(unittest.TestCase):
    def test_parse_date_loose_supports_iso_and_averitec_formats(self):
        self.assertEqual(parse_date_loose("2020-10-31").isoformat(), "2020-10-31")
        self.assertEqual(parse_date_loose("31-10-2020").isoformat(), "2020-10-31")
        self.assertIsNone(parse_date_loose("October 2020"))
        self.assertIsNone(parse_date_loose(None))
        self.assertIsNone(parse_date_loose("25-8-20"))  # not the year 20

    def test_normalize_claim_date(self):
        self.assertEqual(normalize_claim_date("25-8-2020"), "2020-08-25")
        self.assertIsNone(normalize_claim_date("unknown"))

    def test_is_blocked_url_matches_domains_subdomains_and_paths(self):
        blocked = ["politifact.com", "reuters.com/fact-check"]
        self.assertTrue(is_blocked_url("https://www.politifact.com/article/1", blocked))
        self.assertTrue(is_blocked_url("https://sub.politifact.com/x", blocked))
        self.assertTrue(is_blocked_url("https://www.reuters.com/fact-check/some-claim", blocked))
        self.assertFalse(is_blocked_url("https://www.reuters.com/world/some-news", blocked))
        self.assertFalse(is_blocked_url("https://notpolitifact.com/x", blocked))

    def test_blocklist_sees_through_ports_dots_and_archive_wrappers(self):
        blocked = ["snopes.com", "reuters.com/fact-check"]
        for url in (
            "https://www.snopes.com:443/fact-check/x",
            "https://snopes.com./fact-check/x",
            "https://web.archive.org/web/20200101000000/https://www.snopes.com/fact-check/x",
            "https://web.archive.org/web/20200101id_/http://reuters.com/fact-check/abc",
            "https://web.archive.org/web/https://snopes.com/x",  # no timestamp
            "https://web.archive.org:443/web/2/https%3A%2F%2Fsnopes.com%2Fx",  # port, short timestamp, encoded
            "https://web.archive.org/web/2020/https://web.archive.org/web/2019/https://snopes.com/x",  # wrapped twice
        ):
            self.assertTrue(is_blocked_url(url, blocked), url)
        self.assertFalse(is_blocked_url("https://web.archive.org/web/20200101000000/https://www.reuters.com/world/x", blocked))
        self.assertEqual(url_domain("https://User@WWW.Example.org:8443/x"), "example.org")
        self.assertEqual(unwrap_archive_url("https://web.archive.org/web/2020/https:/example.org/a"), "https://example.org/a")
        self.assertEqual(unwrap_archive_url("https://example.org/a"), "https://example.org/a")

    def test_violates_date_cutoff(self):
        self.assertTrue(violates_date_cutoff("2021-01-05", "2020-10-31"))
        self.assertFalse(violates_date_cutoff("2020-10-30", "2020-10-31"))
        self.assertFalse(violates_date_cutoff(None, "2020-10-31"))
        self.assertFalse(violates_date_cutoff("2021-01-05", None))


class SourceFilterTests(unittest.TestCase):
    def _finding(self, urls_and_dates: list[tuple[str, str | None]]) -> AspectFinding:
        finding = _finding("check_1")
        finding.sources = [SourceEvidence(title=url, url=url, snippet="snippet", published_at=date) for url, date in urls_and_dates]
        return finding

    def test_filter_removes_blocked_and_late_sources_and_records_them(self):
        token = activate_constraints(ResearchConstraints(claim_date="2020-10-31", blocked_domains=["politifact.com"]))
        try:
            artifact = ResearchCheckArtifact(check=_check("check_1"))
            finding = self._finding(
                [
                    ("https://example.com/a", "2020-01-01"),
                    ("https://politifact.com/b", "2020-01-01"),
                    ("https://example.com/c", "2021-06-01"),
                ]
            )
            filtered = apply_source_constraints(finding, artifact)
        finally:
            deactivate_constraints(token)
        self.assertEqual([source.url for source in filtered.sources], ["https://example.com/a"])
        self.assertEqual(len(artifact.removed_sources), 2)
        self.assertTrue(any("removed by retrieval constraints" in c for c in filtered.caveats))
        self.assertEqual(len(finding.sources), 3)  # the raw finding is untouched for auditability

    def test_filter_is_noop_without_active_constraints(self):
        artifact = ResearchCheckArtifact(check=_check("check_1"))
        filtered = apply_source_constraints(self._finding([("https://politifact.com/b", None)]), artifact)
        self.assertEqual(len(filtered.sources), 1)
        self.assertEqual(artifact.removed_sources, [])


class UsageAndArtifactTests(unittest.TestCase):
    def test_record_and_summarize_usage(self):
        log, token = activate_usage_log()
        try:
            with track_usage(stage="plan", model="m") as event:
                apply_usage(event, SimpleNamespace(requests=1, input_tokens=100, output_tokens=20, total_tokens=120))
            with track_usage(stage="research", model="m") as event:
                apply_usage(event, SimpleNamespace(requests=2, input_tokens=300, output_tokens=50, total_tokens=350))
        finally:
            deactivate_usage_log(token)
        summary = summarize_usage(log.events)
        self.assertEqual((summary.requests, summary.total_tokens), (3, 470))
        self.assertEqual(summary.per_stage["plan"]["calls"], 1)
        self.assertEqual(summary.per_stage["research"]["input_tokens"], 300)

    def test_tracking_outside_an_active_log_is_a_noop(self):
        with track_usage(stage="plan", model="m") as event:
            self.assertIsNone(event)

    def test_save_writes_one_json_file_per_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = save_artifacts(RunArtifacts(claim="C", normalized_claim="C", claim_id="7"), tmp)
            self.assertEqual(list(Path(tmp).glob("run_*_7_*.json")), [path])
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["claim_id"], "7")


if __name__ == "__main__":
    unittest.main()
