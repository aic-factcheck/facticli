from __future__ import annotations

import asyncio
import unittest

import httpx

from facticli.adapters.payloads import index_sources
from facticli.application.citations import CitationHealthChecker, summarize_url_statuses
from facticli.application.progress import ProgressEvent
from facticli.application.services import FactCheckService, SingleAgentFactCheckService
from facticli.application.stages import CitationCheckStage, JudgeStage, ResearchStage, merge_sources
from facticli.core.artifacts import RunArtifacts
from facticli.core.contracts import (
    AspectFinding,
    EvidenceSignal,
    FactCheckReport,
    FindingStatus,
    InvestigationPlan,
    ReviewAction,
    ReviewDecision,
    SourceEvidence,
    SourceGrounding,
    SourceTier,
    UrlStatus,
    VeracityVerdict,
    VerificationCheck,
)
from facticli.core.retrieval import RetrievalRecord, get_journal
from facticli.core.usage import StageUsageEvent, activate_usage_log, deactivate_usage_log, get_usage_log


def _plan(*aspect_ids: str) -> InvestigationPlan:
    return InvestigationPlan(
        claim="claim",
        checks=[
            VerificationCheck(aspect_id=aspect_id, question=f"Q {aspect_id}", rationale="r", search_queries=["q"])
            for aspect_id in aspect_ids
        ],
    )


def _finding(aspect_id: str, urls: list[str] | None = None, signal: EvidenceSignal = EvidenceSignal.SUPPORTS) -> AspectFinding:
    return AspectFinding(
        aspect_id=aspect_id,
        question=f"Q {aspect_id}",
        signal=signal,
        summary=f"summary {aspect_id}",
        confidence=0.8,
        sources=[SourceEvidence(title=url, url=url, snippet="snip") for url in (urls or [])],
    )


def _spend(tokens: int) -> None:
    """Charge the run's ledger the way a research model call would."""
    get_usage_log().events.append(StageUsageEvent(stage="research", model="m", requests=1, input_tokens=tokens, total_tokens=tokens))


class _Researcher:
    def __init__(self, *, fail: set[str] | None = None, error: Exception | None = None, spend: int = 0) -> None:
        self.fail = fail or set()
        self.error = error or RuntimeError("boom")
        self.spend = spend
        self.calls: list[str] = []

    async def research(self, claim: str, check: VerificationCheck) -> AspectFinding:
        self.calls.append(check.aspect_id)
        if self.spend:
            _spend(self.spend)
        if check.aspect_id in self.fail:
            raise self.error
        return _finding(check.aspect_id, ["https://www.reuters.com/world/x", "https://x.com/u/1"])


class ResearchStageHarnessTests(unittest.IsolatedAsyncioTestCase):
    async def test_sources_get_tiers_and_failures_carry_status(self):
        researcher = _Researcher(fail={"b"})
        stage = ResearchStage(researcher=researcher, max_parallel_research=2, research_timeout_seconds=0, research_retry_attempts=1)
        artifacts = RunArtifacts(claim="c", normalized_claim="c")
        events: list[ProgressEvent] = []

        findings = await stage.execute("c", _plan("a", "b"), artifacts, progress_callback=events.append)

        good = next(f for f in findings if f.aspect_id == "a")
        self.assertEqual([s.tier for s in good.sources], [SourceTier.NEWS, SourceTier.USER_GENERATED])
        bad = next(f for f in findings if f.aspect_id == "b")
        self.assertEqual(bad.status, FindingStatus.FAILED)
        self.assertTrue(bad.failure_reason.startswith("unknown: RuntimeError"))
        failed_event = next(e for e in events if e.kind == "research_check_failed")
        self.assertEqual(failed_event.payload["status"], "failed")
        completed_event = next(e for e in events if e.kind == "research_completed")
        self.assertEqual(completed_event.payload["failed_count"], 1)
        self.assertEqual(researcher.calls.count("b"), 2)

    async def test_non_retryable_errors_stop_retrying(self):
        import openai

        response = httpx.Response(400, request=httpx.Request("GET", "https://api"))
        researcher = _Researcher(fail={"a"}, error=openai.BadRequestError("unsupported value", response=response, body=None))
        stage = ResearchStage(researcher=researcher, max_parallel_research=1, research_timeout_seconds=0, research_retry_attempts=3)
        artifacts = RunArtifacts(claim="c", normalized_claim="c")

        findings = await stage.execute("c", _plan("a"), artifacts)

        self.assertEqual(findings[0].status, FindingStatus.FAILED)
        self.assertTrue(findings[0].failure_reason.startswith("bad_request: BadRequestError"))
        self.assertEqual(researcher.calls, ["a"])
        self.assertEqual(artifacts.research_checks[0].error_kinds, ["bad_request"])

    async def test_a_fatal_error_aborts_the_stage_and_cancels_sibling_checks(self):
        import openai

        cancelled: list[str] = []

        class _DeadKey:
            async def research(self, claim: str, check: VerificationCheck) -> AspectFinding:
                if check.aspect_id == "a":
                    response = httpx.Response(401, request=httpx.Request("GET", "https://api"))
                    raise openai.AuthenticationError("denied", response=response, body=None)
                try:
                    await asyncio.sleep(30)
                except asyncio.CancelledError:
                    cancelled.append(check.aspect_id)
                    raise
                return _finding(check.aspect_id)

        stage = ResearchStage(researcher=_DeadKey(), max_parallel_research=3, research_timeout_seconds=0, research_retry_attempts=3)
        artifacts = RunArtifacts(claim="c", normalized_claim="c")
        with self.assertRaises(openai.AuthenticationError):
            await stage.execute("c", _plan("a", "b", "c"), artifacts)
        # Every later request would fail the same way: no retry, and nobody keeps spending.
        self.assertEqual(artifacts.research_checks[0].attempts, 1)
        self.assertEqual(sorted(cancelled), ["b", "c"])

    async def test_a_researcher_that_does_not_journal_leaves_sources_unassessed(self):
        stage = ResearchStage(
            researcher=_Researcher(), max_parallel_research=1, research_timeout_seconds=0, research_retry_attempts=0, search_cap=3
        )
        artifacts = RunArtifacts(claim="c", normalized_claim="c")
        findings = await stage.execute("c", _plan("a"), artifacts)
        self.assertTrue(all(source.grounding is None for source in findings[0].sources))
        self.assertEqual(artifacts.research_checks[0].retrieval, [])

    async def test_budget_exhaustion_skips_remaining_checks(self):
        researcher = _Researcher(spend=600)
        stage = ResearchStage(researcher=researcher, max_parallel_research=1, research_timeout_seconds=0, research_retry_attempts=0)
        artifacts = RunArtifacts(claim="c", normalized_claim="c")
        log, token = activate_usage_log(token_budget=1000)
        try:
            findings = await stage.execute("c", _plan("a", "b", "c"), artifacts)
        finally:
            deactivate_usage_log(token)

        statuses = [f.status for f in findings]
        self.assertEqual(statuses[0], FindingStatus.COMPLETED)
        self.assertEqual(statuses[1], FindingStatus.COMPLETED)  # budget checked before start; second still allowed
        self.assertEqual(statuses[2], FindingStatus.BUDGET_EXHAUSTED)
        self.assertEqual(researcher.calls, ["a", "b"])
        self.assertEqual(log.skipped_stages, ["research:c"])
        self.assertTrue(log.budget_status().token_budget_exhausted)


class _Judge:
    def __init__(self) -> None:
        self.payload_findings: list[AspectFinding] = []

    async def judge(self, claim, plan, findings):
        self.payload_findings = findings
        return FactCheckReport(
            claim="x",
            verdict=VeracityVerdict.SUPPORTED,
            verdict_confidence=0.8,
            justification="j [S1]",
            key_points=[" a ", "a", ""],
            counter_argument="Could be refuted if ...",
            evidence_gaps=["gap"],
            # The model echoes findings without harness-only fields; they must not overwrite ours.
            findings=[_finding("a"), _finding("b")],
            sources=[SourceEvidence(title="extra", url="https://www.czso.cz/x", snippet="s")],
        )


class JudgeStageHarnessTests(unittest.IsolatedAsyncioTestCase):
    async def test_harness_findings_are_authoritative_and_sources_tiered(self):
        stage = JudgeStage(judge=_Judge())
        artifacts = RunArtifacts(claim="c", normalized_claim="c")
        findings = [
            _finding("a", ["https://www.reuters.com/world/x"]),
            AspectFinding(
                aspect_id="b", question="Q b", signal=EvidenceSignal.INSUFFICIENT, summary="failed", confidence=0.0,
                status=FindingStatus.FAILED, failure_reason="timeout",
            ),
        ]
        findings[0].sources[0].tier = SourceTier.NEWS

        report = await stage.execute("c", _plan("a", "b"), findings, artifacts)

        self.assertEqual(report.findings[1].status, FindingStatus.FAILED)
        self.assertEqual(report.key_points, ["a"])
        self.assertEqual(report.counter_argument, "Could be refuted if ...")
        urls = [s.url for s in report.sources]
        self.assertEqual(urls, ["https://www.reuters.com/world/x", "https://www.czso.cz/x"])
        self.assertEqual([s.tier for s in report.sources], [SourceTier.NEWS, SourceTier.PRIMARY])

    async def test_cited_source_ids_must_exist_in_the_source_table(self):
        class _CitingJudge:
            async def judge(self, claim, plan, findings):
                return FactCheckReport(
                    claim=claim, verdict=VeracityVerdict.SUPPORTED, verdict_confidence=0.8,
                    justification="Confirmed [S1][S2]. Also [S5].", key_points=["Point [S2]"], counter_argument="None [S9].",
                )

        findings = [_finding("a", ["https://www.reuters.com/x", "https://www.bbc.com/y"]), _finding("b", ["https://www.bbc.com/y"])]
        artifacts = RunArtifacts(claim="c", normalized_claim="c")
        events: list[ProgressEvent] = []
        await JudgeStage(judge=_CitingJudge()).execute("c", _plan("a", "b"), findings, artifacts, progress_callback=events.append)
        self.assertEqual(artifacts.unresolved_citations, [5, 9])  # the table has two rows

    async def test_a_source_only_the_judge_lists_is_labelled_untraceable(self):
        class _MiscopyingJudge:
            async def judge(self, claim, plan, findings):
                return FactCheckReport(
                    claim=claim, verdict=VeracityVerdict.SUPPORTED, verdict_confidence=0.8, justification="Confirmed [S1][S2].",
                    counter_argument="c",
                    sources=[
                        SourceEvidence(title="copied", url="https://www.reuters.com/x", snippet="s"),
                        SourceEvidence(title="miscopied", url="https://www.reuters.com/y", snippet="s"),
                    ],
                )

        traced = _finding("a", ["https://www.reuters.com/x"])
        traced.sources[0].grounding = SourceGrounding.QUOTE_MATCHED
        artifacts = RunArtifacts(claim="c", normalized_claim="c")
        report = await JudgeStage(judge=_MiscopyingJudge()).execute("c", _plan("a"), [traced], artifacts)
        self.assertEqual(
            [(source.url, source.grounding) for source in report.sources],
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
        merged = merge_sources([untiered], [_finding("a")])
        self.assertEqual(len(merged), 1)
        merged = merge_sources([tiered], [AspectFinding(aspect_id="a", question="q", signal=EvidenceSignal.SUPPORTS, summary="s", confidence=0.5, sources=[untiered])])
        self.assertEqual(merged[0].tier, SourceTier.OTHER)


class CitationCheckTests(unittest.IsolatedAsyncioTestCase):
    def _checker(self, handler) -> CitationHealthChecker:
        transport = httpx.MockTransport(handler)
        return CitationHealthChecker(
            timeout_seconds=1,
            client_factory=lambda: httpx.AsyncClient(transport=transport, follow_redirects=True),
        )

    async def test_live_archived_and_broken(self):
        def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if url.startswith("https://archive.org/wayback/available"):
                target = request.url.params.get("url", "")
                if "gone-but-archived" in target:
                    return httpx.Response(200, json={"archived_snapshots": {"closest": {"available": True, "url": "https://web.archive.org/web/2020/x"}}})
                return httpx.Response(200, json={"archived_snapshots": {}})
            if "live" in url:
                return httpx.Response(200 if request.method == "HEAD" else 200)
            if "paywall" in url:
                return httpx.Response(403)
            if "head-blocked" in url:
                return httpx.Response(405) if request.method == "HEAD" else httpx.Response(200)
            return httpx.Response(404)

        checker = self._checker(handler)
        sources = [
            SourceEvidence(title="a", url="https://example.org/live", snippet="s"),
            SourceEvidence(title="b", url="https://example.org/paywall", snippet="s"),
            SourceEvidence(title="c", url="https://example.org/head-blocked", snippet="s"),
            SourceEvidence(title="d", url="https://example.org/gone-but-archived", snippet="s"),
            SourceEvidence(title="e", url="https://example.org/gone", snippet="s"),
            SourceEvidence(title="e-dup", url="https://example.org/gone", snippet="s"),
        ]
        checked = await checker.check_sources(sources)
        statuses = [s.url_status for s in checked]
        self.assertEqual(
            statuses,
            [UrlStatus.LIVE, UrlStatus.LIVE, UrlStatus.LIVE, UrlStatus.ARCHIVED, UrlStatus.BROKEN, UrlStatus.BROKEN],
        )
        self.assertEqual(checked[3].archived_url, "https://web.archive.org/web/2020/x")
        self.assertEqual(summarize_url_statuses(checked), {"live": 3, "archived": 1, "broken": 2})
        # Originals untouched
        self.assertIsNone(sources[0].url_status)

    async def test_network_errors_count_as_broken_and_stage_annotates_findings(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if str(request.url).startswith("https://archive.org"):
                return httpx.Response(500)
            raise httpx.ConnectError("refused", request=request)

        stage = CitationCheckStage(checker=self._checker(handler))
        source = SourceEvidence(title="x", url="https://down.example/x", snippet="s")
        report = FactCheckReport(
            claim="c", verdict=VeracityVerdict.SUPPORTED, verdict_confidence=0.5, justification="j",
            findings=[AspectFinding(aspect_id="a", question="q", signal=EvidenceSignal.SUPPORTS, summary="s", confidence=0.5, sources=[source])],
            sources=[source],
        )
        artifacts = RunArtifacts(claim="c", normalized_claim="c")
        events: list[ProgressEvent] = []

        checked = await stage.execute(report, artifacts, progress_callback=events.append)

        self.assertEqual(checked.sources[0].url_status, UrlStatus.BROKEN)
        self.assertEqual(checked.findings[0].sources[0].url_status, UrlStatus.BROKEN)
        self.assertEqual(artifacts.citation_check, {"broken": 1})
        self.assertEqual([e.kind for e in events], ["citation_check_started", "citation_check_completed"])


class _PlanStage:
    async def execute(self, claim, artifacts, progress_callback=None):
        plan = _plan("a")
        artifacts.plan_raw = plan
        artifacts.plan_normalized = plan
        return plan


class _ResearchStage:
    def __init__(self, spend: int = 0) -> None:
        self.calls = 0
        self.spend = spend

    async def execute(self, claim, plan, artifacts, progress_callback=None):
        self.calls += 1
        if self.spend:
            _spend(self.spend)
        return [_finding(check.aspect_id) for check in plan.checks]


class _JudgeStage:
    async def execute(self, claim, plan, findings, artifacts, progress_callback=None):
        report = FactCheckReport(claim=claim, verdict=VeracityVerdict.SUPPORTED, verdict_confidence=0.9, justification="j", findings=findings)
        artifacts.report_final = report
        return report


class _ReviewStage:
    def __init__(self) -> None:
        self.calls = 0

    async def execute(self, claim, plan, findings, artifacts, *, round_index, progress_callback=None):
        self.calls += 1
        return ReviewDecision(
            claim=claim, action=ReviewAction.FOLLOW_UP, rationale="more", gaps=["g"],
            follow_up_checks=[VerificationCheck(aspect_id=f"f{round_index}", question="q", rationale="r", search_queries=["q"])],
        )


class ServiceBudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_budget_exhaustion_skips_review_rounds_and_is_recorded(self):
        review = _ReviewStage()
        service = FactCheckService(
            plan_stage=_PlanStage(),
            research_stage=_ResearchStage(spend=1000),
            judge_stage=_JudgeStage(),
            review_stage=review,
            max_feedback_rounds=2,
            token_budget=1000,
            retry_budget=3,
            stage_models={"plan": "m1", "judge": "m2"},
        )
        events: list[ProgressEvent] = []

        run = await service.check_claim("claim", progress_callback=events.append)

        self.assertEqual(review.calls, 0)
        self.assertTrue(run.artifacts.budget.token_budget_exhausted)
        self.assertEqual(run.artifacts.budget.skipped_stages, ["review_round_1"])
        self.assertEqual(run.artifacts.stage_models, {"plan": "m1", "judge": "m2"})
        self.assertEqual(run.artifacts.strategy, "pipeline")
        self.assertIn("feedback_round_skipped", [e.kind for e in events])
        completed = next(e for e in events if e.kind == "run_completed")
        self.assertTrue(completed.payload["budget"]["token_budget_exhausted"])

    async def test_without_budget_review_rounds_run(self):
        review = _ReviewStage()
        research = _ResearchStage()
        service = FactCheckService(
            plan_stage=_PlanStage(), research_stage=research, judge_stage=_JudgeStage(), review_stage=review, max_feedback_rounds=2,
        )
        run = await service.check_claim("claim")
        self.assertEqual(review.calls, 2)
        self.assertEqual(research.calls, 3)
        self.assertEqual([c.aspect_id for c in run.plan.checks], ["a", "f1", "f2"])
        self.assertIsNone(run.artifacts.budget.token_budget)

    def test_merge_findings_keeps_observation_over_failed_retry(self):
        service = FactCheckService(plan_stage=_PlanStage(), research_stage=_ResearchStage(), judge_stage=_JudgeStage())
        good = _finding("a")
        failed = AspectFinding(aspect_id="a", question="q", signal=EvidenceSignal.INSUFFICIENT, summary="f", confidence=0.0, status=FindingStatus.FAILED)
        merged = service._merge_findings([good], [failed])
        self.assertIs(merged[0], good)
        merged = service._merge_findings([failed], [good])
        self.assertIs(merged[0], good)


class _SingleAgent:
    async def check(self, claim: str, max_checks: int) -> FactCheckReport:
        return FactCheckReport(
            claim="ignored",
            verdict=VeracityVerdict.REFUTED,
            verdict_confidence=0.7,
            justification="j",
            key_points=["k"],
            counter_argument="c",
            findings=[
                _finding("", ["https://www.politifact.com/x", "https://www.bbc.com/news/1"], EvidenceSignal.REFUTES),
                _finding("dates", ["https://www.czso.cz/x"]),
                _finding("extra_over_limit"),
            ],
            sources=[SourceEvidence(title="r", url="https://en.wikipedia.org/wiki/X", snippet="s")],
        )


class _JournalingAgent:
    """Single agent whose "tool calls" land in the journal the service installed."""

    async def check(self, claim: str, max_checks: int) -> FactCheckReport:
        journal = get_journal()
        journal.complete = True
        self.search_cap = journal.search_cap
        quote = "The council approved the budget on Monday evening."
        journal.add(RetrievalRecord(provider="brave", query="q", urls=["https://www.bbc.com/news/1"]), [quote])
        seen = SourceEvidence(title="bbc", url="https://www.bbc.com/news/1", snippet=quote)
        invented = SourceEvidence(title="made up", url="https://example.com/nope", snippet=quote)
        return FactCheckReport(
            claim=claim, verdict=VeracityVerdict.SUPPORTED, verdict_confidence=0.8, justification="j", counter_argument="c",
            findings=[AspectFinding(aspect_id="vote", question="Q", signal=EvidenceSignal.SUPPORTS, summary="s", confidence=0.8, sources=[seen])],
            sources=[seen, invented],
        )


class RunContextTests(unittest.IsolatedAsyncioTestCase):
    async def test_single_agent_run_is_capped_and_grounded_like_the_pipeline(self):
        agent = _JournalingAgent()
        service = SingleAgentFactCheckService(checker=agent, max_checks=2, search_cap=64, stage_efforts={"single_agent": "low"})
        run = await service.check_claim("The claim.")

        self.assertEqual(agent.search_cap, 64)
        self.assertEqual(run.findings[0].sources[0].grounding, SourceGrounding.QUOTE_MATCHED)
        self.assertEqual(
            [(source.url, source.grounding) for source in run.report.sources],
            [("https://www.bbc.com/news/1", SourceGrounding.QUOTE_MATCHED), ("https://example.com/nope", SourceGrounding.UNMATCHED)],
        )
        self.assertEqual(run.artifacts.retrieval_summary.searches, 1)
        self.assertEqual(run.artifacts.stage_efforts, {"single_agent": "low"})
        self.assertIsNone(get_journal())  # the journal does not outlive the run

    async def test_the_claim_is_evaluated_as_of_its_date_or_else_today(self):
        service = SingleAgentFactCheckService(checker=_JournalingAgent(), max_checks=2)
        dated = (await service.check_claim("The claim.", claim_date="25-8-2020")).artifacts.constraints
        self.assertEqual((dated.claim_date, dated.reference_date), ("2020-08-25", "2020-08-25"))
        undated = (await service.check_claim("The claim.", claim_date="  ")).artifacts.constraints
        self.assertIsNone(undated.claim_date)
        self.assertRegex(undated.reference_date, r"^\d{4}-\d{2}-\d{2}$")

    async def test_unparseable_claim_date_stops_the_run_instead_of_dropping_the_cutoff(self):
        service = SingleAgentFactCheckService(checker=_JournalingAgent(), max_checks=2)
        for bad in ("8-25-2020", "25.8.2020", "last week"):
            with self.assertRaisesRegex(ValueError, "Unparseable claim date"):
                await service.check_claim("The claim.", claim_date=bad)


class SingleAgentServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_single_agent_service_applies_constraints_tiers_and_plan_reconstruction(self):
        service = SingleAgentFactCheckService(checker=_SingleAgent(), max_checks=2, blocked_domains=("politifact.com",), token_budget=5000)
        events: list[ProgressEvent] = []

        run = await service.check_claim("The claim.", progress_callback=events.append)

        self.assertEqual(run.artifacts.strategy, "single_agent")
        self.assertEqual([c.aspect_id for c in run.plan.checks], ["aspect_1", "dates"])
        self.assertEqual(len(run.findings), 2)
        first = run.findings[0]
        self.assertEqual([s.url for s in first.sources], ["https://www.bbc.com/news/1"])
        self.assertEqual(first.sources[0].tier, SourceTier.NEWS)
        self.assertTrue(any("removed by retrieval constraints" in c for c in first.caveats))
        self.assertEqual(
            [s.url for s in run.report.sources],
            ["https://www.bbc.com/news/1", "https://www.czso.cz/x", "https://en.wikipedia.org/wiki/X"],
        )
        self.assertEqual(run.report.sources[2].tier, SourceTier.REFERENCE)
        self.assertEqual(run.report.claim, "The claim.")
        kinds = [e.kind for e in events]
        self.assertEqual(kinds[0], "run_started")
        self.assertIn("single_agent_started", kinds)
        self.assertEqual(kinds[-1], "run_completed")
        self.assertEqual(run.artifacts.budget.token_budget, 5000)


if __name__ == "__main__":
    unittest.main()
