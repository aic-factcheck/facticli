from __future__ import annotations

import asyncio
from dataclasses import dataclass

from facticli.core.artifacts import ResearchCheckArtifact, RunArtifacts
from facticli.core.constraints import get_constraints, is_blocked_url, violates_date_cutoff
from facticli.core.contracts import (
    AspectFinding,
    ClaimExtractionResult,
    EvidenceSignal,
    FactCheckReport,
    FindingStatus,
    InvestigationPlan,
    ReviewAction,
    ReviewDecision,
    SourceEvidence,
    SourceGrounding,
    VerificationCheck,
)
from facticli.core.errors import ErrorKind, classify_exception, describe_exception, is_fatal
from facticli.core.grounding import ground_finding, unresolved_citation_ids
from facticli.core.normalize import normalize_plan_checks, normalize_query_list, normalize_source_url, normalize_text_list
from facticli.core.retrieval import RetrievalJournal, activate_journal, deactivate_journal
from facticli.core.source_quality import assign_source_tiers
from facticli.core.usage import get_usage_log

from .citations import CitationHealthChecker, summarize_url_statuses
from .interfaces import ClaimExtractionBackend, Judge, Planner, Researcher, Reviewer
from .progress import ProgressCallback, emit_progress

# Error kinds that re-running the same request cannot fix; the check fails
# without further attempts. Fatal kinds (auth, quota, not found) do not get
# this far: they fail every check the same way, so they abort the run.
_NO_RETRY_KINDS: frozenset[ErrorKind] = frozenset(
    {
        ErrorKind.FORBIDDEN,
        ErrorKind.BAD_REQUEST,
        ErrorKind.CONTEXT_OVERFLOW,
        ErrorKind.REFUSAL,
        ErrorKind.BUDGET,
    }
)


def _apply_source_constraints(finding: AspectFinding, artifact: ResearchCheckArtifact) -> AspectFinding:
    """Drop constraint-violating sources from a finding; keep them in the artifact.

    The unfiltered finding stays in the artifact for post-hoc leakage auditing;
    removed sources are recorded separately so audits do not have to re-derive
    the filtering decision.
    """
    constraints = get_constraints()
    if constraints is None or (not constraints.blocked_domains and not constraints.claim_date):
        return finding

    kept: list[SourceEvidence] = []
    removed: list[SourceEvidence] = []
    for source in finding.sources:
        if is_blocked_url(source.url, constraints.blocked_domains):
            removed.append(source)
        elif violates_date_cutoff(source.published_at, constraints.claim_date):
            removed.append(source)
        else:
            kept.append(source)

    if not removed:
        return finding

    artifact.removed_sources.extend(removed)
    caveat = (
        f"{len(removed)} source(s) removed by retrieval constraints "
        "(fact-checking domain blocklist or evidence date cutoff)."
    )
    return finding.model_copy(update={"sources": kept, "caveats": [*finding.caveats, caveat]})


def _tier_finding_sources(finding: AspectFinding) -> AspectFinding:
    """Attach deterministic authority tiers to a finding's sources."""
    if not finding.sources:
        return finding
    return finding.model_copy(update={"sources": assign_source_tiers(finding.sources)})


def failed_finding(check: VerificationCheck, *, status: FindingStatus, reason: str, attempts: int) -> AspectFinding:
    """Build the placeholder finding for a check the harness could not complete."""
    if status == FindingStatus.BUDGET_EXHAUSTED:
        summary = "Check skipped: the run's shared token budget was exhausted before it could run."
        caveat = "Not researched (budget exhausted); this is a missing observation, not evidence."
    else:
        summary = f"Research subroutine failed after {attempts} attempt(s): {reason}"
        caveat = "Research failed; this is a missing observation, not evidence of absence."
    return AspectFinding(
        aspect_id=check.aspect_id,
        question=check.question,
        signal=EvidenceSignal.INSUFFICIENT,
        summary=summary,
        confidence=0.0,
        sources=[],
        caveats=[caveat],
        status=status,
        failure_reason=reason,
    )


@dataclass(frozen=True)
class PlanStage:
    """Runs planning and normalizes checks into an executable investigation plan."""
    planner: Planner
    max_checks: int
    max_search_queries_per_check: int

    async def execute(
        self,
        claim: str,
        artifacts: RunArtifacts,
        progress_callback: ProgressCallback | None = None,
    ) -> InvestigationPlan:
        """Generate a robust plan, including a direct-check fallback when needed."""
        await emit_progress(progress_callback, "planning_started", {"claim": claim})
        plan_raw = await self.planner.plan(claim=claim, max_checks=self.max_checks)
        artifacts.plan_raw = plan_raw

        checks = normalize_plan_checks(
            claim=claim,
            checks=plan_raw.checks,
            max_checks=self.max_checks,
            max_search_queries_per_check=self.max_search_queries_per_check,
        )
        if not checks:
            checks = [direct_check(claim, max_queries=self.max_search_queries_per_check)]

        plan = plan_raw.model_copy(update={"claim": claim, "checks": checks})
        artifacts.plan_normalized = plan
        await emit_progress(
            progress_callback,
            "planning_completed",
            {
                "claim": claim,
                "check_count": len(plan.checks),
                "checks": [
                    {
                        "aspect_id": check.aspect_id,
                        "question": check.question,
                    }
                    for check in plan.checks
                ],
            },
        )
        return plan


def direct_check(claim: str, *, max_queries: int) -> VerificationCheck:
    """Fallback single check used when planning fails or is bypassed."""
    return VerificationCheck(
        aspect_id="claim_direct_check",
        question=f"Is this claim accurate: {claim}",
        rationale="Fallback direct verification when planning fails.",
        search_queries=normalize_query_list([claim], max_queries=max_queries),
        acceptance_criteria=["The central assertion is confirmed or contradicted by at least two independent sources."],
    )


@dataclass(frozen=True)
class ResearchStage:
    """Runs check-level research concurrently with retries, budget, and timeout safeguards.

    Every attempt runs under its own retrieval journal. The journal enforces
    ``search_cap`` (the hard per-check limit, above the budget the researcher
    is told to plan for) inside the search tools and, once the researcher
    returns, is the reference the finding's sources are grounded against.
    """
    researcher: Researcher
    max_parallel_research: int
    research_timeout_seconds: float
    research_retry_attempts: int
    search_cap: int | None = None

    async def execute(
        self,
        claim: str,
        plan: InvestigationPlan,
        artifacts: RunArtifacts,
        progress_callback: ProgressCallback | None = None,
    ) -> list[AspectFinding]:
        """Return ordered findings; failed or unbudgeted checks become explicit non-observations.

        An error no retry can fix (rejected key, exhausted quota, unknown
        model) is raised instead: it would fail every check the same way, and
        a verdict judged on none of them is worth nothing.
        """
        semaphore = asyncio.Semaphore(max(1, self.max_parallel_research))
        timeout = self.research_timeout_seconds or None
        max_attempts = 1 + max(0, self.research_retry_attempts)
        await emit_progress(
            progress_callback,
            "research_started",
            {"claim": claim, "check_count": len(plan.checks)},
        )

        async def run_check(index: int, check: VerificationCheck) -> tuple[int, VerificationCheck, AspectFinding]:
            artifact = artifacts.get_or_create_check(check)
            async with semaphore:
                usage_log = get_usage_log()
                if usage_log is not None and usage_log.token_budget_exhausted():
                    usage_log.record_skipped_stage(f"research:{check.aspect_id}")
                    finding = failed_finding(
                        check,
                        status=FindingStatus.BUDGET_EXHAUSTED,
                        reason="token budget exhausted before this check started",
                        attempts=0,
                    )
                    artifact.finding = finding
                    return index, check, finding

                last_error: Exception | None = None
                for _attempt in range(max_attempts):
                    artifact.attempts += 1
                    journal = RetrievalJournal(search_cap=self.search_cap)
                    journal_token = activate_journal(journal)
                    try:
                        task = self.researcher.research(claim=claim, check=check)
                        if timeout:
                            finding = await asyncio.wait_for(task, timeout=timeout)
                        else:
                            finding = await task
                        # Provenance first, on the sources exactly as the model returned them.
                        finding = _tier_finding_sources(ground_finding(finding, journal))
                        # Artifact keeps the unfiltered (but tiered and grounded) finding for leakage audits.
                        artifact.finding = finding
                        return index, check, _apply_source_constraints(finding, artifact)
                    except Exception as exc:  # pragma: no cover - exercised via fakes
                        last_error = exc
                        kind = classify_exception(exc)
                        artifact.errors.append(f"{type(exc).__name__}: {exc}")
                        artifact.error_kinds.append(kind.value)
                        if is_fatal(kind):
                            raise
                        if kind in _NO_RETRY_KINDS:
                            break
                    finally:
                        deactivate_journal(journal_token)
                        artifact.retrieval.extend(journal.records)

                assert last_error is not None
                finding = failed_finding(
                    check,
                    status=FindingStatus.FAILED,
                    reason=describe_exception(last_error),
                    attempts=artifact.attempts,
                )
                artifact.finding = finding
                return index, check, finding

        tasks = [
            asyncio.create_task(run_check(index, check)) for index, check in enumerate(plan.checks)
        ]

        ordered_findings: list[AspectFinding | None] = [None] * len(plan.checks)
        try:
            for task in asyncio.as_completed(tasks):
                index, check, finding = await task
                ordered_findings[index] = finding
                if finding.status != FindingStatus.COMPLETED:
                    await emit_progress(
                        progress_callback,
                        "research_check_failed",
                        {
                            "aspect_id": check.aspect_id,
                            "question": check.question,
                            "status": finding.status.value,
                            "error": finding.failure_reason or "",
                            "attempts": artifacts.get_or_create_check(check).attempts,
                            "finding": finding.model_dump(),
                        },
                    )
                    continue

                await emit_progress(
                    progress_callback,
                    "research_check_completed",
                    {
                        "aspect_id": finding.aspect_id,
                        "question": finding.question,
                        "signal": finding.signal.value,
                        "confidence": finding.confidence,
                        "summary": finding.summary,
                        "source_count": len(finding.sources),
                        "stop_reason": finding.stop_reason.value if finding.stop_reason else None,
                    },
                )
        except BaseException:
            # A fatal error or a cancelled run must not leave sibling checks spending tokens.
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        findings = [finding for finding in ordered_findings if finding is not None]
        await emit_progress(
            progress_callback,
            "research_completed",
            {
                "finding_count": len(findings),
                "failed_count": sum(1 for finding in findings if finding.status != FindingStatus.COMPLETED),
            },
        )
        return findings


@dataclass(frozen=True)
class JudgeStage:
    """Builds the final report and normalizes source links for output consistency."""
    judge: Judge

    async def execute(
        self,
        claim: str,
        plan: InvestigationPlan,
        findings: list[AspectFinding],
        artifacts: RunArtifacts,
        progress_callback: ProgressCallback | None = None,
    ) -> FactCheckReport:
        """Synthesize and normalize the final report from collected findings.

        The harness-held findings are authoritative in the final report: they
        carry status, tiers, and constraint filtering the model never sees.
        """
        await emit_progress(
            progress_callback,
            "judging_started",
            {"claim": claim, "finding_count": len(findings)},
        )
        report_raw = await self.judge.judge(claim=claim, plan=plan, findings=findings)
        artifacts.report_raw = report_raw

        # The judge has no search tool. A source it lists that no researcher
        # cited is a mis-copied or remembered URL, and is labelled as such.
        researched = {normalize_source_url(source.url) for finding in findings for source in finding.sources}
        judge_sources = [
            source.model_copy(update={"grounding": SourceGrounding.UNMATCHED})
            for source in report_raw.sources
            if normalize_source_url(source.url) not in researched
        ]
        report_final = report_raw.model_copy(
            update={
                "claim": claim,
                "findings": findings if findings else report_raw.findings,
                "sources": assign_source_tiers(merge_sources(judge_sources, findings)),
                "key_points": normalize_text_list(report_raw.key_points, max_items=12),
                "evidence_gaps": normalize_text_list(report_raw.evidence_gaps, max_items=12),
            }
        )
        artifacts.report_final = report_final
        # The source list starts with the judge's table S1..Sn in order, so the
        # ids in its prose index this list; ids beyond it point at nothing.
        artifacts.unresolved_citations = unresolved_citation_ids(
            [report_final.justification, *report_final.key_points, report_final.counter_argument],
            source_count=len(report_final.sources),
        )
        await emit_progress(
            progress_callback,
            "judging_completed",
            {
                "verdict": report_final.verdict.value,
                "verdict_confidence": report_final.verdict_confidence,
                "source_count": len(report_final.sources),
            },
        )
        return report_final


def merge_sources(
    report_sources: list[SourceEvidence],
    findings: list[AspectFinding],
) -> list[SourceEvidence]:
    """Deduplicate report and finding sources by normalized URL, preferring the better copy.

    Finding sources come first, in order, so position ``n`` in the result is
    source id ``S{n}`` of the judge's source table. When two checks cite one
    URL, the copy that was traced to a search result wins: each check is
    grounded against its own searches, and a URL retrieved by either was
    retrieved.
    """
    combined: list[SourceEvidence] = []
    seen: dict[str, int] = {}

    def add(source: SourceEvidence) -> None:
        normalized = normalize_source_url(source.url)
        if not normalized:
            return
        if normalized in seen:
            existing = combined[seen[normalized]]
            traced_elsewhere = existing.grounding == SourceGrounding.UNMATCHED and source.grounding not in (
                None,
                SourceGrounding.UNMATCHED,
            )
            if traced_elsewhere or (existing.tier is None and source.tier is not None):
                combined[seen[normalized]] = source
            return
        seen[normalized] = len(combined)
        combined.append(source)

    for finding in findings:
        for source in finding.sources:
            add(source)
    for source in report_sources:
        add(source)

    return combined


@dataclass(frozen=True)
class ReviewStage:
    """Grades findings against acceptance criteria; requests bounded follow-up checks."""
    reviewer: Reviewer
    max_follow_up_checks: int
    max_search_queries_per_check: int

    async def execute(
        self,
        claim: str,
        plan: InvestigationPlan,
        findings: list[AspectFinding],
        artifacts: RunArtifacts,
        *,
        round_index: int,
        progress_callback: ProgressCallback | None = None,
    ) -> ReviewDecision:
        """Produce a bounded follow-up decision and normalize requested checks."""
        await emit_progress(
            progress_callback,
            "review_started",
            {
                "claim": claim,
                "round_index": round_index,
                "finding_count": len(findings),
            },
        )
        review_artifact = artifacts.add_review_round(
            round_index=round_index,
            input_plan=plan,
            input_findings=findings,
        )
        decision_raw = await self.reviewer.review(
            claim=claim,
            plan=plan,
            findings=findings,
            max_follow_up_checks=self.max_follow_up_checks,
            round_index=round_index,
        )

        normalized_action = decision_raw.action
        retry_aspect_ids = self._normalize_retry_aspect_ids(decision_raw.retry_aspect_ids, plan)
        follow_up_checks = normalize_plan_checks(
            claim=claim,
            checks=decision_raw.follow_up_checks,
            max_checks=self.max_follow_up_checks,
            max_search_queries_per_check=self.max_search_queries_per_check,
        )
        follow_up_checks = self._dedupe_follow_up_checks(follow_up_checks, plan)

        if normalized_action == ReviewAction.FOLLOW_UP and not retry_aspect_ids and not follow_up_checks:
            normalized_action = ReviewAction.FINALIZE

        decision_final = decision_raw.model_copy(
            update={
                "claim": claim,
                "action": normalized_action,
                "gaps": normalize_text_list(decision_raw.gaps, max_items=12),
                "retry_aspect_ids": retry_aspect_ids,
                "follow_up_checks": follow_up_checks,
            }
        )
        review_artifact.decision = decision_final
        if follow_up_checks:
            review_artifact.follow_up_plan = InvestigationPlan(
                claim=claim,
                checks=follow_up_checks,
                assumptions=[],
            )

        await emit_progress(
            progress_callback,
            "review_completed",
            {
                "round_index": round_index,
                "action": decision_final.action.value,
                "retry_count": len(decision_final.retry_aspect_ids),
                "follow_up_count": len(decision_final.follow_up_checks),
                "gap_count": len(decision_final.gaps),
            },
        )
        return decision_final

    def _normalize_retry_aspect_ids(
        self,
        retry_aspect_ids: list[str],
        plan: InvestigationPlan,
    ) -> list[str]:
        available = {check.aspect_id for check in plan.checks}
        normalized: list[str] = []
        seen: set[str] = set()
        for aspect_id in retry_aspect_ids:
            candidate = aspect_id.strip()
            if not candidate or candidate not in available or candidate in seen:
                continue
            seen.add(candidate)
            normalized.append(candidate)
        return normalized

    def _dedupe_follow_up_checks(
        self,
        follow_up_checks: list[VerificationCheck],
        plan: InvestigationPlan,
    ) -> list[VerificationCheck]:
        used_aspect_ids = {check.aspect_id for check in plan.checks}
        deduped: list[VerificationCheck] = []
        for check in follow_up_checks:
            aspect_id = check.aspect_id
            suffix = 2
            while aspect_id in used_aspect_ids:
                aspect_id = f"{check.aspect_id}_{suffix}"
                suffix += 1
            used_aspect_ids.add(aspect_id)
            deduped.append(check.model_copy(update={"aspect_id": aspect_id}))
        return deduped


@dataclass(frozen=True)
class CitationCheckStage:
    """Post-judge, model-free pass that marks each cited URL live, archived, or broken."""
    checker: CitationHealthChecker

    async def execute(
        self,
        report: FactCheckReport,
        artifacts: RunArtifacts,
        progress_callback: ProgressCallback | None = None,
    ) -> FactCheckReport:
        await emit_progress(progress_callback, "citation_check_started", {"source_count": len(report.sources)})
        checked_sources = await self.checker.check_sources(report.sources)
        status_by_url = {
            normalize_source_url(source.url): (source.url_status, source.archived_url) for source in checked_sources
        }

        def annotate(source: SourceEvidence) -> SourceEvidence:
            status, archived = status_by_url.get(normalize_source_url(source.url), (None, None))
            if status is None or source.url_status is not None:
                return source
            return source.model_copy(update={"url_status": status, "archived_url": archived})

        findings = [
            finding.model_copy(update={"sources": [annotate(source) for source in finding.sources]})
            for finding in report.findings
        ]
        report_final = report.model_copy(update={"sources": checked_sources, "findings": findings})
        counts = summarize_url_statuses(checked_sources)
        artifacts.citation_check = counts
        artifacts.report_final = report_final
        await emit_progress(progress_callback, "citation_check_completed", {"statuses": counts})
        return report_final


@dataclass(frozen=True)
class ClaimExtractionStage:
    """Normalizes claim-extraction output for downstream CLI and automation use."""
    backend: ClaimExtractionBackend
    max_claims: int

    async def execute(self, input_text: str) -> ClaimExtractionResult:
        """Extract and sanitize check-worthy claims from raw input text."""
        normalized_text = input_text.strip()
        if not normalized_text:
            raise ValueError("Input text is empty.")

        extraction = await self.backend.extract(input_text=normalized_text, max_claims=self.max_claims)
        extraction.input_text = normalized_text
        extraction.detected_language = extraction.detected_language.strip().lower()
        extraction.claims = extraction.claims[: self.max_claims]

        seen_ids: set[str] = set()
        for index, claim in enumerate(extraction.claims, start=1):
            if not claim.claim_id.strip():
                claim.claim_id = f"claim_{index}"
            if claim.claim_id in seen_ids:
                claim.claim_id = f"{claim.claim_id}_{index}"
            seen_ids.add(claim.claim_id)

        return extraction
