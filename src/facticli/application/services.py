from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from facticli.core.artifacts import RunArtifacts
from facticli.core.constraints import (
    ResearchConstraints,
    activate_constraints,
    deactivate_constraints,
    normalize_claim_date,
)
from facticli.core.contracts import (
    AspectFinding,
    ClaimExtractionResult,
    EvidenceSignal,
    FactCheckReport,
    InvestigationPlan,
    ReviewAction,
    VerificationCheck,
)
from facticli.core.grounding import ground_finding, ground_sources
from facticli.core.normalize import normalize_text_list
from facticli.core.retrieval import RetrievalJournal, activate_journal, deactivate_journal, summarize_retrieval
from facticli.core.source_quality import assign_source_tiers
from facticli.core.usage import UsageLog, activate_usage_log, deactivate_usage_log, summarize_usage

from .interfaces import SingleAgentChecker
from .progress import ProgressCallback, emit_progress
from .repository import RunArtifactRepository
from .stages import (
    CitationCheckStage,
    ClaimExtractionStage,
    JudgeStage,
    PlanStage,
    ResearchStage,
    ReviewStage,
    _apply_source_constraints,
    direct_check,
    merge_sources,
)


@dataclass
class FactCheckRun:
    """Container holding all outputs produced by one fact-check execution."""
    claim: str
    plan: InvestigationPlan
    findings: list[AspectFinding]
    report: FactCheckReport
    artifacts: RunArtifacts


class _RunContext:
    """Installs constraints + usage/budget ledger for one run and finalizes artifacts."""

    def __init__(
        self,
        *,
        claim: str,
        claim_id: str | int | None,
        claim_date: str | None,
        blocked_domains: tuple[str, ...],
        knowledge_store_dir: str | None,
        token_budget: int | None,
        retry_budget: int | None,
        strategy: str,
        stage_models: dict[str, str],
        stage_efforts: dict[str, str],
    ) -> None:
        normalized_claim = claim.strip()
        if not normalized_claim:
            raise ValueError("Claim is empty.")
        self.normalized_claim = normalized_claim
        started = datetime.now(timezone.utc)
        normalized_date = normalize_claim_date(claim_date)
        if claim_date is not None and str(claim_date).strip() and normalized_date is None:
            # A date that does not parse would silently switch off every
            # date-based leakage control (search freshness, source filter).
            # Refuse the run instead.
            raise ValueError(
                f"Unparseable claim date {claim_date!r}: expected YYYY-MM-DD or D-M-YYYY. "
                "The evidence cutoff cannot be applied."
            )
        self.constraints = ResearchConstraints(
            claim_id=str(claim_id) if claim_id is not None else None,
            claim_date=normalized_date,
            blocked_domains=list(blocked_domains),
            knowledge_store_dir=knowledge_store_dir,
            reference_date=normalized_date or started.date().isoformat(),
        )
        self.artifacts = RunArtifacts(
            claim=claim,
            normalized_claim=normalized_claim,
            claim_id=self.constraints.claim_id,
            constraints=self.constraints,
            started_at=started.isoformat(),
            strategy=strategy,
            stage_models=dict(stage_models),
            stage_efforts=dict(stage_efforts),
        )
        self._token_budget = token_budget
        self._retry_budget = retry_budget
        self.usage_log: UsageLog | None = None

    def __enter__(self) -> "_RunContext":
        self.usage_log, self._usage_token = activate_usage_log(
            token_budget=self._token_budget, retry_budget=self._retry_budget
        )
        self._constraints_token = activate_constraints(self.constraints)
        self._start = time.monotonic()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        deactivate_constraints(self._constraints_token)
        deactivate_usage_log(self._usage_token)
        assert self.usage_log is not None
        self.artifacts.duration_seconds = time.monotonic() - self._start
        self.artifacts.usage_events = self.usage_log.events
        self.artifacts.usage_summary = summarize_usage(self.usage_log.events)
        self.artifacts.budget = self.usage_log.budget_status()
        receipts = [record for check in self.artifacts.research_checks for record in check.retrieval]
        if receipts:
            self.artifacts.retrieval_summary = summarize_retrieval(receipts)


@dataclass(frozen=True, kw_only=True)
class _RunSettings:
    """Run-level controls shared by both strategies, and the run lifecycle around them."""
    citation_stage: CitationCheckStage | None = None
    artifact_repository: RunArtifactRepository | None = None
    blocked_domains: tuple[str, ...] = ()
    knowledge_store_dir: str | None = None
    token_budget: int | None = None
    retry_budget: int | None = None
    stage_models: dict[str, str] = field(default_factory=dict)
    stage_efforts: dict[str, str] = field(default_factory=dict)

    def _open_run(
        self, claim: str, *, claim_id: str | int | None, claim_date: str | None, strategy: str
    ) -> _RunContext:
        return _RunContext(
            claim=claim,
            claim_id=claim_id,
            claim_date=claim_date,
            blocked_domains=self.blocked_domains,
            knowledge_store_dir=self.knowledge_store_dir,
            token_budget=self.token_budget,
            retry_budget=self.retry_budget,
            strategy=strategy,
            stage_models=self.stage_models,
            stage_efforts=self.stage_efforts,
        )

    async def _finish_run(
        self,
        run: _RunContext,
        *,
        plan: InvestigationPlan,
        findings: list[AspectFinding],
        report: FactCheckReport,
        progress_callback: ProgressCallback | None,
    ) -> FactCheckRun:
        artifacts = run.artifacts
        if self.artifact_repository is not None:
            self.artifact_repository.save(artifacts)
        await emit_progress(
            progress_callback,
            "run_completed",
            {
                "claim": run.normalized_claim,
                "verdict": report.verdict.value,
                "verdict_confidence": report.verdict_confidence,
                "budget": artifacts.budget.model_dump() if artifacts.budget else None,
            },
        )
        return FactCheckRun(
            claim=run.normalized_claim, plan=plan, findings=findings, report=report, artifacts=artifacts
        )


@dataclass(frozen=True, kw_only=True)
class FactCheckService(_RunSettings):
    """High-level orchestrator that wires planning, research, review, and judging."""
    plan_stage: PlanStage
    research_stage: ResearchStage
    judge_stage: JudgeStage
    review_stage: ReviewStage | None = None
    max_feedback_rounds: int = 0

    async def check_claim(
        self,
        claim: str,
        progress_callback: ProgressCallback | None = None,
        *,
        claim_id: str | int | None = None,
        claim_date: str | None = None,
    ) -> FactCheckRun:
        """Execute a full fact-check run and emit progress events across stages.

        claim_date (any supported format) caps evidence recency for search
        backends that honor it; claim_id selects the offline knowledge store
        entry when the knowledge_store search provider is active.
        """
        run = self._open_run(claim, claim_id=claim_id, claim_date=claim_date, strategy="pipeline")
        artifacts = run.artifacts
        normalized_claim = run.normalized_claim
        with run:
            await emit_progress(progress_callback, "run_started", {"claim": normalized_claim})
            plan = await self.plan_stage.execute(
                claim=normalized_claim,
                artifacts=artifacts,
                progress_callback=progress_callback,
            )
            findings = await self.research_stage.execute(
                claim=normalized_claim,
                plan=plan,
                artifacts=artifacts,
                progress_callback=progress_callback,
            )
            plan, findings = await self._run_feedback_loop(
                claim=normalized_claim,
                plan=plan,
                findings=findings,
                artifacts=artifacts,
                usage_log=run.usage_log,
                progress_callback=progress_callback,
            )
            report = await self.judge_stage.execute(
                claim=normalized_claim,
                plan=plan,
                findings=findings,
                artifacts=artifacts,
                progress_callback=progress_callback,
            )
            if self.citation_stage is not None:
                report = await self.citation_stage.execute(report, artifacts, progress_callback=progress_callback)

        return await self._finish_run(
            run, plan=plan, findings=findings, report=report, progress_callback=progress_callback
        )

    async def _run_feedback_loop(
        self,
        *,
        claim: str,
        plan: InvestigationPlan,
        findings: list[AspectFinding],
        artifacts: RunArtifacts,
        usage_log: UsageLog | None,
        progress_callback: ProgressCallback | None,
    ) -> tuple[InvestigationPlan, list[AspectFinding]]:
        if self.review_stage is None or self.max_feedback_rounds <= 0:
            return plan, findings

        for round_index in range(1, self.max_feedback_rounds + 1):
            if usage_log is not None and usage_log.token_budget_exhausted():
                usage_log.record_skipped_stage(f"review_round_{round_index}")
                await emit_progress(
                    progress_callback,
                    "feedback_round_skipped",
                    {"round_index": round_index, "reason": "token budget exhausted"},
                )
                break

            decision = await self.review_stage.execute(
                claim=claim,
                plan=plan,
                findings=findings,
                artifacts=artifacts,
                round_index=round_index,
                progress_callback=progress_callback,
            )
            if decision.action != ReviewAction.FOLLOW_UP:
                break

            # ReviewStage keeps only retry ids of planned checks and gives
            # follow-up checks aspect ids the plan does not use yet.
            follow_up_plan = InvestigationPlan(
                claim=claim,
                checks=[
                    *(check for check in plan.checks if check.aspect_id in decision.retry_aspect_ids),
                    *decision.follow_up_checks,
                ],
                assumptions=[],
            )
            if artifacts.review_rounds:
                artifacts.review_rounds[-1].follow_up_plan = follow_up_plan
            await emit_progress(
                progress_callback,
                "feedback_round_started",
                {
                    "round_index": round_index,
                    "check_count": len(follow_up_plan.checks),
                    "gaps": decision.gaps,
                },
            )
            new_findings = await self.research_stage.execute(
                claim=claim,
                plan=follow_up_plan,
                artifacts=artifacts,
                progress_callback=progress_callback,
            )
            plan = plan.model_copy(update={"checks": [*plan.checks, *decision.follow_up_checks]})
            findings = self._merge_findings(findings, new_findings)
            await emit_progress(
                progress_callback,
                "feedback_round_completed",
                {
                    "round_index": round_index,
                    "finding_count": len(findings),
                },
            )

        return plan, findings

    @staticmethod
    def _merge_findings(
        current_findings: list[AspectFinding],
        new_findings: list[AspectFinding],
    ) -> list[AspectFinding]:
        """Merge follow-up findings; a retry never overwrites a completed finding with a failed one."""
        merged = {finding.aspect_id: finding for finding in current_findings}
        for finding in new_findings:
            previous = merged.get(finding.aspect_id)
            if previous is None or finding.is_observation or not previous.is_observation:
                merged[finding.aspect_id] = finding
        return list(merged.values())


@dataclass(frozen=True, kw_only=True)
class SingleAgentFactCheckService(_RunSettings):
    """No-harness baseline: one agent, one prompt, same output contract and controls.

    Used to measure what the plan/research/review/judge decomposition buys
    over a single agent with the same model, search tool, and token budget.
    Retrieval constraints, source tiering, grounding, budget accounting, and
    the citation check apply identically so only the orchestration differs.
    ``search_cap`` is the hard search limit for the whole run (the pipeline's
    per-check cap times ``max_checks``).
    """
    checker: SingleAgentChecker
    max_checks: int = 4
    search_cap: int | None = None

    async def check_claim(
        self,
        claim: str,
        progress_callback: ProgressCallback | None = None,
        *,
        claim_id: str | int | None = None,
        claim_date: str | None = None,
    ) -> FactCheckRun:
        run = self._open_run(claim, claim_id=claim_id, claim_date=claim_date, strategy="single_agent")
        artifacts = run.artifacts
        normalized_claim = run.normalized_claim
        with run:
            await emit_progress(progress_callback, "run_started", {"claim": normalized_claim})
            await emit_progress(progress_callback, "single_agent_started", {"claim": normalized_claim})
            journal = RetrievalJournal(search_cap=self.search_cap)
            journal_token = activate_journal(journal)
            try:
                report_raw = await self.checker.check(claim=normalized_claim, max_checks=self.max_checks)
            finally:
                deactivate_journal(journal_token)
            artifacts.report_raw = report_raw

            checks: list[VerificationCheck] = []
            findings: list[AspectFinding] = []
            for index, finding in enumerate(report_raw.findings[: self.max_checks], start=1):
                aspect_id = finding.aspect_id.strip() or f"aspect_{index}"
                check = VerificationCheck(
                    aspect_id=aspect_id,
                    question=finding.question or f"Aspect {index} of the claim",
                    rationale="Aspect identified by the single-agent baseline.",
                    search_queries=[],
                )
                artifact = artifacts.get_or_create_check(check)
                artifact.attempts = 1
                grounded = ground_finding(finding.model_copy(update={"aspect_id": aspect_id}), journal)
                filtered = _apply_source_constraints(grounded, artifact)
                tiered = filtered.model_copy(update={"sources": assign_source_tiers(filtered.sources)})
                artifact.finding = tiered
                checks.append(check)
                findings.append(tiered)

            plan = InvestigationPlan(claim=normalized_claim, checks=checks or [direct_check(normalized_claim)])
            artifacts.plan_raw = plan
            artifacts.plan_normalized = plan

            report_sources_artifact = artifacts.get_or_create_check(
                VerificationCheck(
                    aspect_id="report_sources",
                    question="Sources cited at report level",
                    rationale="Report-level source list of the single-agent baseline.",
                    search_queries=[],
                )
            )
            # The run has one journal; its receipts live on the report-level artifact.
            report_sources_artifact.retrieval.extend(journal.records)
            report_level = _apply_source_constraints(
                AspectFinding(
                    aspect_id="report_sources",
                    question="Sources cited at report level",
                    signal=EvidenceSignal.INSUFFICIENT,
                    summary="",
                    confidence=0.0,
                    sources=ground_sources(report_raw.sources, journal),
                ),
                report_sources_artifact,
            )
            report = report_raw.model_copy(
                update={
                    "claim": normalized_claim,
                    "findings": findings,
                    "sources": assign_source_tiers(merge_sources(report_level.sources, findings)),
                    "key_points": normalize_text_list(report_raw.key_points, max_items=12),
                    "evidence_gaps": normalize_text_list(report_raw.evidence_gaps, max_items=12),
                }
            )
            artifacts.report_final = report
            await emit_progress(
                progress_callback,
                "judging_completed",
                {
                    "verdict": report.verdict.value,
                    "verdict_confidence": report.verdict_confidence,
                    "source_count": len(report.sources),
                },
            )
            if self.citation_stage is not None:
                report = await self.citation_stage.execute(report, artifacts, progress_callback=progress_callback)

        return await self._finish_run(
            run, plan=plan, findings=findings, report=report, progress_callback=progress_callback
        )


@dataclass(frozen=True)
class ClaimExtractionService:
    """Service wrapper for extracting check-worthy claims from input text."""
    extraction_stage: ClaimExtractionStage

    async def extract_claims(self, input_text: str) -> ClaimExtractionResult:
        """Run extraction stage and return structured claim candidates."""
        return await self.extraction_stage.execute(input_text)
