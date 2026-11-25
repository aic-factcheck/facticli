"""Fact-check pipeline (plan -> parallel research -> judge) and claim extraction.

Each stage is one ``StageRunner`` call. Everything around the calls is
deterministic harness code: plan normalization, bounded parallel research with
retries and a timeout, provenance (grounding) and authority tiers on sources,
the leakage filter, and the run artifacts.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from facticli.config import SEARCH_CAP_FACTOR, ClaimExtractionRuntimeConfig, FactCheckRuntimeConfig
from facticli.core.artifacts import ResearchCheckArtifact, RunArtifacts
from facticli.core.constraints import (
    FACT_CHECK_DOMAINS,
    ResearchConstraints,
    activate_constraints,
    deactivate_constraints,
    get_constraints,
    is_blocked_url,
    normalize_claim_date,
    violates_date_cutoff,
)
from facticli.core.contracts import (
    AspectFinding,
    ClaimExtractionResult,
    EvidenceSignal,
    FactCheckReport,
    FindingStatus,
    InvestigationPlan,
    SourceEvidence,
    SourceGrounding,
    StopReason,
    VerificationCheck,
)
from facticli.core.errors import ErrorKind, classify_exception, describe_exception, is_fatal
from facticli.core.grounding import ground_finding, unresolved_citation_ids
from facticli.core.normalize import normalize_plan_checks, normalize_source_url, normalize_text_list
from facticli.core.retrieval import RetrievalJournal, activate_journal, deactivate_journal, get_journal, summarize_retrieval
from facticli.core.search_cache import SearchCache
from facticli.core.source_quality import assign_source_tiers
from facticli.core.usage import activate_usage_log, deactivate_usage_log, summarize_usage
from facticli.llm import StageRunner, build_search_tools, configure_client, resolve_endpoint
from facticli.payloads import dated, extraction_payload, judge_payload, plan_payload, research_payload

# Error kinds that re-running the same request cannot fix; the check fails
# without further attempts. Fatal kinds (auth, quota, not found) do not get
# this far: they fail every check the same way, so they abort the run.
_NO_RETRY_KINDS: frozenset[ErrorKind] = frozenset(
    {ErrorKind.FORBIDDEN, ErrorKind.BAD_REQUEST, ErrorKind.CONTEXT_OVERFLOW, ErrorKind.REFUSAL}
)

Progress = Callable[[str], None]


class Stage(Protocol):
    """What the pipeline needs from a model stage (``StageRunner`` or a test fake)."""

    async def run(self, payload: dict[str, Any]) -> tuple[Any, bool]: ...


@dataclass
class FactCheckRun:
    """Everything one fact-check run produced."""
    claim: str
    plan: InvestigationPlan
    findings: list[AspectFinding]
    report: FactCheckReport
    artifacts: RunArtifacts


# --- deterministic helpers ---------------------------------------------------


def direct_check(claim: str) -> VerificationCheck:
    """Fallback single check used when the planner returns nothing usable."""
    return VerificationCheck(
        aspect_id="claim_direct_check",
        question=f"Is this claim accurate: {claim}",
        rationale="Fallback direct verification when planning fails.",
        search_queries=[claim],
        acceptance_criteria=["The central assertion is confirmed or contradicted by at least two independent sources."],
    )


def failed_finding(check: VerificationCheck, *, reason: str, attempts: int) -> AspectFinding:
    """Placeholder finding for a check the harness could not complete: a missing observation."""
    return AspectFinding(
        aspect_id=check.aspect_id,
        question=check.question,
        signal=EvidenceSignal.INSUFFICIENT,
        summary=f"Research subroutine failed after {attempts} attempt(s): {reason}",
        confidence=0.0,
        sources=[],
        caveats=["Research failed; this is a missing observation, not evidence of absence."],
        status=FindingStatus.FAILED,
        failure_reason=reason,
    )


def apply_source_constraints(finding: AspectFinding, artifact: ResearchCheckArtifact) -> AspectFinding:
    """Drop sources from blocked domains or after the claim date; record them in the artifact.

    The artifact keeps the unfiltered finding and the removed sources, so
    leakage audits do not have to re-derive the filtering decision.
    """
    constraints = get_constraints()
    if constraints is None or (not constraints.blocked_domains and not constraints.claim_date):
        return finding

    kept: list[SourceEvidence] = []
    removed: list[SourceEvidence] = []
    for source in finding.sources:
        if is_blocked_url(source.url, constraints.blocked_domains) or violates_date_cutoff(
            source.published_at, constraints.claim_date
        ):
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


def merge_sources(report_sources: list[SourceEvidence], findings: list[AspectFinding]) -> list[SourceEvidence]:
    """Deduplicate finding and report sources by normalized URL, preferring the better copy.

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


def resolve_blocked_domains(config: FactCheckRuntimeConfig) -> tuple[str, ...]:
    """Fact-checker blocklist (when enabled) plus extra domains, lowercased and deduplicated."""
    domains = [*(FACT_CHECK_DOMAINS if config.block_fact_checkers else ()), *config.blocked_domains]
    unique: dict[str, None] = {}
    for domain in domains:
        normalized = domain.strip().lower()
        if normalized:
            unique.setdefault(normalized)
    return tuple(unique)


def save_artifacts(artifacts: RunArtifacts, output_dir: str) -> Path:
    """Write one run's artifacts as ``run_<utc stamp>_<claim id or hash>_<random>.json``."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    claim_key = (
        str(artifacts.claim_id)
        if artifacts.claim_id is not None
        else hashlib.sha1(artifacts.normalized_claim.encode("utf-8")).hexdigest()[:10]
    )
    path = directory / f"run_{stamp}_{claim_key}_{uuid.uuid4().hex[:6]}.json"
    path.write_text(artifacts.model_dump_json(indent=2), encoding="utf-8")
    return path


def _truncate(value: str, max_length: int = 140) -> str:
    normalized = " ".join(value.split())
    return normalized if len(normalized) <= max_length else normalized[: max_length - 3] + "..."


# --- fact-check pipeline -----------------------------------------------------


class FactChecker:
    """Plan -> bounded parallel research -> judge, for one claim at a time."""

    def __init__(
        self,
        config: FactCheckRuntimeConfig,
        *,
        planner: Stage,
        researcher: Stage,
        judge: Stage,
        stage_models: dict[str, str] | None = None,
        artifacts_dir: str | None = None,
    ) -> None:
        self.config = config
        self.planner = planner
        self.researcher = researcher
        self.judge = judge
        self.stage_models = dict(stage_models or {})
        self.artifacts_dir = artifacts_dir
        self.blocked_domains = resolve_blocked_domains(config)
        # Hard per-check search cap, above the budget the researcher is told about.
        self.search_cap = SEARCH_CAP_FACTOR * config.tool_call_budget

    async def check_claim(
        self,
        claim: str,
        progress: Progress | None = None,
        *,
        claim_id: str | int | None = None,
        claim_date: str | None = None,
    ) -> FactCheckRun:
        """Run the pipeline on one claim.

        ``claim_date`` (YYYY-MM-DD or D-M-YYYY) is the evidence cutoff; a value
        that does not parse raises instead of running without the cutoff.
        ``claim_id`` selects the knowledge-store entry for that search provider.
        """
        say = progress or (lambda _line: None)
        normalized_claim = claim.strip()
        if not normalized_claim:
            raise ValueError("Claim is empty.")
        started = datetime.now(timezone.utc)
        normalized_date = normalize_claim_date(claim_date)
        if claim_date is not None and str(claim_date).strip() and normalized_date is None:
            raise ValueError(
                f"Unparseable claim date {claim_date!r}: expected YYYY-MM-DD or D-M-YYYY. "
                "The evidence cutoff cannot be applied."
            )
        constraints = ResearchConstraints(
            claim_id=str(claim_id) if claim_id is not None else None,
            claim_date=normalized_date,
            blocked_domains=list(self.blocked_domains),
            knowledge_store_dir=self.config.knowledge_store_dir,
            reference_date=normalized_date or started.date().isoformat(),
        )
        artifacts = RunArtifacts(
            claim=claim,
            normalized_claim=normalized_claim,
            claim_id=constraints.claim_id,
            constraints=constraints,
            started_at=started.isoformat(),
            stage_models=self.stage_models,
            stage_efforts=dict(self.config.stage_efforts),
        )

        usage_log, usage_token = activate_usage_log()
        constraints_token = activate_constraints(constraints)
        start = time.monotonic()
        try:
            say(f"[progress] Starting fact-check: {normalized_claim}")
            plan = await self._plan(normalized_claim, artifacts, say)
            findings = await self._research(normalized_claim, plan, artifacts, say)
            report = await self._judge(normalized_claim, plan, findings, artifacts, say)
        finally:
            deactivate_constraints(constraints_token)
            deactivate_usage_log(usage_token)
            artifacts.duration_seconds = time.monotonic() - start
            artifacts.usage_events = usage_log.events
            artifacts.usage_summary = summarize_usage(usage_log.events)
            receipts = [record for check in artifacts.research_checks for record in check.retrieval]
            if receipts:
                artifacts.retrieval_summary = summarize_retrieval(receipts)

        if self.artifacts_dir:
            save_artifacts(artifacts, self.artifacts_dir)
        say("[progress] Fact-check run completed.")
        return FactCheckRun(claim=normalized_claim, plan=plan, findings=findings, report=report, artifacts=artifacts)

    async def _plan(self, claim: str, artifacts: RunArtifacts, say: Progress) -> InvestigationPlan:
        say("[progress] Planning verification checks...")
        plan_raw, _ = await self.planner.run(plan_payload(claim=claim, max_checks=self.config.max_checks))
        artifacts.plan_raw = plan_raw
        checks = normalize_plan_checks(
            claim=claim,
            checks=plan_raw.checks,
            max_checks=self.config.max_checks,
            max_search_queries_per_check=self.config.max_search_queries_per_check,
        ) or [direct_check(claim)]
        plan = plan_raw.model_copy(update={"claim": claim, "checks": checks})
        artifacts.plan_normalized = plan
        say(f"[progress] Plan ready with {len(checks)} check(s):")
        for check in checks:
            say(f"  - [{check.aspect_id}] {check.question}")
        return plan

    async def _research(
        self, claim: str, plan: InvestigationPlan, artifacts: RunArtifacts, say: Progress
    ) -> list[AspectFinding]:
        """Ordered findings; a check that keeps failing becomes an explicit failed finding.

        An error no retry can fix (rejected key, exhausted quota, unknown
        model) is raised instead: it would fail every check the same way, and
        a verdict judged on none of them is worth nothing.
        """
        say(f"[progress] Running research for {len(plan.checks)} check(s)...")
        semaphore = asyncio.Semaphore(max(1, self.config.max_parallel_research))

        async def run_check(check: VerificationCheck) -> AspectFinding:
            async with semaphore:
                finding = await self.research_check(claim, check, artifacts)
            if finding.status == FindingStatus.COMPLETED:
                cut = f" | cut short: {finding.stop_reason.value}" if finding.stop_reason else ""
                say(f"[progress] [{finding.aspect_id}] {finding.signal.value} | confidence {finding.confidence:.2f}{cut}")
                say(f"           {_truncate(finding.summary)}")
            else:
                say(f"[progress] [{check.aspect_id}] {finding.status.value}: {finding.failure_reason or ''}")
            return finding

        tasks = [asyncio.create_task(run_check(check)) for check in plan.checks]
        try:
            return list(await asyncio.gather(*tasks))
        except BaseException:
            # A fatal error or a cancelled run must not leave sibling checks spending tokens.
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    async def research_check(self, claim: str, check: VerificationCheck, artifacts: RunArtifacts) -> AspectFinding:
        """Research one check with retries; each attempt runs under its own journal and search cap."""
        artifact = artifacts.get_or_create_check(check)
        timeout = self.config.effective_research_timeout_seconds or None
        last_error: Exception | None = None
        for _attempt in range(1 + max(0, self.config.research_retry_attempts)):
            artifact.attempts += 1
            journal = RetrievalJournal(search_cap=self.search_cap)
            journal_token = activate_journal(journal)
            try:
                task = self._research_once(claim, check)
                finding = await (asyncio.wait_for(task, timeout=timeout) if timeout else task)
                # Provenance first, on the sources exactly as the model returned them.
                finding = ground_finding(finding, journal)
                finding = finding.model_copy(update={"sources": assign_source_tiers(finding.sources)})
                artifact.finding = finding  # unfiltered, for leakage audits
                return apply_source_constraints(finding, artifact)
            except Exception as exc:
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
        finding = failed_finding(check, reason=describe_exception(last_error), attempts=artifact.attempts)
        artifact.finding = finding
        return finding

    async def _research_once(self, claim: str, check: VerificationCheck) -> AspectFinding:
        constraints = get_constraints()
        payload = research_payload(
            claim=claim,
            check=check,
            search_provider=self.config.search_provider,
            constraints=constraints,
            tool_call_budget=self.config.tool_call_budget,
        )
        finding, wrapped_up = await self.researcher.run(dated(payload, constraints))
        finding = finding.without_harness_fields()
        updates: dict[str, Any] = {}
        if finding.aspect_id != check.aspect_id:
            updates["aspect_id"] = check.aspect_id
        if not finding.question.strip():
            updates["question"] = check.question
        journal = get_journal()
        if wrapped_up:
            updates["stop_reason"] = StopReason.TURN_LIMIT
        elif journal is not None and journal.refusals:
            updates["stop_reason"] = StopReason.SEARCH_CAP
        return finding.model_copy(update=updates) if updates else finding

    async def _judge(
        self,
        claim: str,
        plan: InvestigationPlan,
        findings: list[AspectFinding],
        artifacts: RunArtifacts,
        say: Progress,
    ) -> FactCheckReport:
        """Judge the findings. The harness-held findings are authoritative in the final report."""
        say("[progress] Synthesizing final verdict...")
        payload = judge_payload(claim=claim, plan=plan, findings=findings)
        report_raw, _ = await self.judge.run(dated(payload, get_constraints()))
        report_raw = report_raw.without_harness_fields()
        artifacts.report_raw = report_raw

        # The judge has no search tool. A source it lists that no researcher
        # cited is a mis-copied or remembered URL, and is labelled as such.
        researched = {normalize_source_url(source.url) for finding in findings for source in finding.sources}
        judge_sources = [
            source.model_copy(update={"grounding": SourceGrounding.UNMATCHED})
            for source in report_raw.sources
            if normalize_source_url(source.url) not in researched
        ]
        report = report_raw.model_copy(
            update={
                "claim": claim,
                "findings": findings if findings else report_raw.findings,
                "sources": assign_source_tiers(merge_sources(judge_sources, findings)),
                "key_points": normalize_text_list(report_raw.key_points, max_items=12),
                "evidence_gaps": normalize_text_list(report_raw.evidence_gaps, max_items=12),
            }
        )
        artifacts.report_final = report
        # The source list starts with the judge's table S1..Sn in order, so the
        # ids in its prose index this list; ids beyond it point at nothing.
        artifacts.unresolved_citations = unresolved_citation_ids(
            [report.justification, *report.key_points, report.counter_argument],
            source_count=len(report.sources),
        )
        say(f"[progress] Verdict: {report.verdict.value} (confidence {report.verdict_confidence:.2f})")
        return report


def build_fact_checker(config: FactCheckRuntimeConfig, *, artifacts_dir: str | None = None) -> FactChecker:
    """Wire the three stages to the configured endpoint, models, and search tool."""
    if config.search_cache_dir and config.search_provider not in ("brave", "exa"):
        raise ValueError(
            "--search-cache-dir records Brave or Exa responses and needs --search-provider brave or exa: hosted OpenAI search "
            "runs inside the provider and the knowledge store is already local."
        )
    endpoint = resolve_endpoint(requested_model=config.model, base_url=config.base_url)
    configure_client(endpoint, max_retries=config.model_retry_attempts)
    models = {stage: config.model_for(stage, endpoint.model) for stage in ("plan", "research", "judge")}
    tools = build_search_tools(
        search_provider=config.search_provider,
        search_context_size=config.search_context_size,
        max_chars_per_field=config.max_chars_per_result_field,
        max_total_chars=config.max_tool_output_chars,
        search_results_per_query=config.search_results_per_query,
        cache=SearchCache(config.search_cache_dir) if config.search_cache_dir else None,
    )

    def runner(name: str, stage: str, output_type: type[Any], tools: list[Any] | None = None) -> StageRunner:
        return StageRunner(
            name=name,
            stage=stage,
            output_type=output_type,
            model=models[stage],
            max_turns=config.max_turns,
            reasoning_effort=config.effort_for(stage),
            tools=tools or (),
            service_tier=config.service_tier,
        )

    return FactChecker(
        config,
        planner=runner("claim_planner", "plan", InvestigationPlan),
        researcher=runner("check_researcher", "research", AspectFinding, tools),
        judge=runner("veracity_judge", "judge", FactCheckReport),
        stage_models=models,
        artifacts_dir=artifacts_dir,
    )


# --- claim extraction ----------------------------------------------------------


class ClaimExtractor:
    """Extracts check-worthy, decontextualized claims from free text."""

    def __init__(self, stage: Stage, *, max_claims: int) -> None:
        self.stage = stage
        self.max_claims = max_claims

    async def extract_claims(self, input_text: str) -> ClaimExtractionResult:
        normalized_text = input_text.strip()
        if not normalized_text:
            raise ValueError("Input text is empty.")
        extraction, _ = await self.stage.run(extraction_payload(input_text=normalized_text, max_claims=self.max_claims))
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


def build_claim_extractor(config: ClaimExtractionRuntimeConfig) -> ClaimExtractor:
    endpoint = resolve_endpoint(requested_model=config.model, base_url=config.base_url)
    configure_client(endpoint, max_retries=config.model_retry_attempts)
    stage = StageRunner(
        name="checkworthy_claim_extractor",
        stage="extract_claims",
        output_type=ClaimExtractionResult,
        model=config.model_for("extract_claims", endpoint.model),
        max_turns=config.max_turns,
        reasoning_effort=config.effort_for("extract_claims"),
        service_tier=config.service_tier,
    )
    return ClaimExtractor(stage, max_claims=config.max_claims)
