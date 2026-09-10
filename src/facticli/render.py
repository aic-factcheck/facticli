from __future__ import annotations

from .application.services import FactCheckRun
from .core.contracts import FindingStatus, SourceEvidence


def _source_tag(source: SourceEvidence) -> str:
    tags: list[str] = []
    if source.tier is not None:
        tags.append(source.tier.value)
    if source.published_at:
        tags.append(source.published_at)
    if source.url_status is not None:
        tags.append(source.url_status.value)
    return f" ({', '.join(tags)})" if tags else ""


def format_run_text(run: FactCheckRun, show_plan: bool = False, show_usage: bool = False) -> str:
    report = run.report

    lines: list[str] = []
    lines.append("Claim")
    lines.append(f"  {run.claim}")
    lines.append("")
    lines.append("Verdict")
    lines.append(f"  {report.verdict.value} (confidence: {report.verdict_confidence:.2f})")
    lines.append("")
    lines.append("Justification")
    lines.append(f"  {report.justification}")

    if report.key_points:
        lines.append("")
        lines.append("Key Points")
        for point in report.key_points:
            lines.append(f"  - {point}")

    if report.counter_argument:
        lines.append("")
        lines.append("Counter-argument Considered")
        lines.append(f"  {report.counter_argument}")

    if report.evidence_gaps:
        lines.append("")
        lines.append("Evidence Gaps")
        for gap in report.evidence_gaps:
            lines.append(f"  - {gap}")

    if show_plan:
        lines.append("")
        lines.append("Plan")
        for check in run.plan.checks:
            lines.append(f"  - [{check.aspect_id}] {check.question}")
            lines.append(f"    rationale: {check.rationale}")
            if check.search_queries:
                lines.append(f"    queries: {', '.join(check.search_queries)}")
            if check.acceptance_criteria:
                lines.append(f"    done when: {'; '.join(check.acceptance_criteria)}")

    lines.append("")
    lines.append("Findings")
    if not report.findings:
        lines.append("  - no findings returned")
    for finding in report.findings:
        if finding.status != FindingStatus.COMPLETED:
            label = "skipped (budget)" if finding.status == FindingStatus.BUDGET_EXHAUSTED else "failed"
            lines.append(f"  - [{finding.aspect_id}] {label} | not an observation")
            lines.append(f"    question: {finding.question}")
            if finding.failure_reason:
                lines.append(f"    reason: {finding.failure_reason}")
            continue
        lines.append(
            f"  - [{finding.aspect_id}] {finding.signal.value} | confidence {finding.confidence:.2f}"
        )
        lines.append(f"    question: {finding.question}")
        lines.append(f"    summary: {finding.summary}")
        if finding.caveats:
            lines.append(f"    caveats: {'; '.join(finding.caveats)}")

    lines.append("")
    lines.append("Sources")
    if not report.sources:
        lines.append("  - no sources returned")
    for idx, source in enumerate(report.sources, start=1):
        lines.append(f"  [S{idx}] {source.title}{_source_tag(source)}")
        lines.append(f"      {source.url}")
        if source.url_status is not None and source.url_status.value == "archived" and source.archived_url:
            lines.append(f"      archived: {source.archived_url}")
        lines.append(f"      {source.snippet}")

    if show_usage:
        lines.extend(_format_usage(run))

    return "\n".join(lines)


def _format_usage(run: FactCheckRun) -> list[str]:
    artifacts = run.artifacts
    lines = ["", "Run Stats"]
    lines.append(f"  strategy: {artifacts.strategy}")
    if artifacts.stage_models:
        distinct = sorted(set(artifacts.stage_models.values()))
        if len(distinct) == 1:
            lines.append(f"  model: {distinct[0]}")
        else:
            lines.append("  models: " + ", ".join(f"{stage}={model}" for stage, model in artifacts.stage_models.items()))
    if artifacts.duration_seconds is not None:
        lines.append(f"  duration: {artifacts.duration_seconds:.1f}s")
    summary = artifacts.usage_summary
    if summary is not None:
        lines.append(
            "  tokens: "
            f"{summary.total_tokens} total ({summary.input_tokens} in, {summary.output_tokens} out, "
            f"{summary.cached_input_tokens} cached, {summary.reasoning_tokens} reasoning) "
            f"over {summary.requests} request(s), cache hit rate {summary.cache_hit_rate:.0%}"
        )
        for stage, bucket in summary.per_stage.items():
            lines.append(
                f"    {stage}: {int(bucket['total_tokens'])} tokens, {int(bucket['calls'])} call(s), "
                f"{bucket['llm_seconds']:.1f}s"
            )
    budget = artifacts.budget
    if budget is not None and (budget.token_budget is not None or budget.retries_used):
        parts = []
        if budget.token_budget is not None:
            parts.append(f"token budget {budget.tokens_used}/{budget.token_budget}" + (" (exhausted)" if budget.token_budget_exhausted else ""))
        if budget.retry_budget is not None or budget.retries_used:
            cap = budget.retry_budget if budget.retry_budget is not None else "unlimited"
            parts.append(f"retries {budget.retries_used}/{cap}")
        lines.append("  budget: " + "; ".join(parts))
        if budget.skipped_stages:
            lines.append(f"  skipped: {', '.join(budget.skipped_stages)}")
    if artifacts.citation_check:
        lines.append("  citations: " + ", ".join(f"{k}={v}" for k, v in sorted(artifacts.citation_check.items())))
    return lines
