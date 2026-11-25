from __future__ import annotations

from .pipeline import FactCheckRun
from .core.contracts import FindingStatus, SourceEvidence, SourceGrounding, StopReason

# What the harness could trace back to search-tool output. The labels say
# where a source came from, not whether it is right.
_GROUNDING_LABELS = {
    SourceGrounding.QUOTE_MATCHED: "quote found in retrieved text",
    SourceGrounding.URL_MATCHED: "url retrieved, quote not matched",
    SourceGrounding.UNMATCHED: "NOT in search results",
}
_STOP_LABELS = {
    StopReason.TURN_LIMIT: "cut short at the turn limit",
    StopReason.SEARCH_CAP: "cut short at the search cap",
}


def _source_tag(source: SourceEvidence) -> str:
    tags: list[str] = []
    if source.tier is not None:
        tags.append(source.tier.value)
    if source.published_at:
        tags.append(source.published_at)
    if source.grounding is not None:
        tags.append(_GROUNDING_LABELS[source.grounding])
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
            lines.append(f"  - [{finding.aspect_id}] failed | not an observation")
            lines.append(f"    question: {finding.question}")
            if finding.failure_reason:
                lines.append(f"    reason: {finding.failure_reason}")
            continue
        stop = f" | {_STOP_LABELS[finding.stop_reason]}" if finding.stop_reason else ""
        lines.append(
            f"  - [{finding.aspect_id}] {finding.signal.value} | confidence {finding.confidence:.2f}{stop}"
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
        lines.append(f"      {source.snippet}")
    if run.artifacts.unresolved_citations:
        cited = ", ".join(f"S{number}" for number in run.artifacts.unresolved_citations)
        lines.append(f"  cited above but absent from this list: {cited}")

    if show_usage:
        lines.extend(_format_usage(run))

    return "\n".join(lines)


def _format_usage(run: FactCheckRun) -> list[str]:
    artifacts = run.artifacts
    lines = ["", "Run Stats"]
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
            f"{summary.cached_input_tokens} cached, {summary.cache_write_tokens} cache writes, "
            f"{summary.reasoning_tokens} reasoning) "
            f"over {summary.requests} request(s), cache hit rate {summary.cache_hit_rate:.0%}"
        )
        for stage, bucket in summary.per_stage.items():
            failed = int(bucket.get("failed_attempts", 0))
            failed_note = (
                f", {failed} failed attempt(s) costing {int(bucket.get('failed_attempt_tokens', 0))} tokens"
                if failed
                else ""
            )
            lines.append(
                f"    {stage}: {int(bucket['total_tokens'])} tokens, {int(bucket['calls'])} call(s), "
                f"{bucket['llm_seconds']:.1f}s{failed_note}"
            )
        if summary.wrap_up_calls:
            lines.append(f"  wrap-up calls at the turn limit: {summary.wrap_up_calls}")
    retrieval = artifacts.retrieval_summary
    if retrieval is not None:
        counts = {
            "search(es)": retrieval.searches,
            "page(s) opened": retrieval.pages_opened,
            "distinct URL(s)": retrieval.distinct_urls,
            "served from cache": retrieval.cache_hits,
            "refused at the cap": retrieval.refused,
            "failed": retrieval.errors,
        }
        lines.append("  retrieval: " + ", ".join(f"{count} {label}" for label, count in counts.items() if count))
    return lines
