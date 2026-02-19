from __future__ import annotations

from .orchestrator import FactCheckRun


def format_run_text(run: FactCheckRun, show_plan: bool = False) -> str:
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

    if show_plan:
        lines.append("")
        lines.append("Plan")
        for check in run.plan.checks:
            lines.append(f"  - [{check.aspect_id}] {check.question}")
            lines.append(f"    rationale: {check.rationale}")
            if check.search_queries:
                lines.append(f"    queries: {', '.join(check.search_queries)}")

    lines.append("")
    lines.append("Findings")
    if not report.findings:
        lines.append("  - no findings returned")
    for finding in report.findings:
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
        lines.append(f"  [{idx}] {source.title}")
        lines.append(f"      {source.url}")
        lines.append(f"      {source.snippet}")

    return "\n".join(lines)

