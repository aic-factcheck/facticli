from __future__ import annotations

from typing import Any

from facticli.core.constraints import ResearchConstraints
from facticli.core.contracts import (
    AspectFinding,
    EvidenceSignal,
    FindingStatus,
    InvestigationPlan,
    VerificationCheck,
)
from facticli.core.normalize import normalize_source_url
from facticli.core.usage import UsageLog

# Stage inputs are built here as plain dicts so they can be unit-tested and
# diffed without a model call. The shape follows two 2026 findings: LLM
# judges commit early and cherry-pick under mixed evidence (so evidence is
# grouped by signal with counts and the judge must argue the other side), and
# orchestrators introduce most citation errors when they only see synthesized
# notes (so verbatim snippets travel alongside summaries, addressed by stable
# source ids the justification can cite).

FAILED_CHECK_NOTICE = (
    "The checks listed under failed_checks did not run to completion (tool or model failure, "
    "timeout, or budget exhaustion). They are missing observations, not evidence of absence. "
    "Do not infer anything about the claim from them; mention them in evidence_gaps."
)


def constraints_payload(constraints: ResearchConstraints | None) -> dict[str, Any] | None:
    """Render active retrieval constraints for a research-style prompt."""
    if constraints is None or not (constraints.claim_date or constraints.blocked_domains):
        return None
    return {
        "evidence_cutoff_date": constraints.claim_date,
        "blocked_domains": list(constraints.blocked_domains),
        "instruction": (
            "Only use evidence published on or before the cutoff date (if set). "
            "Never cite or rely on sources from the blocked domains; these are "
            "fact-checking sites excluded to keep the verdict independent."
        ),
    }


def budget_payload(log: UsageLog | None) -> dict[str, Any] | None:
    """Codex-style budget reminder for a stage prompt; None when unbudgeted."""
    if log is None or log.token_budget is None:
        return None
    remaining = log.tokens_remaining() or 0
    return {
        "shared_token_budget": log.token_budget,
        "tokens_remaining": remaining,
        "instruction": (
            "This run shares one token budget across all stages. Be economical: prefer few, "
            "well-targeted searches and stop as soon as the acceptance criteria are met or "
            "clearly unreachable."
        ),
    }


def research_payload(
    *,
    claim: str,
    check: VerificationCheck,
    search_provider: str,
    constraints: ResearchConstraints | None,
    usage_log: UsageLog | None,
    tool_call_budget: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "claim": claim,
        "check": check.model_dump(),
        "requirements": {
            "min_sources": 2,
            "must_use_search_tool": True,
            "tool_call_budget": tool_call_budget,
            "preferred_provider": search_provider,
            "snippets_verbatim": True,
            "do_not_adjudicate_whole_claim": True,
        },
    }
    constraints_block = constraints_payload(constraints)
    if constraints_block:
        payload["constraints"] = constraints_block
    budget_block = budget_payload(usage_log)
    if budget_block:
        payload["budget"] = budget_block
    return payload


def _finding_status(finding: AspectFinding) -> str:
    return finding.status.value if isinstance(finding.status, FindingStatus) else str(finding.status)


def review_payload(
    *,
    claim: str,
    plan: InvestigationPlan,
    findings: list[AspectFinding],
    max_follow_up_checks: int,
    round_index: int,
) -> dict[str, Any]:
    """Pair each check with its finding so the reviewer grades against criteria."""
    findings_by_aspect = {finding.aspect_id: finding for finding in findings}
    graded: list[dict[str, Any]] = []
    for check in plan.checks:
        finding = findings_by_aspect.get(check.aspect_id)
        entry: dict[str, Any] = {
            "aspect_id": check.aspect_id,
            "question": check.question,
            "acceptance_criteria": check.acceptance_criteria,
        }
        if finding is None:
            entry["finding"] = None
            entry["status"] = "missing"
        else:
            entry["status"] = _finding_status(finding)
            entry["finding"] = {
                "signal": finding.signal.value,
                "confidence": finding.confidence,
                "summary": finding.summary,
                "caveats": finding.caveats,
                "failure_reason": finding.failure_reason,
                "sources": [
                    {
                        "url": source.url,
                        "tier": source.tier.value if source.tier else None,
                        "published_at": source.published_at,
                        "snippet": source.snippet,
                    }
                    for source in finding.sources
                ],
            }
        graded.append(entry)
    return {
        "claim": claim,
        "assumptions": plan.assumptions,
        "round_index": round_index,
        "limits": {"max_follow_up_checks": max_follow_up_checks},
        "checks": graded,
    }


def index_sources(findings: list[AspectFinding]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Assign stable ids (S1..Sn) to deduplicated sources across findings.

    Returns the source table and a mapping ``normalized_url -> source_id`` so
    findings can reference sources by id.
    """
    table: list[dict[str, Any]] = []
    ids_by_url: dict[str, str] = {}
    cited_by: dict[str, list[str]] = {}
    for finding in findings:
        for source in finding.sources:
            key = normalize_source_url(source.url)
            if not key:
                continue
            if key not in ids_by_url:
                source_id = f"S{len(table) + 1}"
                ids_by_url[key] = source_id
                cited_by[source_id] = []
                table.append(
                    {
                        "id": source_id,
                        "title": source.title,
                        "url": source.url,
                        "publisher": source.publisher,
                        "published_at": source.published_at,
                        "tier": source.tier.value if source.tier else None,
                        "snippet": source.snippet,
                        "cited_by": cited_by[source_id],
                    }
                )
            source_id = ids_by_url[key]
            if finding.aspect_id not in cited_by[source_id]:
                cited_by[source_id].append(finding.aspect_id)
    return table, ids_by_url


def judge_payload(*, claim: str, plan: InvestigationPlan, findings: list[AspectFinding]) -> dict[str, Any]:
    """Judge input grouped by signal, with a shared source table and failed-check notice."""
    checks_by_aspect = {check.aspect_id: check for check in plan.checks}
    source_table, ids_by_url = index_sources(findings)

    grouped: dict[str, list[dict[str, Any]]] = {signal.value: [] for signal in EvidenceSignal}
    failed_checks: list[dict[str, Any]] = []
    for finding in findings:
        if finding.status != FindingStatus.COMPLETED:
            failed_checks.append(
                {
                    "aspect_id": finding.aspect_id,
                    "question": finding.question,
                    "status": _finding_status(finding),
                    "failure_reason": finding.failure_reason,
                }
            )
            continue
        check = checks_by_aspect.get(finding.aspect_id)
        source_ids = []
        for source in finding.sources:
            source_id = ids_by_url.get(normalize_source_url(source.url))
            if source_id and source_id not in source_ids:
                source_ids.append(source_id)
        grouped[finding.signal.value].append(
            {
                "aspect_id": finding.aspect_id,
                "question": finding.question,
                "acceptance_criteria": check.acceptance_criteria if check else [],
                "summary": finding.summary,
                "confidence": finding.confidence,
                "caveats": finding.caveats,
                "source_ids": source_ids,
            }
        )

    completed = [finding for finding in findings if finding.status == FindingStatus.COMPLETED]
    payload: dict[str, Any] = {
        "claim": claim,
        "assumptions": plan.assumptions,
        "evidence_overview": {
            "checks_planned": len(plan.checks),
            "checks_completed": len(completed),
            "checks_failed": len(failed_checks),
            "signal_counts": {signal: len(items) for signal, items in grouped.items()},
            "source_tier_counts": _tier_counts(source_table),
        },
        "findings_by_signal": grouped,
        "sources": source_table,
        "citation_rule": (
            "Reference sources in justification and key_points by their id in square brackets, "
            "e.g. [S2]. Sentences without a source id must be explicitly marked as inference."
        ),
    }
    if failed_checks:
        payload["failed_checks"] = failed_checks
        payload["failed_checks_notice"] = FAILED_CHECK_NOTICE
    return payload


def _tier_counts(source_table: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in source_table:
        key = row.get("tier") or "untiered"
        counts[key] = counts.get(key, 0) + 1
    return counts


def single_agent_payload(
    *,
    claim: str,
    max_checks: int,
    search_provider: str,
    constraints: ResearchConstraints | None,
    usage_log: UsageLog | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "claim": claim,
        "requirements": {
            "max_aspects": max_checks,
            "min_sources": 2,
            "must_use_search_tool": True,
            "preferred_provider": search_provider,
            "snippets_verbatim": True,
        },
    }
    constraints_block = constraints_payload(constraints)
    if constraints_block:
        payload["constraints"] = constraints_block
    budget_block = budget_payload(usage_log)
    if budget_block:
        payload["budget"] = budget_block
    return payload

