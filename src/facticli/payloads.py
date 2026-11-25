from __future__ import annotations

from typing import Any

from facticli.core.constraints import ResearchConstraints
from facticli.core.contracts import (
    AspectFinding,
    EvidenceSignal,
    FindingStatus,
    InvestigationPlan,
    SourceEvidence,
    SourceGrounding,
    VerificationCheck,
)
from facticli.core.normalize import normalize_source_url

# Stage inputs are built here as plain dicts so they can be unit-tested and
# diffed without a model call. The shape follows two 2026 findings: LLM
# judges commit early and cherry-pick under mixed evidence (so evidence is
# grouped by signal with counts and the judge must argue the other side), and
# orchestrators introduce most citation errors when they only see synthesized
# notes (so verbatim snippets travel alongside summaries, addressed by stable
# source ids the justification can cite).

FAILED_CHECK_NOTICE = (
    "The checks listed under failed_checks did not run to completion (tool or model failure, "
    "or timeout). They are missing observations, not evidence of absence. "
    "Do not infer anything about the claim from them; mention them in evidence_gaps."
)


# Images travel as separate message parts after the JSON payload (see
# ``llm.stage_input``). The payload names them and says, per stage, how they
# may be used. The block exists only when the claim has images, so text-only
# inputs are unchanged.
IMAGE_NOTICES: dict[str, str] = {
    "plan": (
        "The claim comes with the images attached after this message, each after a line with its id. Read them as "
        "part of the claim: what they show, any text in them, and what the claim presents them as. When the claim "
        "rests on an image, plan checks on where and when the image appeared before the claim and on whether it "
        "shows what the claim says it shows."
    ),
    "research": (
        "The claim's images are attached after this message, each after a line with its id. What you see in an "
        "image tells you what is being claimed; it is not evidence. Evidence about an image (where and when it "
        "appeared, what it really shows) must come from sources your tools returned, cited by URL."
    ),
    "judge": (
        "The claim's images are attached after this message, each after a line with its id, so you can see what "
        "the claim presents. Base the verdict on the findings and their sources, not on your own reading of the "
        "images."
    ),
    "extract_claims": (
        "The input also includes the images attached after this message, each after a line with its id. Extract "
        "check-worthy claims stated in them (text in the image) and claims the input makes about them (what an "
        "image is presented as showing). For such a claim, source_fragment quotes the text in the image or the "
        "passage of input_text that presents the image."
    ),
}


def images_payload(stage: str, image_ids: list[str]) -> dict[str, Any]:
    """The ``images`` block of a stage payload."""
    return {"ids": list(image_ids), "instruction": IMAGE_NOTICES[stage]}


def with_images(payload: dict[str, Any], images: dict[str, Any] | None) -> dict[str, Any]:
    """Insert an ``images`` block right after the claim or input text; unchanged when there is none."""
    if not images:
        return payload
    result: dict[str, Any] = {}
    for key, value in payload.items():
        result[key] = value
        if key in ("claim", "input_text"):
            result["images"] = images
    if "images" not in result:
        result["images"] = images
    return result


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


def dated(payload: dict[str, Any], constraints: ResearchConstraints | None) -> dict[str, Any]:
    """Put the date the claim is evaluated as of in front of a stage payload.

    Without it a model resolves "current", "latest", and "this year" against
    its training cutoff. The planner is deliberately not told: given the date,
    it added checks on whether a title in the claim ("House Speaker X said")
    held on that date, and their "supports" findings diluted the verdict.
    """
    if constraints is None or not constraints.reference_date:
        return payload
    return {"reference_date": constraints.reference_date, **payload}


def _unmatched_marker(source: SourceEvidence) -> dict[str, str]:
    """Flag a cited URL that was not in the search results of the check citing it.

    Only this grounding level goes to a model: it is the one a reader can act
    on, and it means the same for every search provider. The full label stays
    on the finding for the report and the artifacts.
    """
    return {"grounding": SourceGrounding.UNMATCHED.value} if source.grounding == SourceGrounding.UNMATCHED else {}


def research_payload(
    *,
    claim: str,
    check: VerificationCheck,
    search_provider: str,
    constraints: ResearchConstraints | None,
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
    return payload


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
                        **_unmatched_marker(source),
                    }
                )
            elif source.grounding not in (None, SourceGrounding.UNMATCHED):
                # Another check did retrieve this URL (see ``merge_sources``).
                table[int(ids_by_url[key][1:]) - 1].pop("grounding", None)
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
                    "status": finding.status.value,
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
        entry: dict[str, Any] = {
            "aspect_id": finding.aspect_id,
            "question": finding.question,
            "acceptance_criteria": check.acceptance_criteria if check else [],
            "summary": finding.summary,
            "confidence": finding.confidence,
            "caveats": finding.caveats,
            "source_ids": source_ids,
        }
        if finding.stop_reason is not None:
            entry["stop_reason"] = finding.stop_reason.value
        grouped[finding.signal.value].append(entry)

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



def plan_payload(*, claim: str, max_checks: int) -> dict[str, Any]:
    """Planner input. It carries no reference date (see ``dated``)."""
    return {
        "claim": claim,
        "requirements": {
            "max_checks": max_checks,
            "acceptance_criteria_per_check": True,
            "checks_must_be_independent": True,
        },
    }


def extraction_payload(*, input_text: str, max_claims: int) -> dict[str, Any]:
    return {
        "input_text": input_text,
        "requirements": {
            "max_claims": max_claims,
            "decontextualized": True,
            "atomic_claims": True,
            "maximize_checkworthy_coverage": True,
            "only_directly_mentioned_facts": True,
            "detect_and_report_language": True,
            "write_output_in_input_language": True,
            "preserve_original_diacritics": True,
        },
    }
