from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from facticli.core.artifacts import RunArtifacts
from facticli.core.constraints import FACT_CHECK_DOMAINS, is_blocked_url, violates_date_cutoff
from facticli.core.contracts import FindingStatus, SourceTier
from facticli.core.source_quality import classify_source_tier

# Post-hoc audit over persisted run artifacts (FileRunArtifactRepository JSON).
#
# Two questions from the 2026 fact-checking literature that a benchmark number
# alone cannot answer:
#   1. Leakage: how often did the final report rest on fact-checking sites or
#      on evidence published after the claim date (URL blocklists are known to
#      be insufficient; content-level auditing is required)?
#   2. Evidence dependence: does the verdict actually depend on the retrieved
#      evidence, or does the judge answer from parametric knowledge? FAE/REAL
#      (2026) measure this by removing the evidence and re-judging.


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m facticli.artifacts_audit",
        description="Audit persisted facticli run artifacts for leakage, failures, budget use, and evidence dependence.",
    )
    parser.add_argument("--artifacts-dir", required=True, help="Directory of run_*.json artifact files.")
    parser.add_argument(
        "--gold",
        default=None,
        help="Optional AVeriTeC gold JSON; enables accuracy splits (leaked vs clean) keyed by claim_id row index.",
    )
    parser.add_argument("--output", default=None, help="Optional path to write the audit JSON.")
    parser.add_argument("--limit", type=int, default=None, help="Audit at most N artifact files.")
    parser.add_argument(
        "--evidence-ablation",
        action="store_true",
        help="Re-run the judge with all findings removed and report how many verdicts survive (needs API access).",
    )
    parser.add_argument("--model", default=None, help="Judge model for --evidence-ablation (default: OPENAI_API_MODEL).")
    parser.add_argument("--base-url", default=None, help="Judge base URL for --evidence-ablation.")
    parser.add_argument("--ablation-workers", type=int, default=4, help="Concurrent judge calls (default: 4).")
    return parser


def load_artifacts(directory: Path, limit: int | None = None) -> list[tuple[Path, RunArtifacts]]:
    """Load and validate every run_*.json in the directory (sorted by name)."""
    runs: list[tuple[Path, RunArtifacts]] = []
    for path in sorted(directory.glob("run_*.json")):
        try:
            runs.append((path, RunArtifacts.model_validate_json(path.read_text(encoding="utf-8"))))
        except Exception as exc:  # corrupt or foreign file
            print(f"[warn] skipping {path.name}: {exc}", file=sys.stderr)
        if limit is not None and len(runs) >= limit:
            break
    return runs


def audit_run(artifacts: RunArtifacts) -> dict[str, Any]:
    """Per-run leakage, failure, and usage indicators."""
    report = artifacts.report_final
    claim_date = artifacts.constraints.claim_date if artifacts.constraints else None
    final_sources = report.sources if report else []

    fact_checker_final = sum(
        1
        for source in final_sources
        if (source.tier or classify_source_tier(source.url)) == SourceTier.FACT_CHECKER
        or is_blocked_url(source.url, FACT_CHECK_DOMAINS)
    )
    post_date_final = sum(1 for source in final_sources if violates_date_cutoff(source.published_at, claim_date))
    undated_final = sum(1 for source in final_sources if not source.published_at)

    raw_fact_checker = 0
    removed = 0
    failed_checks = 0
    budget_skipped = 0
    for check in artifacts.research_checks:
        removed += len(check.removed_sources)
        finding = check.finding
        if finding is None:
            continue
        if finding.status == FindingStatus.FAILED:
            failed_checks += 1
        elif finding.status == FindingStatus.BUDGET_EXHAUSTED:
            budget_skipped += 1
        raw_fact_checker += sum(
            1 for source in [*finding.sources, *check.removed_sources] if is_blocked_url(source.url, FACT_CHECK_DOMAINS)
        )

    tier_counts: dict[str, int] = {}
    for source in final_sources:
        tier = source.tier or classify_source_tier(source.url)
        tier_counts[tier.value] = tier_counts.get(tier.value, 0) + 1

    usage = artifacts.usage_summary
    return {
        "claim_id": artifacts.claim_id,
        "claim": artifacts.normalized_claim,
        "strategy": artifacts.strategy,
        "verdict": report.verdict.value if report else None,
        "verdict_confidence": report.verdict_confidence if report else None,
        "claim_date": claim_date,
        "final_source_count": len(final_sources),
        "fact_checker_sources_in_final": fact_checker_final,
        "fact_checker_sources_seen_raw": raw_fact_checker,
        "removed_sources": removed,
        "post_claim_date_sources_in_final": post_date_final,
        "undated_sources_in_final": undated_final,
        "tier_counts": tier_counts,
        "checks": len(artifacts.research_checks),
        "failed_checks": failed_checks,
        "budget_skipped_checks": budget_skipped,
        "review_rounds": len(artifacts.review_rounds),
        "total_tokens": usage.total_tokens if usage else None,
        "cache_hit_rate": usage.cache_hit_rate if usage else None,
        "duration_seconds": artifacts.duration_seconds,
        "leaked": bool(fact_checker_final or raw_fact_checker or post_date_final),
    }


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def aggregate(rows: list[dict[str, Any]], gold: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Aggregate per-run rows into rates; add accuracy splits when gold is given."""
    total = len(rows)
    summary: dict[str, Any] = {
        "runs": total,
        "runs_with_fact_checker_source_in_final": sum(1 for row in rows if row["fact_checker_sources_in_final"]),
        "runs_with_fact_checker_source_seen": sum(1 for row in rows if row["fact_checker_sources_seen_raw"]),
        "runs_with_post_claim_date_source": sum(1 for row in rows if row["post_claim_date_sources_in_final"]),
        "runs_leaked": sum(1 for row in rows if row["leaked"]),
        "mean_removed_sources": _mean([row["removed_sources"] for row in rows]),
        "mean_final_sources": _mean([row["final_source_count"] for row in rows]),
        "share_undated_final_sources": _mean(
            [
                row["undated_sources_in_final"] / row["final_source_count"]
                for row in rows
                if row["final_source_count"]
            ]
        ),
        "failed_check_share": _mean(
            [row["failed_checks"] / row["checks"] for row in rows if row["checks"]]
        ),
        "runs_with_budget_skips": sum(1 for row in rows if row["budget_skipped_checks"]),
        "mean_total_tokens": _mean([row["total_tokens"] for row in rows if row["total_tokens"] is not None]),
        "mean_cache_hit_rate": _mean([row["cache_hit_rate"] for row in rows if row["cache_hit_rate"] is not None]),
        "mean_duration_seconds": _mean([row["duration_seconds"] for row in rows if row["duration_seconds"] is not None]),
        "verdict_counts": _count([row["verdict"] for row in rows]),
        "tier_counts": _sum_dicts([row["tier_counts"] for row in rows]),
    }
    if gold is not None:
        leaked_correct: list[bool] = []
        clean_correct: list[bool] = []
        for row in rows:
            gold_label = _gold_label(gold, row["claim_id"])
            if gold_label is None or row["verdict"] is None:
                continue
            correct = gold_label == row["verdict"]
            (leaked_correct if row["leaked"] else clean_correct).append(correct)
        summary["accuracy_split"] = {
            "leaked": {"n": len(leaked_correct), "accuracy": _mean([float(c) for c in leaked_correct])},
            "clean": {"n": len(clean_correct), "accuracy": _mean([float(c) for c in clean_correct])},
        }
    return summary


def _gold_label(gold: list[dict[str, Any]], claim_id: Any) -> str | None:
    try:
        index = int(claim_id)
    except (TypeError, ValueError):
        return None
    if not 0 <= index < len(gold):
        return None
    return str(gold[index].get("label", "")) or None


def _count(values: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _sum_dicts(dicts: list[dict[str, int]]) -> dict[str, int]:
    total: dict[str, int] = {}
    for entry in dicts:
        for key, value in entry.items():
            total[key] = total.get(key, 0) + value
    return total


async def run_evidence_ablation(
    runs: list[tuple[Path, RunArtifacts]],
    *,
    model: str | None,
    base_url: str | None,
    workers: int,
) -> dict[str, Any]:
    """Re-judge every run with an empty evidence set and count surviving verdicts.

    A verdict that survives without evidence was not evidence-driven; a high
    survival rate for Supported/Refuted means the judge answers from memory.
    """
    from facticli.adapters import CompatibleJudgeAdapter, configure_inference_client, load_inference_config

    inference = load_inference_config(requested_model=model, base_url=base_url)
    configure_inference_client(inference)
    judge = CompatibleJudgeAdapter(model=inference.model, max_turns=6, retry_attempts=2)
    semaphore = asyncio.Semaphore(max(1, workers))

    async def ablate(path: Path, artifacts: RunArtifacts) -> dict[str, Any]:
        report = artifacts.report_final
        plan = artifacts.plan_normalized
        if report is None or plan is None:
            return {"file": path.name, "skipped": True}
        async with semaphore:
            try:
                ablated = await judge.judge(claim=artifacts.normalized_claim, plan=plan, findings=[])
            except Exception as exc:
                return {"file": path.name, "error": f"{type(exc).__name__}: {exc}"}
        return {
            "file": path.name,
            "claim_id": artifacts.claim_id,
            "original_verdict": report.verdict.value,
            "ablated_verdict": ablated.verdict.value,
            "ablated_confidence": ablated.verdict_confidence,
            "survived": ablated.verdict == report.verdict,
        }

    results = await asyncio.gather(*(ablate(path, artifacts) for path, artifacts in runs))
    scored = [row for row in results if "survived" in row]
    survived = [row for row in scored if row["survived"]]
    by_verdict: dict[str, dict[str, int]] = {}
    for row in scored:
        bucket = by_verdict.setdefault(row["original_verdict"], {"n": 0, "survived": 0})
        bucket["n"] += 1
        bucket["survived"] += int(row["survived"])
    return {
        "judge_model": inference.model,
        "runs_scored": len(scored),
        "verdicts_surviving_without_evidence": len(survived),
        "survival_rate": _mean([float(row["survived"]) for row in scored]),
        "ablated_nei_rate": _mean([float(row["ablated_verdict"] == "Not Enough Evidence") for row in scored]),
        "by_original_verdict": by_verdict,
        "rows": results,
    }


def _run(args: argparse.Namespace) -> int:
    directory = Path(args.artifacts_dir)
    if not directory.is_dir():
        print(f"Artifacts directory does not exist: {directory}", file=sys.stderr)
        return 2
    runs = load_artifacts(directory, limit=args.limit)
    if not runs:
        print("No run_*.json artifacts found.", file=sys.stderr)
        return 2

    gold = None
    if args.gold:
        gold_path = Path(args.gold)
        if not gold_path.is_file():
            print(f"Gold file does not exist: {gold_path}", file=sys.stderr)
            return 2
        gold = json.loads(gold_path.read_text(encoding="utf-8"))

    rows = [audit_run(artifacts) for _, artifacts in runs]
    audit: dict[str, Any] = {
        "artifacts_dir": str(directory),
        "summary": aggregate(rows, gold),
        "runs": rows,
    }

    if args.evidence_ablation:
        if not os.getenv("OPENAI_API_KEY"):
            print("Evidence ablation requires OPENAI_API_KEY.", file=sys.stderr)
            return 2
        audit["evidence_ablation"] = asyncio.run(
            run_evidence_ablation(runs, model=args.model, base_url=args.base_url, workers=args.ablation_workers)
        )

    _print_summary(audit)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[info] Wrote audit to {output_path}.", file=sys.stderr)
    return 0


def _print_summary(audit: dict[str, Any]) -> None:
    summary = audit["summary"]
    print(f"runs: {summary['runs']}")
    print(
        "leakage: "
        f"{summary['runs_with_fact_checker_source_in_final']} run(s) cite a fact-checker in the final report, "
        f"{summary['runs_with_fact_checker_source_seen']} saw one during research, "
        f"{summary['runs_with_post_claim_date_source']} cite post-claim-date evidence "
        f"({summary['runs_leaked']} leaked overall)"
    )
    print(
        f"sources: mean {summary['mean_final_sources']} per report, mean {summary['mean_removed_sources']} removed by constraints, "
        f"undated share {summary['share_undated_final_sources']}"
    )
    print(f"tiers: {summary['tier_counts']}")
    print(
        f"robustness: failed-check share {summary['failed_check_share']}, "
        f"{summary['runs_with_budget_skips']} run(s) hit the token budget"
    )
    print(
        f"cost: mean tokens {summary['mean_total_tokens']}, mean cache hit rate {summary['mean_cache_hit_rate']}, "
        f"mean duration {summary['mean_duration_seconds']}s"
    )
    print(f"verdicts: {summary['verdict_counts']}")
    if "accuracy_split" in summary:
        split = summary["accuracy_split"]
        print(
            f"accuracy: leaked n={split['leaked']['n']} acc={split['leaked']['accuracy']} | "
            f"clean n={split['clean']['n']} acc={split['clean']['accuracy']}"
        )
    if "evidence_ablation" in audit:
        ablation = audit["evidence_ablation"]
        print(
            f"evidence ablation ({ablation['judge_model']}): {ablation['verdicts_surviving_without_evidence']}/"
            f"{ablation['runs_scored']} verdicts survive with evidence removed "
            f"(survival {ablation['survival_rate']}, ablated NEI rate {ablation['ablated_nei_rate']})"
        )
        for verdict, bucket in ablation["by_original_verdict"].items():
            print(f"  {verdict:<38} {bucket['survived']}/{bucket['n']} survive")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
