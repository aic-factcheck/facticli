from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from facticli.application.config import REASONING_EFFORTS, STRATEGIES, FactCheckRuntimeConfig, parse_stage_assignments
from facticli.application.factory import build_fact_check_service
from facticli.application.repository import FileRunArtifactRepository
from facticli.cli_validators import non_negative_int, positive_int, search_results_int
from facticli.core.contracts import FactCheckReport, VeracityVerdict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m facticli.averitec_submission",
        description=("Run facticli on Averitec-formatted claims and write an Averitec submission JSON file."),
    )
    parser.add_argument("--input", required=True, help="Path to Averitec input JSON file.")
    parser.add_argument("--output", required=True, help="Path to write submission JSON file.")
    parser.add_argument(
        "--claim-field",
        default="claim",
        help="Field containing claim text in each input record (default: claim).",
    )
    parser.add_argument(
        "--claim-id-field",
        default=None,
        help=(
            "Optional field containing claim ID. If missing, falls back to claim_id/id or "
            "the zero-based input row index."
        ),
    )
    parser.add_argument(
        "--offset",
        type=non_negative_int,
        default=0,
        help="Zero-based index of first claim to process (default: 0).",
    )
    parser.add_argument(
        "--limit",
        type=positive_int,
        default=None,
        help="Maximum number of claims to process (default: all from offset).",
    )
    parser.add_argument(
        "--parallel-claims",
        type=positive_int,
        default=1,
        help="Number of claims to fact-check concurrently (default: 1).",
    )
    parser.add_argument(
        "--max-evidence",
        type=positive_int,
        default=10,
        help="Maximum evidence entries per submission row (default: 10).",
    )
    parser.add_argument(
        "--empty-question",
        action="store_true",
        help="Set evidence question to empty string and keep only declarative answer text.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first failed claim instead of writing a fallback prediction.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name override. Falls back to OPENAI_API_MODEL.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible base URL override. Falls back to OPENAI_API_BASE_URL.",
    )
    parser.add_argument(
        "--strategy",
        choices=list(STRATEGIES),
        default="pipeline",
        help="pipeline (default) or single_agent (no-harness baseline with the same model, tool, and budget).",
    )
    parser.add_argument(
        "--stage-model",
        action="append",
        default=None,
        metavar="STAGE=MODEL",
        help="Route one stage to a different model (repeatable), e.g. research=gpt-5-mini.",
    )
    parser.add_argument(
        "--stage-effort",
        action="append",
        default=None,
        metavar="STAGE=EFFORT",
        help="Reasoning effort per stage (repeatable): " + ", ".join(REASONING_EFFORTS) + ".",
    )
    parser.add_argument(
        "--token-budget",
        type=positive_int,
        default=None,
        help="Shared token budget per claim across all stages (default: unlimited).",
    )
    parser.add_argument(
        "--model-retries",
        type=non_negative_int,
        default=3,
        help="Max SDK-managed retries per model call for transient errors (default: 3).",
    )
    parser.add_argument(
        "--retry-budget",
        type=non_negative_int,
        default=12,
        help="Max model-call retries per claim run (default: 12).",
    )
    parser.add_argument(
        "--feedback-rounds",
        type=non_negative_int,
        default=0,
        help="Bounded review/follow-up rounds per claim (default: 0).",
    )
    parser.add_argument(
        "--follow-up-checks",
        type=positive_int,
        default=2,
        help="Max new follow-up checks per feedback round (default: 2).",
    )
    parser.add_argument(
        "--verify-sources",
        action="store_true",
        help="Run the citation URL health pass (live/archived/broken) after judging.",
    )
    parser.add_argument(
        "--max-checks",
        type=positive_int,
        default=4,
        help="Maximum number of verification checks per claim.",
    )
    parser.add_argument(
        "--parallel",
        type=positive_int,
        default=4,
        help="Maximum parallel research workers within each claim.",
    )
    parser.add_argument(
        "--search-provider",
        choices=["openai", "brave", "knowledge_store"],
        default=os.getenv("FACTICLI_SEARCH_PROVIDER", "openai"),
        help="Search backend for research stage.",
    )
    parser.add_argument(
        "--knowledge-store-dir",
        default=None,
        help="Directory of per-claim AVeriTeC knowledge store files ({claim_id}.json[l]); required for --search-provider knowledge_store.",
    )
    parser.add_argument(
        "--claim-date-field",
        default="claim_date",
        help="Input field holding the claim date; used as evidence cutoff (default: claim_date). Pass '' to disable.",
    )
    parser.add_argument(
        "--block-fact-checkers",
        action="store_true",
        help="Block known fact-checking domains from research sources (label-leakage control).",
    )
    parser.add_argument(
        "--blocked-domain",
        action="append",
        default=None,
        dest="blocked_domains",
        help="Additional domain to block (repeatable).",
    )
    parser.add_argument(
        "--artifacts-dir",
        default=None,
        help="Directory to persist full per-run artifacts JSON (plans, findings, sources, usage) for replay and audits.",
    )
    parser.add_argument(
        "--run-info",
        default=None,
        help="Path for the per-claim usage/latency manifest (default: <output>.runinfo.json).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip claims whose claim_id already exists in the output file and merge new rows into it.",
    )
    parser.add_argument(
        "--search-context-size",
        choices=["low", "medium", "high"],
        default="high",
        help="Hosted search context size for --search-provider openai.",
    )
    parser.add_argument(
        "--search-results",
        type=search_results_int,
        default=5,
        dest="search_results_per_query",
        help="Number of search results per query (1..20).",
    )
    return parser


def _validate_env(args: argparse.Namespace) -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not (api_key and api_key.strip()):
        raise RuntimeError("OPENAI_API_KEY is not set. Export it or add it to .env.")
    env_model = os.getenv("OPENAI_API_MODEL")
    if not (args.model and args.model.strip()) and not (
        env_model and env_model.strip()
    ):
        raise RuntimeError(
            "OPENAI_API_MODEL is not set. Export it, add it to .env, or pass --model."
        )
    if args.search_provider == "brave" and not os.getenv("BRAVE_SEARCH_API_KEY"):
        raise RuntimeError("BRAVE_SEARCH_API_KEY is not set. Export it or use --search-provider openai.")
    if args.search_provider == "knowledge_store" and not args.knowledge_store_dir:
        raise RuntimeError("--knowledge-store-dir is required with --search-provider knowledge_store.")
    try:
        parse_stage_assignments(args.stage_model, option_name="--stage-model")
        parse_stage_assignments(args.stage_effort, allowed_values=REASONING_EFFORTS, option_name="--stage-effort")
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc


def _load_input_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        if "claims" in payload and isinstance(payload["claims"], list):
            payload = payload["claims"]
        else:
            raise ValueError("Input JSON must be a list or a dict containing a 'claims' list.")
    if not isinstance(payload, list):
        raise ValueError("Input JSON must be a list.")
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Input row {index} is not an object.")
    return payload


def _resolve_claim_id(record: dict[str, Any], row_index: int, claim_id_field: str | None) -> Any:
    candidate_fields: list[str] = []
    if claim_id_field:
        candidate_fields.append(claim_id_field)
    candidate_fields.extend(["claim_id", "id"])
    for field in candidate_fields:
        value = record.get(field)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return row_index


def _extract_claim_text(record: dict[str, Any], row_index: int, claim_field: str) -> str:
    claim_text = record.get(claim_field)
    if claim_text is None:
        raise ValueError(f"Input row {row_index} does not contain claim field {claim_field!r}.")
    claim = str(claim_text).strip()
    if not claim:
        raise ValueError(f"Input row {row_index} has empty claim in field {claim_field!r}.")
    return claim


def _append_evidence_entry(
    *,
    evidence: list[dict[str, str]],
    seen: set[tuple[str, str, str]],
    question: str,
    answer: str,
    url: str,
    scraped_text: str,
    max_evidence: int,
    empty_question: bool,
) -> None:
    if len(evidence) >= max_evidence:
        return
    normalized_url = url.strip()
    if not normalized_url:
        return
    normalized_answer = answer.strip()
    if not normalized_answer:
        return
    normalized_question = "" if empty_question else question.strip()
    normalized_scraped = scraped_text.strip() or normalized_answer
    dedupe_key = (normalized_url, normalized_question, normalized_answer)
    if dedupe_key in seen:
        return
    seen.add(dedupe_key)
    evidence.append(
        {
            "question": normalized_question,
            "answer": normalized_answer,
            "url": normalized_url,
            "scraped_text": normalized_scraped,
        }
    )


def build_submission_evidence(
    report: FactCheckReport,
    *,
    max_evidence: int,
    empty_question: bool,
) -> list[dict[str, str]]:
    evidence: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    for finding in report.findings:
        question = finding.question.strip()
        answer = finding.summary.strip() or report.justification.strip()
        for source in finding.sources:
            _append_evidence_entry(
                evidence=evidence,
                seen=seen,
                question=question,
                answer=answer,
                url=source.url,
                scraped_text=source.snippet,
                max_evidence=max_evidence,
                empty_question=empty_question,
            )

    if len(evidence) < max_evidence:
        general_question = (
            "" if empty_question else "What evidence supports the final verdict for this claim?"
        )
        general_answer = report.justification.strip()
        for source in report.sources:
            _append_evidence_entry(
                evidence=evidence,
                seen=seen,
                question=general_question,
                answer=general_answer,
                url=source.url,
                scraped_text=source.snippet,
                max_evidence=max_evidence,
                empty_question=empty_question,
            )

    return evidence[:max_evidence]


def build_submission_row(
    *,
    record: dict[str, Any],
    row_index: int,
    claim_field: str,
    claim_id_field: str | None,
    report: FactCheckReport,
    max_evidence: int,
    empty_question: bool,
) -> dict[str, Any]:
    claim = _extract_claim_text(record, row_index, claim_field)
    return {
        "claim_id": _resolve_claim_id(record, row_index, claim_id_field),
        "claim": claim,
        "pred_label": report.verdict.value,
        "evidence": build_submission_evidence(
            report,
            max_evidence=max_evidence,
            empty_question=empty_question,
        ),
    }


def build_failed_submission_row(
    *,
    record: dict[str, Any],
    row_index: int,
    claim_field: str,
    claim_id_field: str | None,
) -> dict[str, Any]:
    claim = _extract_claim_text(record, row_index, claim_field)
    return {
        "claim_id": _resolve_claim_id(record, row_index, claim_id_field),
        "claim": claim,
        "pred_label": VeracityVerdict.NOT_ENOUGH_EVIDENCE.value,
        "evidence": [],
    }


async def _run_batch(
    *,
    indexed_records: list[tuple[int, dict[str, Any]]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    config = FactCheckRuntimeConfig(
        model=args.model,
        base_url=args.base_url,
        stage_models=parse_stage_assignments(args.stage_model, option_name="--stage-model"),
        stage_efforts=parse_stage_assignments(
            args.stage_effort, allowed_values=REASONING_EFFORTS, option_name="--stage-effort"
        ),
        model_retry_attempts=args.model_retries,
        retry_budget=args.retry_budget,
        strategy=args.strategy,
        max_checks=args.max_checks,
        max_parallel_research=args.parallel,
        max_feedback_rounds=args.feedback_rounds,
        max_follow_up_checks=args.follow_up_checks,
        token_budget=args.token_budget,
        verify_sources=args.verify_sources,
        search_context_size=args.search_context_size,
        search_provider=args.search_provider,
        search_results_per_query=args.search_results_per_query,
        block_fact_checkers=args.block_fact_checkers,
        blocked_domains=tuple(args.blocked_domains or ()),
        knowledge_store_dir=args.knowledge_store_dir,
    )
    artifact_repository = (
        FileRunArtifactRepository(output_dir=args.artifacts_dir) if args.artifacts_dir else None
    )
    service = build_fact_check_service(config=config, artifact_repository=artifact_repository)
    semaphore = asyncio.Semaphore(max(1, args.parallel_claims))
    ordered_rows: list[dict[str, Any] | None] = [None] * len(indexed_records)
    ordered_run_info: list[dict[str, Any] | None] = [None] * len(indexed_records)

    async def process_one(
        local_index: int, row_index: int, record: dict[str, Any]
    ) -> tuple[int, dict[str, Any], dict[str, Any]]:
        claim = _extract_claim_text(record, row_index, args.claim_field)
        claim_id = _resolve_claim_id(record, row_index, args.claim_id_field)
        claim_date = (
            str(record.get(args.claim_date_field) or "").strip() or None
            if args.claim_date_field
            else None
        )
        run_info: dict[str, Any] = {"claim_id": claim_id, "claim_date": claim_date}
        async with semaphore:
            try:
                run = await service.check_claim(claim, claim_id=claim_id, claim_date=claim_date)
            except Exception as exc:
                if args.fail_fast:
                    raise RuntimeError(
                        f"Claim row {row_index} failed ({claim[:120]}): {type(exc).__name__}: {exc}"
                    ) from exc
                print(
                    f"[warn] Claim row {row_index} failed, writing fallback label: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                run_info["error"] = f"{type(exc).__name__}: {exc}"
                return (
                    local_index,
                    build_failed_submission_row(
                        record=record,
                        row_index=row_index,
                        claim_field=args.claim_field,
                        claim_id_field=args.claim_id_field,
                    ),
                    run_info,
                )

        artifacts = run.artifacts
        run_info["duration_seconds"] = artifacts.duration_seconds
        if artifacts.usage_summary is not None:
            run_info["usage"] = artifacts.usage_summary.model_dump()
        run_info["verdict"] = run.report.verdict.value
        run_info["verdict_confidence"] = run.report.verdict_confidence
        run_info["strategy"] = artifacts.strategy
        if artifacts.budget is not None:
            run_info["budget"] = artifacts.budget.model_dump()
        run_info["failed_checks"] = sum(
            1 for check in artifacts.research_checks if check.finding is not None and not check.finding.is_observation
        )
        if artifacts.citation_check:
            run_info["citation_check"] = artifacts.citation_check
        run_info["removed_source_count"] = sum(
            len(check.removed_sources) for check in artifacts.research_checks
        )
        row = build_submission_row(
            record=record,
            row_index=row_index,
            claim_field=args.claim_field,
            claim_id_field=args.claim_id_field,
            report=run.report,
            max_evidence=args.max_evidence,
            empty_question=args.empty_question,
        )
        return local_index, row, run_info

    tasks = [
        asyncio.create_task(process_one(local_index, row_index, record))
        for local_index, (row_index, record) in enumerate(indexed_records)
    ]

    completed = 0
    for task in asyncio.as_completed(tasks):
        local_index, row, run_info = await task
        ordered_rows[local_index] = row
        ordered_run_info[local_index] = run_info
        completed += 1
        print(
            f"[progress] Completed {completed}/{len(indexed_records)} claims.",
            file=sys.stderr,
            flush=True,
        )

    return (
        [row for row in ordered_rows if row is not None],
        [info for info in ordered_run_info if info is not None],
    )


def _slice_records(
    records: list[dict[str, Any]],
    *,
    offset: int,
    limit: int | None,
) -> tuple[int, list[dict[str, Any]]]:
    if offset >= len(records):
        return offset, []
    if limit is None:
        return offset, records[offset:]
    return offset, records[offset : offset + limit]


async def _run(args: argparse.Namespace) -> int:
    try:
        _validate_env(args)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        print(f"Input file does not exist: {input_path}", file=sys.stderr)
        return 2
    if not input_path.is_file():
        print(f"Input path is not a file: {input_path}", file=sys.stderr)
        return 2

    try:
        records = _load_input_records(input_path)
    except Exception as exc:
        print(f"Failed to load input JSON: {exc}", file=sys.stderr)
        return 2

    offset, sliced_records = _slice_records(records, offset=args.offset, limit=args.limit)
    if not sliced_records:
        print("No claims selected for processing.", file=sys.stderr)
        return 2

    indexed_records = [(offset + index, record) for index, record in enumerate(sliced_records)]

    existing_rows: list[dict[str, Any]] = []
    if args.resume and output_path.is_file():
        existing_rows = json.loads(output_path.read_text(encoding="utf-8"))
        existing_ids = {str(row.get("claim_id")) for row in existing_rows}
        indexed_records = [
            (row_index, record)
            for row_index, record in indexed_records
            if str(_resolve_claim_id(record, row_index, args.claim_id_field)) not in existing_ids
        ]
        print(
            f"[info] Resume: {len(existing_rows)} rows already in {output_path}, "
            f"{len(indexed_records)} claims left to process.",
            file=sys.stderr,
            flush=True,
        )
        if not indexed_records:
            print("[info] Nothing to do.", file=sys.stderr)
            return 0

    print(
        f"[info] Processing {len(indexed_records)} claims from {input_path} "
        f"(offset={offset}, parallel_claims={args.parallel_claims}, "
        f"search_provider={args.search_provider}, block_fact_checkers={args.block_fact_checkers}).",
        file=sys.stderr,
        flush=True,
    )

    try:
        submission_rows, run_info_rows = await _run_batch(indexed_records=indexed_records, args=args)
    except Exception as exc:
        print(f"Batch run failed: {exc}", file=sys.stderr)
        return 1

    all_rows = [*existing_rows, *submission_rows]

    def _row_sort_key(row: dict[str, Any]) -> Any:
        claim_id = row.get("claim_id")
        try:
            return (0, int(claim_id))
        except (TypeError, ValueError):
            return (1, str(claim_id))

    all_rows.sort(key=_row_sort_key)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(all_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"[info] Wrote {len(all_rows)} submission rows to {output_path}.",
        file=sys.stderr,
        flush=True,
    )

    run_info_path = Path(args.run_info) if args.run_info else output_path.with_name(
        output_path.stem + ".runinfo.json"
    )
    manifest = {
        "input": str(input_path),
        "output": str(output_path),
        "settings": {
            "model": args.model or os.getenv("OPENAI_API_MODEL"),
            "stage_models": args.stage_model or [],
            "stage_efforts": args.stage_effort or [],
            "strategy": args.strategy,
            "token_budget": args.token_budget,
            "model_retries": args.model_retries,
            "retry_budget": args.retry_budget,
            "feedback_rounds": args.feedback_rounds,
            "follow_up_checks": args.follow_up_checks,
            "verify_sources": args.verify_sources,
            "base_url": args.base_url or os.getenv("OPENAI_API_BASE_URL"),
            "search_provider": args.search_provider,
            "search_context_size": args.search_context_size,
            "max_checks": args.max_checks,
            "parallel_research": args.parallel,
            "parallel_claims": args.parallel_claims,
            "block_fact_checkers": args.block_fact_checkers,
            "blocked_domains": args.blocked_domains or [],
            "claim_date_field": args.claim_date_field,
            "knowledge_store_dir": args.knowledge_store_dir,
            "artifacts_dir": args.artifacts_dir,
        },
        "claims": run_info_rows,
    }
    if args.resume and run_info_path.is_file():
        previous = json.loads(run_info_path.read_text(encoding="utf-8"))
        manifest["claims"] = [*previous.get("claims", []), *run_info_rows]
    run_info_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[info] Wrote run manifest to {run_info_path}.", file=sys.stderr, flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
