from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from itertools import zip_longest
from pathlib import Path
from typing import Any

from facticli.check_options import add_fact_check_args, fact_check_config, missing_environment
from facticli.cli_validators import non_negative_int, positive_int
from facticli.core.constraints import normalize_claim_date
from facticli.core.contracts import FactCheckReport, VeracityVerdict
from facticli.core.errors import classify_exception, describe_exception, is_fatal
from facticli.core.grounding import summarize_grounding
from facticli.core.images import ImageError, load_images
from facticli.pipeline import build_fact_checker

GENERAL_QUESTION = "What evidence supports the final verdict for this claim?"
# Exit code when a batch stops early on an error no retry can fix (rejected key, exhausted quota, unknown model).
EXIT_ABORTED = 3


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
        help="Stop at the first failed claim instead of writing a fallback prediction; completed rows are kept (exit 1).",
    )
    add_fact_check_args(parser, search_providers=("openai", "brave", "exa", "knowledge_store"))
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
        "--image-field",
        default="images",
        help=(
            "Input field holding the claim's images as a list of file names or URLs (default: images); records "
            "without it are text-only."
        ),
    )
    parser.add_argument(
        "--image-dir",
        default=None,
        help="Directory that relative image file names are resolved against (default: the input file's directory).",
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
    return parser


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


def image_sources(record: dict[str, Any], row_index: int, image_field: str, image_dir: Path) -> list[str]:
    """Paths or URLs of a record's images, in order; relative names resolve against ``image_dir``."""
    value = record.get(image_field) if image_field else None
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"Input row {row_index}: field {image_field!r} must be a list of file names or URLs.")
    sources = []
    for item in value:
        item = item.strip()
        if item.startswith(("http://", "https://")) or Path(item).is_absolute():
            sources.append(item)
        elif item:
            sources.append(str(image_dir / item))
    return sources


def build_submission_evidence(
    report: FactCheckReport,
    *,
    max_evidence: int,
    empty_question: bool,
) -> list[dict[str, str]]:
    """Turn a report into at most ``max_evidence`` question-answer rows that each state a different fact.

    Evidence-recall scoring matches the submitted question-answer pairs against
    the reference pairs, so a finding summary repeated under each of its
    sources is one fact occupying several slots. Every finding that observed
    something gets one row (question + summary) before any gets a second; the
    remaining rows carry the sources' own snippets, round-robin across
    findings, then the report-level sources.
    """
    observed = [finding for finding in report.findings if finding.is_observation and finding.sources]
    candidates = [
        (finding.question, finding.summary.strip() or finding.sources[0].snippet, finding.sources[0])
        for finding in observed
    ]
    for rank in zip_longest(*([(finding.question, source) for source in finding.sources] for finding in observed)):
        candidates.extend((question, source.snippet, source) for question, source in filter(None, rank))
    candidates.extend((GENERAL_QUESTION, source.snippet, source) for source in report.sources)

    evidence: list[dict[str, str]] = []
    seen_answers: set[str] = set()
    for question, answer, source in candidates:
        if len(evidence) >= max_evidence:
            break
        answer = " ".join(answer.split())
        url = source.url.strip()
        if not answer or not url or answer.casefold() in seen_answers:
            continue
        seen_answers.add(answer.casefold())
        evidence.append(
            {
                "question": "" if empty_question else question.strip(),
                "answer": answer,
                "url": url,
                "scraped_text": source.snippet.strip() or answer,
            }
        )
    return evidence


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


def _build_service(args: argparse.Namespace) -> Any:
    return build_fact_checker(
        fact_check_config(args, knowledge_store_dir=args.knowledge_store_dir), artifacts_dir=args.artifacts_dir
    )


async def _run_batch(
    *,
    indexed_records: list[tuple[int, dict[str, Any]]],
    args: argparse.Namespace,
    service: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], tuple[int, str] | None]:
    """Run the selected claims; return ``(rows, run info, abort)``.

    ``abort`` is ``(exit code, reason)`` when the batch stopped early.

    A claim that fails for an ordinary reason gets a fallback row. A failure
    no retry can fix aborts the batch instead: the failing claim and everything
    not yet started are left out of the output, so ``--resume`` picks them up
    rather than keeping fallback labels. ``--fail-fast`` stops the same way
    on any failure.
    """
    semaphore = asyncio.Semaphore(max(1, args.parallel_claims))
    ordered_rows: list[dict[str, Any] | None] = [None] * len(indexed_records)
    ordered_run_info: list[dict[str, Any] | None] = [None] * len(indexed_records)
    abort: tuple[int, str] | None = None  # (exit code, reason) once the batch stops early

    async def process_one(
        local_index: int, row_index: int, record: dict[str, Any]
    ) -> tuple[int, dict[str, Any] | None, dict[str, Any] | None]:
        nonlocal abort
        claim = _extract_claim_text(record, row_index, args.claim_field)
        claim_id = _resolve_claim_id(record, row_index, args.claim_id_field)
        claim_date = (
            str(record.get(args.claim_date_field) or "").strip() or None
            if args.claim_date_field
            else None
        )
        run_info: dict[str, Any] = {"claim_id": claim_id, "claim_date": claim_date}
        async with semaphore:
            if abort:
                return local_index, None, None
            try:
                images = await asyncio.to_thread(
                    load_images, image_sources(record, row_index, args.image_field, args.image_dir)
                )
                if images:
                    run_info["images"] = len(images)
                run = await service.check_claim(claim, claim_id=claim_id, claim_date=claim_date, images=images)
            except Exception as exc:
                if is_fatal(classify_exception(exc)):
                    abort = abort or (EXIT_ABORTED, f"{describe_exception(exc)}. No retry can fix this error")
                    return local_index, None, None
                if args.fail_fast:
                    abort = abort or (1, f"claim row {row_index} failed under --fail-fast: {type(exc).__name__}: {exc}")
                    return local_index, None, None
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
        run_info["failed_checks"] = sum(
            1 for check in artifacts.research_checks if check.finding is not None and not check.finding.is_observation
        )
        run_info["removed_source_count"] = sum(
            len(check.removed_sources) for check in artifacts.research_checks
        )
        run_info["cut_short_checks"] = sum(
            1 for check in artifacts.research_checks if check.finding is not None and check.finding.stop_reason
        )
        run_info["grounding"] = summarize_grounding(run.report.sources)
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
        if row is None:
            continue
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
        abort,
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
    missing = missing_environment(args)
    if missing:
        print(missing, file=sys.stderr)
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
    if args.claim_date_field:
        # An evidence cutoff that cannot be applied stops the batch before it spends anything.
        undated = [
            row_index
            for row_index, record in indexed_records
            if (value := str(record.get(args.claim_date_field) or "").strip()) and normalize_claim_date(value) is None
        ]
        if undated:
            print(
                f"Unparseable claim date in field {args.claim_date_field!r} of {len(undated)} row(s), first at row "
                f"{undated[0]}: expected YYYY-MM-DD or D-M-YYYY.",
                file=sys.stderr,
            )
            return 2

    args.image_dir = Path(args.image_dir) if args.image_dir else input_path.parent
    # A missing or unusable image stops the batch before it spends anything.
    for row_index, record in indexed_records:
        try:
            for source in image_sources(record, row_index, args.image_field, args.image_dir):
                if not source.startswith(("http://", "https://")) and not Path(source).is_file():
                    raise ImageError(f"Input row {row_index}: image file not found: {source}")
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2

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
        service = _build_service(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    try:
        submission_rows, run_info_rows, abort = await _run_batch(
            indexed_records=indexed_records, args=args, service=service
        )
    except Exception as exc:
        print(f"Batch run failed: {exc}", file=sys.stderr)
        return 1
    if abort:
        print(
            f"[error] Batch aborted after {len(submission_rows)} of {len(indexed_records)} claims: {abort[1]}. "
            "The remaining claims were not given fallback labels. "
            "Completed rows are written to the output file; fix the cause and rerun with --resume.",
            file=sys.stderr,
            flush=True,
        )

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
            "service_tier": getattr(args, "service_tier", None),
            "model_retries": args.model_retries,
            "base_url": args.base_url or os.getenv("OPENAI_API_BASE_URL"),
            "search_provider": args.search_provider,
            "search_context_size": args.search_context_size,
            "search_results": args.search_results_per_query,
            "tool_call_budget": args.tool_call_budget,
            "search_cache_dir": args.search_cache_dir,
            "max_checks": args.max_checks,
            "parallel_research": args.parallel,
            "parallel_claims": args.parallel_claims,
            "block_fact_checkers": args.block_fact_checkers,
            "blocked_domains": args.blocked_domains or [],
            "claim_date_field": args.claim_date_field,
            "knowledge_store_dir": args.knowledge_store_dir,
            "image_field": args.image_field,
            "image_dir": str(args.image_dir),
            "artifacts_dir": args.artifacts_dir,
        },
        "claims": run_info_rows,
    }
    if abort:
        manifest["aborted"] = abort[1]
    if args.resume and run_info_path.is_file():
        previous = json.loads(run_info_path.read_text(encoding="utf-8"))
        manifest["claims"] = [*previous.get("claims", []), *run_info_rows]
    run_info_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[info] Wrote run manifest to {run_info_path}.", file=sys.stderr, flush=True)
    return abort[0] if abort else 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
