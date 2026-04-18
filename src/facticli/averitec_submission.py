from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from facticli.adapters import resolve_model_name, resolve_provider_profile
from facticli.application.config import FactCheckRuntimeConfig
from facticli.application.factory import build_fact_check_service
from facticli.cli_validators import non_negative_int, positive_int, search_results_int
from facticli.core.contracts import FactCheckReport, VeracityVerdict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m facticli.averitec_submission",
        description=(
            "Run facticli on Averitec-formatted claims and write an Averitec submission JSON file."
        ),
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
        "--inference-provider",
        choices=["openai", "gemini", "ollama", "openai-agents"],
        default=os.getenv("FACTICLI_INFERENCE_PROVIDER", "openai"),
        help="Inference provider profile (default: FACTICLI_INFERENCE_PROVIDER or openai).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model name override. Defaults to FACTICLI_MODEL (openai), "
            "FACTICLI_GEMINI_MODEL (gemini), or OLLAMA_MODEL (ollama)."
        ),
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("FACTICLI_BASE_URL"),
        help="Optional OpenAI-compatible base URL override.",
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
        choices=["openai", "brave"],
        default=os.getenv("FACTICLI_SEARCH_PROVIDER", "openai"),
        help="Search backend for research stage.",
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
    profile = resolve_provider_profile(args.inference_provider)
    if not os.getenv(profile.api_key_env):
        raise RuntimeError(
            f"{profile.api_key_env} is not set. Export it or change --inference-provider."
        )
    if args.search_provider == "brave" and not os.getenv("BRAVE_SEARCH_API_KEY"):
        raise RuntimeError(
            "BRAVE_SEARCH_API_KEY is not set. Export it or use --search-provider openai."
        )


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
            ""
            if empty_question
            else "What evidence supports the final verdict for this claim?"
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
    records: list[dict[str, Any]],
    offset: int,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    config = FactCheckRuntimeConfig(
        inference_provider=args.inference_provider,
        model=resolve_model_name(args.inference_provider, args.model),
        base_url=args.base_url,
        max_checks=args.max_checks,
        max_parallel_research=args.parallel,
        search_context_size=args.search_context_size,
        search_provider=args.search_provider,
        search_results_per_query=args.search_results_per_query,
    )
    service = build_fact_check_service(config=config)
    semaphore = asyncio.Semaphore(max(1, args.parallel_claims))
    ordered_rows: list[dict[str, Any] | None] = [None] * len(records)

    async def process_one(local_index: int, record: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        row_index = offset + local_index
        claim = _extract_claim_text(record, row_index, args.claim_field)
        async with semaphore:
            try:
                run = await service.check_claim(claim)
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
                return local_index, build_failed_submission_row(
                    record=record,
                    row_index=row_index,
                    claim_field=args.claim_field,
                    claim_id_field=args.claim_id_field,
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
        return local_index, row

    tasks = [
        asyncio.create_task(process_one(local_index, record))
        for local_index, record in enumerate(records)
    ]

    completed = 0
    for task in asyncio.as_completed(tasks):
        local_index, row = await task
        ordered_rows[local_index] = row
        completed += 1
        print(
            f"[progress] Completed {completed}/{len(records)} claims.",
            file=sys.stderr,
            flush=True,
        )

    return [row for row in ordered_rows if row is not None]


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

    print(
        f"[info] Processing {len(sliced_records)} claims from {input_path} "
        f"(offset={offset}, parallel_claims={args.parallel_claims}).",
        file=sys.stderr,
        flush=True,
    )

    try:
        submission_rows = await _run_batch(records=sliced_records, offset=offset, args=args)
    except Exception as exc:
        print(f"Batch run failed: {exc}", file=sys.stderr)
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(submission_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"[info] Wrote {len(submission_rows)} submission rows to {output_path}.",
        file=sys.stderr,
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
