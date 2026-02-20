from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import traceback
from pathlib import Path

from .application.progress import ProgressEvent
from .claim_extraction import ClaimExtractor, ClaimExtractorConfig
from .core.artifacts import RunArtifacts
from .orchestrator import FactCheckOrchestrator, OrchestratorConfig
from .render import format_run_text
from .skills import list_skills


def _positive_int(raw_value: str) -> int:
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected integer, got: {raw_value!r}") from exc
    if value < 1:
        raise argparse.ArgumentTypeError(f"Value must be >= 1, got: {value}")
    return value


def _bounded_int(raw_value: str, *, minimum: int, maximum: int) -> int:
    value = _positive_int(raw_value)
    if value < minimum or value > maximum:
        raise argparse.ArgumentTypeError(
            f"Value must be between {minimum} and {maximum}, got: {value}"
        )
    return value


def _search_results_int(raw_value: str) -> int:
    return _bounded_int(raw_value, minimum=1, maximum=20)


def _add_inference_provider_args(command_parser: argparse.ArgumentParser) -> None:
    command_parser.add_argument(
        "--inference-provider",
        choices=["openai-agents", "gemini"],
        default=os.getenv("FACTICLI_INFERENCE_PROVIDER", "openai-agents"),
        help=(
            "Inference backend (default: FACTICLI_INFERENCE_PROVIDER or openai-agents). "
            "openai-agents uses OpenAI Agents SDK, gemini uses google genai.Client."
        ),
    )
    command_parser.add_argument(
        "--model",
        default=os.getenv("FACTICLI_MODEL", "gpt-4.1-mini"),
        help="Model used when --inference-provider openai-agents.",
    )
    command_parser.add_argument(
        "--gemini-model",
        default=os.getenv("FACTICLI_GEMINI_MODEL", "gemini-2.0-flash"),
        help="Model used when --inference-provider gemini.",
    )


def _serialize_run_artifacts(artifacts: RunArtifacts) -> dict[str, object]:
    return {
        "claim": artifacts.claim,
        "normalized_claim": artifacts.normalized_claim,
        "plan_raw": artifacts.plan_raw.model_dump() if artifacts.plan_raw else None,
        "plan_normalized": artifacts.plan_normalized.model_dump() if artifacts.plan_normalized else None,
        "research_checks": [
            {
                "check": check.check.model_dump(),
                "attempts": check.attempts,
                "errors": list(check.errors),
                "finding": check.finding.model_dump() if check.finding else None,
            }
            for check in artifacts.research_checks
        ],
        "report_raw": artifacts.report_raw.model_dump() if artifacts.report_raw else None,
        "report_final": artifacts.report_final.model_dump() if artifacts.report_final else None,
    }


def _truncate_text(value: str, max_length: int = 140) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= max_length:
        return normalized
    return normalized[: max_length - 3] + "..."


def _format_progress_event(event: ProgressEvent) -> list[str]:
    payload = event.payload
    if event.kind == "run_started":
        return [f"[progress] Starting fact-check: {payload.get('claim', '')}"]
    if event.kind == "planning_started":
        return ["[progress] Planning verification checks..."]
    if event.kind == "planning_completed":
        lines = [f"[progress] Plan ready with {payload.get('check_count', 0)} check(s):"]
        checks = payload.get("checks", [])
        if isinstance(checks, list):
            for check in checks:
                if not isinstance(check, dict):
                    continue
                aspect_id = check.get("aspect_id", "")
                question = check.get("question", "")
                lines.append(f"  - [{aspect_id}] {question}")
        return lines
    if event.kind == "research_started":
        return [f"[progress] Running research for {payload.get('check_count', 0)} check(s)..."]
    if event.kind == "research_check_completed":
        aspect_id = payload.get("aspect_id", "")
        signal = payload.get("signal", "")
        confidence = float(payload.get("confidence", 0.0))
        summary = _truncate_text(str(payload.get("summary", "")))
        return [
            f"[progress] [{aspect_id}] {signal} | confidence {confidence:.2f}",
            f"           {summary}",
        ]
    if event.kind == "research_check_failed":
        aspect_id = payload.get("aspect_id", "")
        error = payload.get("error", "")
        return [f"[progress] [{aspect_id}] failed: {error}"]
    if event.kind == "judging_started":
        return ["[progress] Synthesizing final verdict..."]
    if event.kind == "judging_completed":
        verdict = payload.get("verdict", "")
        confidence = float(payload.get("verdict_confidence", 0.0))
        return [f"[progress] Verdict draft: {verdict} (confidence {confidence:.2f})"]
    if event.kind == "run_completed":
        return ["[progress] Fact-check run completed."]
    return []


def _build_progress_callback(stream_progress: bool):
    if not stream_progress:
        return None

    def callback(event: ProgressEvent) -> None:
        for line in _format_progress_event(event):
            print(line, file=sys.stderr, flush=True)

    return callback


def _validate_inference_provider_keys(inference_provider: str) -> int:
    if inference_provider == "openai-agents" and not os.getenv("OPENAI_API_KEY"):
        print(
            "OPENAI_API_KEY is not set. Export it or use --inference-provider gemini.",
            file=sys.stderr,
        )
        return 2

    if inference_provider == "gemini" and not os.getenv("GEMINI_API_KEY"):
        print(
            "GEMINI_API_KEY is not set. Export it or use --inference-provider openai-agents.",
            file=sys.stderr,
        )
        return 2

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="facticli",
        description="Agentic fact-checking CLI with pluggable inference providers.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Print full stack traces on errors instead of one-line messages.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser("check", help="Fact-check a claim.")
    check_parser.add_argument("claim", help="Claim text to verify.")
    _add_inference_provider_args(check_parser)
    check_parser.add_argument(
        "--max-checks",
        type=_positive_int,
        default=4,
        help="Maximum number of verification sub-checks.",
    )
    check_parser.add_argument(
        "--parallel",
        type=_positive_int,
        default=4,
        help="Maximum parallel research workers.",
    )
    check_parser.add_argument(
        "--search-provider",
        choices=["openai", "brave"],
        default=os.getenv("FACTICLI_SEARCH_PROVIDER", "openai"),
        help="Search backend for research stage (default: FACTICLI_SEARCH_PROVIDER or openai).",
    )
    check_parser.add_argument(
        "--search-context-size",
        choices=["low", "medium", "high"],
        default="high",
        help="Context size for hosted OpenAI web search tool (used when --search-provider openai).",
    )
    check_parser.add_argument(
        "--search-results",
        type=_search_results_int,
        default=5,
        dest="search_results_per_query",
        help="Number of search results to fetch per query (1-20, default 5).",
    )
    check_parser.add_argument(
        "--show-plan",
        action="store_true",
        help="Print the generated verification plan in text mode.",
    )
    check_parser.add_argument(
        "--json",
        action="store_true",
        help="Return machine-readable JSON output.",
    )
    check_parser.add_argument(
        "--include-artifacts",
        action="store_true",
        help="When used with --json, include plan, findings, and run artifacts.",
    )
    check_parser.add_argument(
        "--stream-progress",
        action="store_true",
        help="Stream plan and per-check progress updates to stderr while the run executes.",
    )

    extract_parser = subparsers.add_parser(
        "extract-claims",
        help="Extract decontextualized atomic check-worthy claims from arbitrary text.",
    )
    extract_parser.add_argument(
        "text",
        nargs="?",
        default=None,
        help="Raw input text to extract claims from.",
    )
    extract_parser.add_argument(
        "--from-file",
        dest="from_file",
        default=None,
        help="Path to a UTF-8 text file containing the input text.",
    )
    _add_inference_provider_args(extract_parser)
    extract_parser.add_argument(
        "--max-claims",
        type=_positive_int,
        default=12,
        help="Maximum number of extracted claims.",
    )
    extract_parser.add_argument(
        "--json",
        action="store_true",
        help="Return machine-readable JSON output.",
    )

    subparsers.add_parser("skills", help="List built-in agent skills.")
    return parser


async def run_check_command(args: argparse.Namespace) -> int:
    provider_validation_code = _validate_inference_provider_keys(args.inference_provider)
    if provider_validation_code:
        return provider_validation_code

    config = OrchestratorConfig(
        inference_provider=args.inference_provider,
        model=args.model,
        gemini_model=args.gemini_model,
        max_checks=args.max_checks,
        max_parallel_research=args.parallel,
        search_context_size=args.search_context_size,
        search_provider=args.search_provider,
        search_results_per_query=args.search_results_per_query,
    )

    if args.inference_provider == "gemini" and args.search_provider != "brave":
        print(
            "Gemini inference currently requires --search-provider brave.",
            file=sys.stderr,
        )
        return 2

    if args.search_provider == "brave" and not os.getenv("BRAVE_SEARCH_API_KEY"):
        print(
            "BRAVE_SEARCH_API_KEY is not set. Export it or switch to --search-provider openai.",
            file=sys.stderr,
        )
        return 2

    orchestrator = FactCheckOrchestrator(config=config)
    stream_progress = bool(getattr(args, "stream_progress", False))
    progress_callback = _build_progress_callback(stream_progress)
    try:
        run = await orchestrator.check_claim(args.claim, progress_callback=progress_callback)
    except Exception as exc:
        if getattr(args, "debug", False):
            traceback.print_exc(file=sys.stderr)
        else:
            print(f"Fact-check failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        payload: dict[str, object] = {"report": run.report.model_dump()}
        if args.include_artifacts:
            payload["plan"] = run.plan.model_dump()
            payload["findings"] = [finding.model_dump() for finding in run.findings]
            payload["artifacts"] = _serialize_run_artifacts(run.artifacts)
        print(json.dumps(payload, indent=2))
    else:
        print(format_run_text(run, show_plan=args.show_plan))

    return 0


def _load_extract_input_text(args: argparse.Namespace) -> str:
    if args.from_file and args.text:
        raise ValueError(
            "Provide input text either as positional argument or with --from-file, not both."
        )

    if args.from_file:
        path = Path(args.from_file)
        if not path.exists():
            raise FileNotFoundError(f"Input file does not exist: {path}")
        if not path.is_file():
            raise ValueError(f"Input path is not a file: {path}")
        return path.read_text(encoding="utf-8")

    if args.text:
        return args.text

    raise ValueError("Provide input text as positional argument or use --from-file.")


async def run_extract_claims_command(args: argparse.Namespace) -> int:
    provider_validation_code = _validate_inference_provider_keys(args.inference_provider)
    if provider_validation_code:
        return provider_validation_code

    try:
        input_text = _load_extract_input_text(args)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2

    extractor = ClaimExtractor(
        config=ClaimExtractorConfig(
            inference_provider=args.inference_provider,
            model=args.model,
            gemini_model=args.gemini_model,
            max_claims=args.max_claims,
        )
    )
    try:
        result = await extractor.extract(input_text)
    except Exception as exc:
        if getattr(args, "debug", False):
            traceback.print_exc(file=sys.stderr)
        else:
            print(f"Claim extraction failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result.model_dump(), indent=2))
        return 0

    print("Input")
    print(f"  {result.input_text}")
    print("")
    print("Claims")
    if not result.claims:
        print("  - no check-worthy claims extracted")
    for claim in result.claims:
        print(f"  - [{claim.claim_id}] {claim.claim_text}")
        print(f"    source: {claim.source_fragment}")
        print(f"    reason: {claim.checkworthy_reason}")

    if result.coverage_notes:
        print("")
        print("Coverage Notes")
        for note in result.coverage_notes:
            print(f"  - {note}")

    if result.excluded_nonfactual:
        print("")
        print("Excluded Non-factual")
        for item in result.excluded_nonfactual:
            print(f"  - {item}")

    return 0


def run_skills_command() -> int:
    for skill in list_skills():
        web = "yes" if skill.uses_web_search else "no"
        print(f"- {skill.name}: {skill.description} | web_search={web}")
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "check":
        sys.exit(asyncio.run(run_check_command(args)))
    if args.command == "extract-claims":
        sys.exit(asyncio.run(run_extract_claims_command(args)))
    if args.command == "skills":
        sys.exit(run_skills_command())

    parser.print_help()
    sys.exit(1)
