from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from .claim_extraction import ClaimExtractor, ClaimExtractorConfig
from .orchestrator import FactCheckOrchestrator, OrchestratorConfig
from .render import format_run_text
from .skills import list_skills


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
        default=os.getenv("FACTICLI_GEMINI_MODEL", "gemini-3-pro"),
        help="Model used when --inference-provider gemini.",
    )


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
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser("check", help="Fact-check a claim.")
    check_parser.add_argument("claim", help="Claim text to verify.")
    _add_inference_provider_args(check_parser)
    check_parser.add_argument(
        "--max-checks",
        type=int,
        default=4,
        help="Maximum number of verification sub-checks.",
    )
    check_parser.add_argument(
        "--parallel",
        type=int,
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
        help="When used with --json, include plan and per-check findings.",
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
        type=int,
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
        max_checks=max(1, args.max_checks),
        max_parallel_research=max(1, args.parallel),
        search_context_size=args.search_context_size,
        search_provider=args.search_provider,
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
    try:
        run = await orchestrator.check_claim(args.claim)
    except Exception as exc:
        print(f"Fact-check failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        payload: dict[str, object] = {"report": run.report.model_dump()}
        if args.include_artifacts:
            payload["plan"] = run.plan.model_dump()
            payload["findings"] = [finding.model_dump() for finding in run.findings]
        print(json.dumps(payload, indent=2))
    else:
        print(format_run_text(run, show_plan=args.show_plan))

    return 0


def _load_extract_input_text(args: argparse.Namespace) -> str:
    if args.from_file:
        path = Path(args.from_file)
        if not path.exists():
            raise FileNotFoundError(f"Input file does not exist: {path}")
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
            max_claims=max(1, args.max_claims),
        )
    )
    try:
        result = await extractor.extract(input_text)
    except Exception as exc:
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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "check":
        return asyncio.run(run_check_command(args))
    if args.command == "extract-claims":
        return asyncio.run(run_extract_claims_command(args))
    if args.command == "skills":
        return run_skills_command()

    parser.print_help()
    return 1
