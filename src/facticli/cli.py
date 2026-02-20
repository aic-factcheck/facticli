from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

from .orchestrator import FactCheckOrchestrator, OrchestratorConfig
from .render import format_run_text
from .skills import list_skills


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="facticli",
        description="Agentic fact-checking CLI with pluggable inference providers.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser("check", help="Fact-check a claim.")
    check_parser.add_argument("claim", help="Claim text to verify.")
    check_parser.add_argument(
        "--inference-provider",
        choices=["openai-agents", "gemini"],
        default=os.getenv("FACTICLI_INFERENCE_PROVIDER", "openai-agents"),
        help=(
            "Inference backend (default: FACTICLI_INFERENCE_PROVIDER or openai-agents). "
            "openai-agents uses OpenAI Agents SDK, gemini uses google genai.Client."
        ),
    )
    check_parser.add_argument(
        "--model",
        default=os.getenv("FACTICLI_MODEL", "gpt-4.1-mini"),
        help="Model used when --inference-provider openai-agents.",
    )
    check_parser.add_argument(
        "--gemini-model",
        default=os.getenv("FACTICLI_GEMINI_MODEL", "gemini-3-pro"),
        help="Model used when --inference-provider gemini.",
    )
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

    subparsers.add_parser("skills", help="List built-in agent skills.")
    return parser


async def run_check_command(args: argparse.Namespace) -> int:
    if args.inference_provider == "openai-agents" and not os.getenv("OPENAI_API_KEY"):
        print(
            "OPENAI_API_KEY is not set. Export it or use --inference-provider gemini.",
            file=sys.stderr,
        )
        return 2

    if args.inference_provider == "gemini" and not os.getenv("GEMINI_API_KEY"):
        print(
            "GEMINI_API_KEY is not set. Export it or use --inference-provider openai-agents.",
            file=sys.stderr,
        )
        return 2

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
    run = await orchestrator.check_claim(args.claim)

    if args.json:
        payload: dict[str, object] = {"report": run.report.model_dump()}
        if args.include_artifacts:
            payload["plan"] = run.plan.model_dump()
            payload["findings"] = [finding.model_dump() for finding in run.findings]
        print(json.dumps(payload, indent=2))
    else:
        print(format_run_text(run, show_plan=args.show_plan))

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
    if args.command == "skills":
        return run_skills_command()

    parser.print_help()
    return 1
