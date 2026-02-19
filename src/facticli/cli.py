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
        description="Agentic fact-checking CLI built on openai-agents.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser("check", help="Fact-check a claim.")
    check_parser.add_argument("claim", help="Claim text to verify.")
    check_parser.add_argument(
        "--model",
        default=os.getenv("FACTICLI_MODEL", "gpt-4.1-mini"),
        help="Model name passed to openai-agents (default: FACTICLI_MODEL or gpt-4.1-mini).",
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
        "--search-context-size",
        choices=["low", "medium", "high"],
        default="high",
        help="Web search context size for hosted web search tool.",
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
    if not os.getenv("OPENAI_API_KEY"):
        print(
            "OPENAI_API_KEY is not set. Export it before running facticli.",
            file=sys.stderr,
        )
        return 2

    config = OrchestratorConfig(
        model=args.model,
        max_checks=max(1, args.max_checks),
        max_parallel_research=max(1, args.parallel),
        search_context_size=args.search_context_size,
    )
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

