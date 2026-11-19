from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import traceback
from pathlib import Path

from .application.config import ClaimExtractionRuntimeConfig
from .application.factory import build_claim_extraction_service, build_fact_check_service
from .application.progress import ProgressEvent
from .application.repository import FileRunArtifactRepository
from .application.settings_file import PROFILE_ENV_VAR, apply_settings, find_config_path, load_settings
from .check_options import add_fact_check_args, add_inference_args, fact_check_config, missing_environment, stage_routing
from .cli_validators import positive_int
from .render import format_run_text
from .skills import get_skill, list_skills, load_skill_prompt


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
    if event.kind == "single_agent_started":
        return ["[progress] Running single-agent baseline (no planner/review/judge split)..."]
    if event.kind == "research_started":
        return [f"[progress] Running research for {payload.get('check_count', 0)} check(s)..."]
    if event.kind == "research_check_completed":
        aspect_id = payload.get("aspect_id", "")
        signal = payload.get("signal", "")
        confidence = float(payload.get("confidence", 0.0))
        summary = _truncate_text(str(payload.get("summary", "")))
        stop_reason = payload.get("stop_reason")
        suffix = f" | cut short: {stop_reason}" if stop_reason else ""
        return [
            f"[progress] [{aspect_id}] {signal} | confidence {confidence:.2f}{suffix}",
            f"           {summary}",
        ]
    if event.kind == "research_check_failed":
        aspect_id = payload.get("aspect_id", "")
        status = payload.get("status", "failed")
        error = payload.get("error", "")
        return [f"[progress] [{aspect_id}] {status}: {error}"]
    if event.kind == "judging_started":
        return ["[progress] Synthesizing final verdict..."]
    if event.kind == "judging_completed":
        verdict = payload.get("verdict", "")
        confidence = float(payload.get("verdict_confidence", 0.0))
        return [f"[progress] Verdict draft: {verdict} (confidence {confidence:.2f})"]
    if event.kind == "review_started":
        round_index = int(payload.get("round_index", 0))
        return [f"[progress] Reviewing evidence for follow-up round {round_index}..."]
    if event.kind == "review_completed":
        round_index = int(payload.get("round_index", 0))
        action = payload.get("action", "")
        follow_up_count = int(payload.get("follow_up_count", 0))
        retry_count = int(payload.get("retry_count", 0))
        gap_count = int(payload.get("gap_count", 0))
        return [
            (
                f"[progress] Review round {round_index}: {action} "
                f"(gaps {gap_count}, retries {retry_count}, new checks {follow_up_count})"
            )
        ]
    if event.kind == "feedback_round_started":
        round_index = int(payload.get("round_index", 0))
        check_count = int(payload.get("check_count", 0))
        return [f"[progress] Running follow-up research round {round_index} for {check_count} check(s)..."]
    if event.kind == "feedback_round_skipped":
        round_index = int(payload.get("round_index", 0))
        return [f"[progress] Follow-up round {round_index} skipped: {payload.get('reason', '')}"]
    if event.kind == "feedback_round_completed":
        round_index = int(payload.get("round_index", 0))
        return [f"[progress] Follow-up research round {round_index} completed."]
    if event.kind == "citation_check_started":
        return [f"[progress] Checking {payload.get('source_count', 0)} cited URL(s)..."]
    if event.kind == "citation_check_completed":
        statuses = payload.get("statuses", {})
        summary = ", ".join(f"{key}={value}" for key, value in sorted(statuses.items())) if isinstance(statuses, dict) else ""
        return [f"[progress] Citation check: {summary}"]
    if event.kind == "run_completed":
        budget = payload.get("budget")
        if isinstance(budget, dict) and budget.get("token_budget_exhausted"):
            skipped = budget.get("skipped_stages") or []
            if skipped:
                return [f"[progress] Fact-check run completed (token budget exhausted; skipped: {', '.join(skipped)})."]
            return ["[progress] Fact-check run completed (token budget exhausted after the last stage)."]
        return ["[progress] Fact-check run completed."]
    return []


def _build_progress_callback(stream_progress: bool):
    if not stream_progress:
        return None

    def callback(event: ProgressEvent) -> None:
        for line in _format_progress_event(event):
            print(line, file=sys.stderr, flush=True)

    return callback


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="facticli",
        description="Agentic fact-checking CLI for OpenAI-compatible inference APIs.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Print full stack traces on errors instead of one-line messages.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "TOML config file with [defaults] and [profiles.<name>] tables whose keys mirror the long "
            "flag names (default: FACTICLI_CONFIG, ./facticli.toml, ./.facticli.toml, "
            "~/.config/facticli/config.toml)."
        ),
    )
    parser.add_argument(
        "--profile",
        default=None,
        help=f"Profile from the config file to apply (default: {PROFILE_ENV_VAR}). Explicit flags win.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser("check", help="Fact-check a claim.")
    check_parser.add_argument("claim", help="Claim text to verify.")
    add_fact_check_args(check_parser, search_providers=("openai", "brave", "exa"))
    check_parser.add_argument(
        "--show-plan",
        action="store_true",
        help="Print the generated verification plan in text mode.",
    )
    check_parser.add_argument(
        "--show-usage",
        action="store_true",
        help="Print token usage, cache hit rate, budget, and per-stage stats in text mode.",
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
    add_inference_args(extract_parser)
    extract_parser.add_argument(
        "--max-claims",
        type=positive_int,
        default=12,
        help="Maximum number of extracted claims.",
    )
    extract_parser.add_argument(
        "--json",
        action="store_true",
        help="Return machine-readable JSON output.",
    )

    skills_parser = subparsers.add_parser("skills", help="List built-in agent skills.")
    skills_parser.add_argument(
        "--show",
        default=None,
        metavar="NAME",
        help="Print the full prompt of one skill instead of the listing.",
    )
    return parser


async def run_check_command(args: argparse.Namespace) -> int:
    missing = missing_environment(args)
    if missing:
        print(missing, file=sys.stderr)
        return 2
    artifact_repository = FileRunArtifactRepository(output_dir=args.artifacts_dir) if args.artifacts_dir else None
    try:
        service = build_fact_check_service(config=fact_check_config(args), artifact_repository=artifact_repository)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    progress_callback = _build_progress_callback(args.stream_progress)
    try:
        run = await service.check_claim(args.claim, progress_callback=progress_callback)
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
            payload["artifacts"] = run.artifacts.model_dump()
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(
            format_run_text(
                run,
                show_plan=args.show_plan,
                show_usage=args.show_usage,
            )
        )

    return 0


def _load_extract_input_text(args: argparse.Namespace) -> str:
    if args.from_file and args.text:
        raise ValueError("Provide input text either as positional argument or with --from-file, not both.")

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
    missing = missing_environment(args)
    if missing:
        print(missing, file=sys.stderr)
        return 2

    try:
        input_text = _load_extract_input_text(args)
        stage_models, stage_efforts = stage_routing(args)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2

    extraction_service = build_claim_extraction_service(
        ClaimExtractionRuntimeConfig(
            model=args.model,
            base_url=args.base_url,
            stage_models=stage_models,
            stage_efforts=stage_efforts,
            model_retry_attempts=args.model_retries,
            retry_budget=args.retry_budget,
            service_tier=getattr(args, "service_tier", None),
            max_claims=args.max_claims,
        )
    )
    try:
        result = await extraction_service.extract_claims(input_text)
    except Exception as exc:
        if getattr(args, "debug", False):
            traceback.print_exc(file=sys.stderr)
        else:
            print(f"Claim extraction failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))
        return 0

    print("Input")
    print(f"  {result.input_text}")
    if result.detected_language:
        print("")
        print("Detected Language")
        print(f"  {result.detected_language}")
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


def run_skills_command(show: str | None = None) -> int:
    if show:
        try:
            skill = get_skill(show)
        except KeyError:
            print(f"Unknown skill: {show}. Run `facticli skills` to list skills.", file=sys.stderr)
            return 2
        print(f"# {skill.name}: {skill.description}")
        print(f"# output: {skill.output_model.__name__} | web_search={'yes' if skill.uses_web_search else 'no'}")
        print("")
        print(load_skill_prompt(skill.name))
        return 0
    for skill in list_skills():
        web = "yes" if skill.uses_web_search else "no"
        print(f"- {skill.name}: {skill.description} | web_search={web}")
    return 0


def _apply_config_file(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Layer config-file settings under explicit CLI flags; return an exit code (0 = ok)."""
    try:
        path = find_config_path(getattr(args, "config", None))
        if path is None:
            return 0
        profile = getattr(args, "profile", None) or os.getenv(PROFILE_ENV_VAR) or None
        settings = load_settings(path, profile)
        apply_settings(args, settings, parser)
    except (OSError, ValueError) as exc:  # tomllib.TOMLDecodeError is a ValueError
        print(f"Config error: {exc}", file=sys.stderr)
        return 2
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    config_code = _apply_config_file(parser, args)
    if config_code:
        sys.exit(config_code)

    if args.command == "check":
        sys.exit(asyncio.run(run_check_command(args)))
    if args.command == "extract-claims":
        sys.exit(asyncio.run(run_extract_claims_command(args)))
    if args.command == "skills":
        sys.exit(run_skills_command(show=getattr(args, "show", None)))

    parser.print_help()
    sys.exit(1)
