from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

from .application.config import (
    REASONING_EFFORTS,
    STAGE_NAMES,
    STRATEGIES,
    ClaimExtractionRuntimeConfig,
    FactCheckRuntimeConfig,
    parse_stage_assignments,
)
from .application.factory import build_claim_extraction_service, build_fact_check_service
from .application.progress import ProgressEvent
from .application.repository import FileRunArtifactRepository
from .application.settings_file import PROFILE_ENV_VAR, apply_settings, find_config_path, load_settings
from .cli_validators import non_negative_int, positive_int, search_results_int
from .render import format_run_text
from .skills import get_skill, list_skills, load_skill_prompt


def _add_inference_args(command_parser: argparse.ArgumentParser) -> None:
    command_parser.add_argument(
        "--model",
        default=None,
        help="Model name override. Falls back to OPENAI_API_MODEL.",
    )
    command_parser.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible base URL override. Falls back to OPENAI_API_BASE_URL.",
    )
    command_parser.add_argument(
        "--stage-model",
        action="append",
        default=None,
        metavar="STAGE=MODEL",
        help=(
            "Route one stage to a different model (repeatable). Stages: "
            + ", ".join(STAGE_NAMES)
            + ". Example: --stage-model research=gpt-5-mini --stage-model judge=gpt-5.4"
        ),
    )
    command_parser.add_argument(
        "--stage-effort",
        action="append",
        default=None,
        metavar="STAGE=EFFORT",
        help=(
            "Reasoning effort per stage for reasoning models (repeatable). Efforts: "
            + ", ".join(REASONING_EFFORTS)
            + ". Example: --stage-effort research=low --stage-effort judge=high"
        ),
    )
    command_parser.add_argument(
        "--model-retries",
        type=non_negative_int,
        default=3,
        help="Max SDK-managed retries per model call for transient errors (default: 3).",
    )
    command_parser.add_argument(
        "--retry-budget",
        type=non_negative_int,
        default=12,
        help="Max model-call retries across the whole run; 0 disables retries (default: 12).",
    )


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
        return [
            f"[progress] [{aspect_id}] {signal} | confidence {confidence:.2f}",
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


def _validate_inference_env(requested_model: str | None) -> int:
    """Validate required OpenAI-compatible inference environment variables."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not (api_key and api_key.strip()):
        print(
            "OPENAI_API_KEY is not set. Export it or add it to .env.",
            file=sys.stderr,
        )
        return 2

    env_model = os.getenv("OPENAI_API_MODEL")
    if not (requested_model and requested_model.strip()) and not (
        env_model and env_model.strip()
    ):
        print(
            "OPENAI_API_MODEL is not set. Export it, add it to .env, or pass --model.",
            file=sys.stderr,
        )
        return 2

    return 0


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
    _add_inference_args(check_parser)
    check_parser.add_argument(
        "--strategy",
        choices=list(STRATEGIES),
        default="pipeline",
        help=(
            "pipeline: plan -> parallel research -> optional review -> judge (default). "
            "single_agent: no-harness baseline, one agent with the search tool."
        ),
    )
    check_parser.add_argument(
        "--max-checks",
        type=positive_int,
        default=4,
        help="Maximum number of verification sub-checks.",
    )
    check_parser.add_argument(
        "--parallel",
        type=positive_int,
        default=4,
        help="Maximum parallel research workers.",
    )
    check_parser.add_argument(
        "--feedback-rounds",
        type=non_negative_int,
        default=0,
        help="Maximum bounded follow-up research rounds after the initial pass (default: 0).",
    )
    check_parser.add_argument(
        "--follow-up-checks",
        type=positive_int,
        default=2,
        help="Maximum new follow-up checks allowed per feedback round (default: 2).",
    )
    check_parser.add_argument(
        "--token-budget",
        type=positive_int,
        default=None,
        help=(
            "Shared token budget for the whole run (all stages). When exhausted, remaining checks are "
            "marked budget_exhausted and follow-up rounds are skipped (default: unlimited)."
        ),
    )
    check_parser.add_argument(
        "--search-provider",
        choices=["openai", "brave", "knowledge_store"],
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
        type=search_results_int,
        default=5,
        dest="search_results_per_query",
        help="Number of search results to fetch per query (1-20, default 5).",
    )
    check_parser.add_argument(
        "--block-fact-checkers",
        action="store_true",
        default=False,
        help="Exclude known fact-checking domains from evidence (label-leakage control for benchmarks).",
    )
    check_parser.add_argument(
        "--blocked-domain",
        action="append",
        default=None,
        dest="blocked_domains",
        metavar="DOMAIN",
        help="Additional domain (or domain/path prefix) to exclude from evidence (repeatable).",
    )
    check_parser.add_argument(
        "--verify-sources",
        action="store_true",
        default=False,
        help="After judging, check each cited URL (HEAD/GET, Wayback fallback) and mark it live/archived/broken.",
    )
    check_parser.add_argument(
        "--artifacts-dir",
        default=None,
        help="Directory to persist the full run artifacts JSON (plan, findings, usage, budget) for audits.",
    )
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
    _add_inference_args(extract_parser)
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


def _arg(args: argparse.Namespace, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def _stage_routing(args: argparse.Namespace) -> tuple[dict[str, str], dict[str, str]]:
    stage_models = parse_stage_assignments(_arg(args, "stage_model", None), option_name="--stage-model")
    stage_efforts = parse_stage_assignments(
        _arg(args, "stage_effort", None),
        allowed_values=REASONING_EFFORTS,
        option_name="--stage-effort",
    )
    return stage_models, stage_efforts


async def run_check_command(args: argparse.Namespace) -> int:
    inference_validation_code = _validate_inference_env(args.model)
    if inference_validation_code:
        return inference_validation_code

    try:
        stage_models, stage_efforts = _stage_routing(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    retry_budget = _arg(args, "retry_budget", 12)
    config = FactCheckRuntimeConfig(
        model=args.model,
        base_url=args.base_url,
        stage_models=stage_models,
        stage_efforts=stage_efforts,
        model_retry_attempts=int(_arg(args, "model_retries", 3)),
        retry_budget=int(retry_budget) if retry_budget is not None else None,
        strategy=_arg(args, "strategy", "pipeline"),
        max_checks=args.max_checks,
        max_parallel_research=args.parallel,
        max_feedback_rounds=args.feedback_rounds,
        max_follow_up_checks=args.follow_up_checks,
        search_context_size=args.search_context_size,
        search_provider=args.search_provider,
        search_results_per_query=args.search_results_per_query,
        token_budget=_arg(args, "token_budget", None),
        verify_sources=bool(_arg(args, "verify_sources", False)),
        block_fact_checkers=bool(_arg(args, "block_fact_checkers", False)),
        blocked_domains=tuple(_arg(args, "blocked_domains", None) or ()),
    )

    if args.search_provider == "brave" and not os.getenv("BRAVE_SEARCH_API_KEY"):
        print(
            "BRAVE_SEARCH_API_KEY is not set. Export it or switch to --search-provider openai.",
            file=sys.stderr,
        )
        return 2

    artifacts_dir = _arg(args, "artifacts_dir", None)
    artifact_repository = FileRunArtifactRepository(output_dir=artifacts_dir) if artifacts_dir else None
    try:
        service = build_fact_check_service(config=config, artifact_repository=artifact_repository)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    stream_progress = bool(_arg(args, "stream_progress", False))
    progress_callback = _build_progress_callback(stream_progress)
    try:
        run = await service.check_claim(args.claim, progress_callback=progress_callback)
    except Exception as exc:
        if _arg(args, "debug", False):
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
                show_plan=bool(_arg(args, "show_plan", False)),
                show_usage=bool(_arg(args, "show_usage", False)),
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
    inference_validation_code = _validate_inference_env(args.model)
    if inference_validation_code:
        return inference_validation_code

    try:
        input_text = _load_extract_input_text(args)
        stage_models, stage_efforts = _stage_routing(args)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2

    retry_budget = _arg(args, "retry_budget", 12)
    extraction_service = build_claim_extraction_service(
        ClaimExtractionRuntimeConfig(
            model=args.model,
            base_url=args.base_url,
            stage_models=stage_models,
            stage_efforts=stage_efforts,
            model_retry_attempts=int(_arg(args, "model_retries", 3)),
            retry_budget=int(retry_budget) if retry_budget is not None else None,
            max_claims=args.max_claims,
        )
    )
    try:
        result = await extraction_service.extract_claims(input_text)
    except Exception as exc:
        if _arg(args, "debug", False):
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
    except (FileNotFoundError, ValueError) as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # tomllib.TOMLDecodeError and friends
        print(f"Config error: failed to parse config file: {exc}", file=sys.stderr)
        return 2
    apply_settings(args, settings, parser)
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
