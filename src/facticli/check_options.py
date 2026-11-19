"""Flags, environment checks, and runtime config shared by `facticli check` and the AVeriTeC runner."""

from __future__ import annotations

import argparse
import os
from typing import Any

from .application.config import (
    REASONING_EFFORTS,
    STAGE_NAMES,
    STRATEGIES,
    FactCheckRuntimeConfig,
    parse_stage_assignments,
)
from .cli_validators import non_negative_int, positive_int, search_results_int


def add_inference_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=None, help="Model name override. Falls back to OPENAI_API_MODEL.")
    parser.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible base URL override. Falls back to OPENAI_API_BASE_URL.",
    )
    parser.add_argument(
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
    parser.add_argument(
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
        help="Max model-call retries across the whole run; 0 disables retries (default: 12).",
    )


def add_fact_check_args(parser: argparse.ArgumentParser, *, search_providers: tuple[str, ...]) -> None:
    """Flags that configure one fact-check run (one claim)."""
    add_inference_args(parser)
    parser.add_argument(
        "--strategy",
        choices=list(STRATEGIES),
        default="pipeline",
        help=(
            "pipeline: plan -> parallel research -> optional review -> judge (default). "
            "single_agent: no-harness baseline, one agent with the search tool."
        ),
    )
    parser.add_argument("--max-checks", type=positive_int, default=4, help="Maximum number of verification sub-checks.")
    parser.add_argument("--parallel", type=positive_int, default=4, help="Maximum parallel research workers.")
    parser.add_argument(
        "--feedback-rounds",
        type=non_negative_int,
        default=0,
        help="Maximum bounded follow-up research rounds after the initial pass (default: 0).",
    )
    parser.add_argument(
        "--follow-up-checks",
        type=positive_int,
        default=2,
        help="Maximum new follow-up checks allowed per feedback round (default: 2).",
    )
    parser.add_argument(
        "--token-budget",
        type=positive_int,
        default=None,
        help=(
            "Shared token budget for the whole run (all stages). When exhausted, remaining checks are "
            "marked budget_exhausted and follow-up rounds are skipped (default: unlimited)."
        ),
    )
    parser.add_argument(
        "--search-provider",
        choices=list(search_providers),
        default=os.getenv("FACTICLI_SEARCH_PROVIDER", "openai"),
        help="Search backend for research stage (default: FACTICLI_SEARCH_PROVIDER or openai).",
    )
    parser.add_argument(
        "--search-context-size",
        choices=["low", "medium", "high"],
        default="high",
        help="Context size for hosted OpenAI web search tool (used when --search-provider openai).",
    )
    parser.add_argument(
        "--search-results",
        type=search_results_int,
        default=5,
        dest="search_results_per_query",
        help=(
            "Results per query when the model does not ask for a number (1-20, default 5); "
            "used by the brave and knowledge_store tools."
        ),
    )
    parser.add_argument(
        "--tool-call-budget",
        type=positive_int,
        default=8,
        help=(
            "Searches a researcher is told to plan for per check (default: 8). The brave and knowledge_store "
            "tools refuse calls beyond twice this number."
        ),
    )
    parser.add_argument(
        "--search-cache-dir",
        default=None,
        help=(
            "Directory that records Brave responses and replays them on identical requests, so repeated or "
            "paired runs see the same retrieval (needs --search-provider brave)."
        ),
    )
    parser.add_argument(
        "--block-fact-checkers",
        action="store_true",
        default=False,
        help="Exclude known fact-checking domains from evidence (label-leakage control for benchmarks).",
    )
    parser.add_argument(
        "--blocked-domain",
        action="append",
        default=None,
        dest="blocked_domains",
        metavar="DOMAIN",
        help="Additional domain (or domain/path prefix) to exclude from evidence (repeatable).",
    )
    parser.add_argument(
        "--verify-sources",
        action="store_true",
        default=False,
        help="After judging, check each cited URL (HEAD/GET, Wayback fallback) and mark it live/archived/broken.",
    )
    parser.add_argument(
        "--artifacts-dir",
        default=None,
        help="Directory to persist the full run artifacts JSON (plan, findings, usage, budget) for audits.",
    )


def missing_environment(args: argparse.Namespace) -> str | None:
    """Message naming the first required setting that is missing, or None."""
    if not (os.getenv("OPENAI_API_KEY") or "").strip():
        return "OPENAI_API_KEY is not set. Export it or add it to .env."
    if not (args.model or "").strip() and not (os.getenv("OPENAI_API_MODEL") or "").strip():
        return "OPENAI_API_MODEL is not set. Export it, add it to .env, or pass --model."
    provider = getattr(args, "search_provider", None)
    if provider == "brave" and not os.getenv("BRAVE_SEARCH_API_KEY"):
        return "BRAVE_SEARCH_API_KEY is not set. Export it or switch to --search-provider openai."
    if provider == "knowledge_store" and not getattr(args, "knowledge_store_dir", None):
        # A choices list does not cover a FACTICLI_SEARCH_PROVIDER default, so `check` gets here too.
        return (
            "--search-provider knowledge_store needs a store directory and claim ids: use "
            "python -m facticli.averitec_submission --knowledge-store-dir DIR."
        )
    return None


def stage_routing(args: argparse.Namespace) -> tuple[dict[str, str], dict[str, str]]:
    """Parse --stage-model / --stage-effort; raises ValueError on a malformed entry."""
    stage_models = parse_stage_assignments(args.stage_model, option_name="--stage-model")
    stage_efforts = parse_stage_assignments(
        args.stage_effort, allowed_values=REASONING_EFFORTS, option_name="--stage-effort"
    )
    return stage_models, stage_efforts


def fact_check_config(args: argparse.Namespace, **extra: Any) -> FactCheckRuntimeConfig:
    """Runtime config from parsed fact-check flags; raises ValueError on a malformed stage entry."""
    stage_models, stage_efforts = stage_routing(args)
    return FactCheckRuntimeConfig(
        model=args.model,
        base_url=args.base_url,
        stage_models=stage_models,
        stage_efforts=stage_efforts,
        model_retry_attempts=args.model_retries,
        retry_budget=args.retry_budget,
        strategy=args.strategy,
        max_checks=args.max_checks,
        max_parallel_research=args.parallel,
        max_feedback_rounds=args.feedback_rounds,
        max_follow_up_checks=args.follow_up_checks,
        search_context_size=args.search_context_size,
        search_provider=args.search_provider,
        search_results_per_query=args.search_results_per_query,
        token_budget=args.token_budget,
        verify_sources=args.verify_sources,
        block_fact_checkers=args.block_fact_checkers,
        blocked_domains=tuple(args.blocked_domains or ()),
        tool_call_budget=args.tool_call_budget,
        search_cache_dir=args.search_cache_dir,
        **extra,
    )
