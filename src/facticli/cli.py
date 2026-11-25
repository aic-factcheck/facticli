from __future__ import annotations

import argparse
import asyncio
import json
import sys
import traceback
from pathlib import Path

from .check_options import (
    add_fact_check_args,
    add_inference_args,
    fact_check_config,
    missing_environment,
    resolve_reverse_image_search,
    stage_routing,
)
from .cli_validators import positive_int
from .config import ClaimExtractionRuntimeConfig
from .core.images import ClaimImage, ImageError, load_images
from .pipeline import build_claim_extractor, build_fact_checker
from .render import format_run_text
from .skills import get_skill, list_skills, load_skill_prompt


def _add_image_arg(parser: argparse.ArgumentParser, what: str) -> None:
    parser.add_argument(
        "--image",
        action="append",
        default=None,
        dest="images",
        metavar="PATH_OR_URL",
        help=f"{what} (repeatable): a local file or an http(s) URL; JPEG, PNG, WebP, or GIF up to 20 MB.",
    )


def _load_cli_images(args: argparse.Namespace) -> list[ClaimImage]:
    """Load --image values (raises ImageError) and settle whether reverse image search runs."""
    images = load_images(args.images or [], url_base=getattr(args, "image_url_base", None))
    if hasattr(args, "reverse_image_search"):
        warning = resolve_reverse_image_search(args, searchable_images=any(image.public_url for image in images))
        if warning:
            print(warning, file=sys.stderr)
    return images


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
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser("check", help="Fact-check a claim.")
    check_parser.add_argument("claim", help="Claim text to verify.")
    _add_image_arg(check_parser, "Image that is part of the claim")
    add_fact_check_args(check_parser, search_providers=("openai", "brave", "exa"))
    check_parser.add_argument(
        "--show-plan",
        action="store_true",
        help="Print the generated verification plan in text mode.",
    )
    check_parser.add_argument(
        "--show-usage",
        action="store_true",
        help="Print token usage, cache hit rate, and per-stage stats in text mode.",
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
    _add_image_arg(extract_parser, "Image that comes with the input text, e.g. a screenshot of a post")
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
    try:
        images = _load_cli_images(args)  # before the environment check: images decide the image-search keys
    except ValueError as exc:  # ImageError is a ValueError
        print(str(exc), file=sys.stderr)
        return 2
    missing = missing_environment(args)
    if missing:
        print(missing, file=sys.stderr)
        return 2
    try:
        checker = build_fact_checker(fact_check_config(args), artifacts_dir=args.artifacts_dir)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    progress = (lambda line: print(line, file=sys.stderr, flush=True)) if args.stream_progress else None
    try:
        run = await checker.check_claim(args.claim, progress=progress, images=images)
    except Exception as exc:
        if getattr(args, "debug", False):
            traceback.print_exc(file=sys.stderr)
        else:
            print(f"Fact-check failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        payload: dict[str, object] = {"report": run.report.model_dump()}
        if images:
            payload["images"] = [image.model_dump() for image in images]
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


def _load_extract_input_text(args: argparse.Namespace, *, allow_empty: bool = False) -> str:
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
    if allow_empty:  # images alone are an input
        return ""

    raise ValueError("Provide input text as positional argument, use --from-file, or pass --image.")


async def run_extract_claims_command(args: argparse.Namespace) -> int:
    missing = missing_environment(args)
    if missing:
        print(missing, file=sys.stderr)
        return 2

    try:
        input_text = _load_extract_input_text(args, allow_empty=bool(args.images))
        images = _load_cli_images(args)
        stage_models, stage_efforts = stage_routing(args)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2

    extractor = build_claim_extractor(
        ClaimExtractionRuntimeConfig(
            model=args.model,
            base_url=args.base_url,
            stage_models=stage_models,
            stage_efforts=stage_efforts,
            model_retry_attempts=args.model_retries,
            service_tier=getattr(args, "service_tier", None),
            max_claims=args.max_claims,
        )
    )
    try:
        result = await extractor.extract_claims(input_text, images=images)
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
    if result.input_text:
        print(f"  {result.input_text}")
    for image in images:
        print(f"  [{image.image_id}] {image.source}")
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


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "check":
        sys.exit(asyncio.run(run_check_command(args)))
    if args.command == "extract-claims":
        sys.exit(asyncio.run(run_extract_claims_command(args)))
    if args.command == "skills":
        sys.exit(run_skills_command(show=getattr(args, "show", None)))

    parser.print_help()
    sys.exit(1)
