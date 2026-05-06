# AGENTS.md

This file is the project operating manual for AI coding agents working in `facticli`.

## 1) Project Purpose

`facticli` is an agentic fact-checking Python CLI. It verifies a natural-language claim by:

1. planning verification checks,
2. running web-grounded research checks in parallel,
3. judging a final veracity verdict with explicit sources and justification.

The project is inspired by:
- Codex-style modular prompting and workflow segmentation,
- AVeriTeC-style claim decomposition -> evidence gathering -> verdict synthesis.

## 2) Current Scope

This repository is currently a bootstrap implementation focused on architecture and usable CLI behavior, not benchmark-level quality yet.

Implemented:
- pip-installable package (`pyproject.toml`),
- CLI with `facticli check`, `facticli extract-claims`, and `facticli skills`,
- modular prompt “skills” (`plan`, `research`, `judge`),
- claim extraction skill (`extract_claims`) for arbitrary text,
- orchestrator with bounded parallelism for sub-checks,
- opt-in bounded review loop for targeted follow-up research rounds,
- typed output contracts for plans, findings, verdicts, and sources.

Not yet implemented (expected future work):
- robust retry/backoff and richer failure policy,
- source quality scoring/ranking,
- deterministic tests and benchmark harnesses,
- advanced citation controls and duplicate clustering.

## 3) Tech Stack

- Language: Python 3.11+
- Packaging: `pyproject.toml` (Hatchling backend)
- Runtime dependencies:
  - `openai-agents` (Agents SDK)
  - `pydantic` (typed schemas/contracts)
  - `httpx` (Brave Search HTTP client)
- Model/tool runtime:
  - OpenAI-compatible models via Agents SDK
  - hosted `WebSearchTool` for open web retrieval
  - Brave Search API for custom retrieval

## 4) Repository Layout

- `pyproject.toml`: packaging metadata and console entrypoint
- `README.md`: user-facing setup and usage docs
- `TODO.md`: project-level TODO checklist
- `src/facticli/__init__.py`: package metadata
- `src/facticli/__main__.py`: `python -m facticli` runner
- `src/facticli/core/*`: domain contracts, normalization, run artifacts
- `src/facticli/application/*`: strategy interfaces, explicit stages, services, inference wiring
- `src/facticli/adapters/openai_provider.py`: shared Agents SDK stage adapters
- `src/facticli/adapters/provider_profile.py`: OpenAI-compatible env resolution + client bootstrap
- `src/facticli/cli.py`: CLI parser and command handlers
- `src/facticli/averitec_submission.py`: AVeriTeC submission generation entrypoint
- `src/facticli/brave_search.py`: Brave Search tool implementation
- `src/facticli/cli_validators.py`: CLI argument validators
- `src/facticli/skills.py`: skill registry + prompt loading
- `src/facticli/render.py`: human-readable output formatter
- `src/facticli/prompts/*.md`: reusable prompt instructions per skill

## 5) Core Architecture

### 5.1 Pipeline

The runtime uses layered architecture:

1. `core` layer
   - Typed contracts and normalization logic
   - Run artifact models for debugging/evaluation
2. `application` layer
   - Provider-agnostic strategy interfaces (`Planner`, `Researcher`, `Reviewer`, `Judge`)
   - Explicit stage objects (`PlanStage`, `ResearchStage`, `ReviewStage`, `JudgeStage`, extraction stage)
   - Service orchestration and artifact repository integration
3. `adapters` layer
   - Shared OpenAI-compatible strategy implementations
   - OpenAI-compatible client configuration (key/base URL/model/API mode)

Fact-check pipeline stages:

1. Planner stage (`plan` skill)
   - Input: claim text
   - Output: `InvestigationPlan` with independent `VerificationCheck` entries
2. Research stage (`research` skill), one run per check
   - Input: claim + one check payload
   - Tooling: web search
   - Output: `AspectFinding` with signal, summary, confidence, and sources
3. Review stage (`review` skill), optional bounded follow-up controller
   - Input: claim + current plan + current findings
   - Output: `ReviewDecision` with either finalize or targeted follow-up requests
4. Judge stage (`judge` skill)
   - Input: claim + plan + all findings
   - Output: `FactCheckReport` with final verdict, justification, findings, sources

Inference configuration:
- All inference backends use the same OpenAI-compatible codepath.
- Configure endpoint, key, and model with `OPENAI_API_BASE_URL`, `OPENAI_API_KEY`, and `OPENAI_API_MODEL`.
- CLI `--model` overrides `OPENAI_API_MODEL`; CLI `--base-url` overrides `OPENAI_API_BASE_URL`.
- OpenAI-hosted endpoints use the Responses API internally; other base URLs use Chat Completions internally.
- Model settings avoid optional parameters such as explicit `temperature` when they are not accepted uniformly across OpenAI-compatible providers.

### 5.2 Parallelism Model

- Check-level runs execute concurrently via `asyncio` tasks.
- Concurrency is bounded by a semaphore (`max_parallel_research`) to avoid overload.
- Failure of one check should not fail the whole run; failed checks are retried and then converted into `insufficient` findings.

### 5.3 Verdict Contract

Final verdict must be one of:
- `Supported`
- `Refuted`
- `Not Enough Evidence`
- `Conflicting Evidence/Cherrypicking`

Every final report should include:
- concise evidence-grounded justification,
- per-aspect findings,
- deduplicated source list with URLs and snippets.

## 6) Data Contracts

Primary schemas live in `src/facticli/core/contracts.py`:
- `InvestigationPlan`
- `VerificationCheck`
- `AspectFinding`
- `SourceEvidence`
- `ReviewDecision`
- `FactCheckReport`

Design intent:
- Keep outputs structured and machine-serializable.
- Minimize free-form text ambiguity at stage boundaries.
- Preserve enough per-stage artifacts for debugging and later evaluation.

## 7) Prompting Strategy

Prompt files are local and modular:
- `plan.md`: decomposition/planning behavior
- `research.md`: web-grounded evidence collection behavior
- `review.md`: bounded follow-up decision behavior
- `judge.md`: synthesis and verdict policy
- `extract_claims.md`: check-worthy claim extraction behavior

Prompt design principles:
- stage-specific responsibilities,
- strict schema compliance,
- explicit source-grounding requirements,
- deterministic tone through strict instructions and typed output contracts,
- minimal overlap between stage instructions.

## 8) CLI Behavior and UX

### 8.1 Commands

- `facticli check "<claim>"`
- `facticli extract-claims "<text>"`
- `facticli skills`

### 8.2 Key Flags

- `--model`
- `--max-checks`
- `--parallel`
- `--feedback-rounds`
- `--follow-up-checks`
- `--search-provider`
- `--search-results`
- `--search-context-size`
- `--base-url`
- `--max-claims` (extract-claims command)
- `--show-plan`
- `--stream-progress`
- `--json`
- `--include-artifacts`

Input/validation rules:
- `extract-claims` accepts either positional text or `--from-file` (mutually exclusive).
- `--max-checks`, `--parallel`, and `--max-claims` are positive integers.
- `--feedback-rounds` is a non-negative integer.
- `--follow-up-checks` is a positive integer.
- `--search-results` accepts integers in the range `1..20`.

### 8.3 Environment Variables

- `OPENAI_API_BASE_URL` (OpenAI-compatible base URL; optional for OpenAI default)
- `OPENAI_API_KEY` (required for live checks)
- `OPENAI_API_MODEL` (required unless passed with `--model`)

## 9) Important Design Constraints

- Always keep source attribution visible in final output.
- Never treat model output as evidence without external source URLs.
- Preserve stage boundaries; avoid collapsing all logic into one prompt.
- Keep review/follow-up loops bounded and explicit.
- Keep orchestrator resilient to partial failures.
- Prefer explicit typed schemas over ad-hoc dict contracts.
- Keep this project CLI-first and automation-friendly.

## 10) Engineering Conventions

- Use ASCII unless file already requires Unicode.
- Keep modules small and responsibility-focused.
- Keep public CLI output stable where possible.
- Add/adjust docs when commands/flags/schema change.
- Avoid hidden behavior; config should be discoverable via CLI/help/docs.

## 11) Validation Checklist (for agents)

After significant changes, run:

1. `python3 -m compileall src`
2. `python3 -m facticli --help`
3. `python3 -m facticli skills`
4. `facticli --help` (if package installed in environment)
5. `python3 -m unittest discover -s tests -p "test_*.py" -v`
6. `./scripts/test_routine.sh` (loads `.env`, same checks as above)
7. CI mirrors these checks via `.github/workflows/ci.yml`

If API key is available, also run at least one live smoke test:

`FACTICLI_RUN_LIVE_SMOKE=1 python3 -m unittest tests.test_live_smoke -v`

GitHub live smoke automation:
- `.github/workflows/live-smoke.yml` runs smoke tests on manual dispatch and schedule when `OPENAI_API_KEY` secret is configured.

## 12) Extension Roadmap

Recommended next increments:

1. retry/backoff + error taxonomy in orchestrator,
2. source quality scoring (primary source preference, freshness, authority),
3. cross-source contradiction analysis for `Conflicting` verdicts,
4. deterministic tests with mocked Agents SDK runs,
5. evaluation CLI for datasets and regression tracking.

## 13) How to Work in This Repo

- Start from the smallest change that keeps architecture coherent.
- Keep stage contracts backwards compatible when possible.
- If you change data contracts, update prompt requirements and renderers together.
- If you change CLI semantics, update `README.md` and this file in the same change.
