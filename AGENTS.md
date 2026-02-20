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
  - `google-genai` (Gemini client SDK)
  - `pydantic` (typed schemas/contracts)
- Model/tool runtime:
  - OpenAI Responses-compatible models via Agents SDK
  - Gemini models via `genai.Client`
  - hosted `WebSearchTool` for open web retrieval
  - Brave Search API for custom retrieval

## 4) Repository Layout

- `pyproject.toml`: packaging metadata and console entrypoint
- `README.md`: user-facing setup and usage docs
- `TODO.md`: project-level TODO checklist
- `src/facticli/__init__.py`: package metadata
- `src/facticli/__main__.py`: `python -m facticli` runner
- `src/facticli/core/*`: domain contracts, normalization, run artifacts
- `src/facticli/application/*`: strategy interfaces, explicit stages, services, provider wiring
- `src/facticli/adapters/*`: OpenAI/Gemini provider adapters and Brave retriever
- `src/facticli/cli.py`: CLI parser and command handlers
- `src/facticli/orchestrator.py`: compatibility facade for fact-check service
- `src/facticli/claim_extraction.py`: compatibility facade for extraction service
- `src/facticli/agents.py`: OpenAI Agents SDK builder functions
- `src/facticli/types.py`: compatibility re-export for domain contracts
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
   - Provider-agnostic strategy interfaces (`Planner`, `Researcher`, `Judge`, `Retriever`)
   - Explicit stage objects (`PlanStage`, `ResearchStage`, `JudgeStage`, extraction stage)
   - Service orchestration and artifact repository integration
3. `adapters` layer
   - OpenAI and Gemini concrete strategy implementations
   - Brave retrieval strategy

Fact-check pipeline stages:

1. Planner stage (`plan` skill)
   - Input: claim text
   - Output: `InvestigationPlan` with independent `VerificationCheck` entries
2. Research stage (`research` skill), one run per check
   - Input: claim + one check payload
   - Tooling: web search
   - Output: `AspectFinding` with signal, summary, confidence, and sources
3. Judge stage (`judge` skill)
   - Input: claim + plan + all findings
   - Output: `FactCheckReport` with final verdict, justification, findings, sources

Inference providers:
- `openai-agents` (default): planner/research/judge run through Agents SDK.
- `gemini`: planner/research/judge run through `genai.Client` structured prompting.
  - Current constraint: Gemini path uses `search_provider=brave`.

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

Primary schemas in `src/facticli/core/contracts.py` (re-exported via `src/facticli/types.py`):
- `InvestigationPlan`
- `VerificationCheck`
- `AspectFinding`
- `SourceEvidence`
- `FactCheckReport`

Design intent:
- Keep outputs structured and machine-serializable.
- Minimize free-form text ambiguity at stage boundaries.
- Preserve enough per-stage artifacts for debugging and later evaluation.

## 7) Prompting Strategy

Prompt files are local and modular:
- `plan.md`: decomposition/planning behavior
- `research.md`: web-grounded evidence collection behavior
- `judge.md`: synthesis and verdict policy

Prompt design principles:
- stage-specific responsibilities,
- strict schema compliance,
- explicit source-grounding requirements,
- deterministic tone (low temperature),
- minimal overlap between stage instructions.

## 8) CLI Behavior and UX

### 8.1 Commands

- `facticli check "<claim>"`
- `facticli skills`

### 8.2 Key Flags

- `--model`
- `--max-checks`
- `--parallel`
- `--search-provider`
- `--search-results`
- `--search-context-size`
- `--inference-provider`
- `--gemini-model`
- `--max-claims` (extract-claims command)
- `--show-plan`
- `--json`
- `--include-artifacts`

Input/validation rules:
- `extract-claims` accepts either positional text or `--from-file` (mutually exclusive).
- `--max-checks`, `--parallel`, and `--max-claims` are positive integers.
- `--search-results` accepts integers in the range `1..20`.

### 8.3 Environment Variables

- `OPENAI_API_KEY` (required for live checks)
- `GEMINI_API_KEY` (required for Gemini provider)
- `FACTICLI_MODEL` (optional default model override)
- `FACTICLI_GEMINI_MODEL` (optional Gemini model override)
- `FACTICLI_INFERENCE_PROVIDER`

## 9) Important Design Constraints

- Always keep source attribution visible in final output.
- Never treat model output as evidence without external source URLs.
- Preserve stage boundaries; avoid collapsing all logic into one prompt.
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

If API key is available, also run at least one live smoke test:

`facticli check "The Eiffel Tower was built in 1889."`

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
