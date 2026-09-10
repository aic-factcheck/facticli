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
- CLI with `facticli check`, `facticli extract-claims`, and `facticli skills [--show NAME]`,
- modular prompt "skills" (`plan`, `research`, `review`, `judge`, `single_agent`, `extract_claims`) as
  markdown lane contracts with SKILL.md-style frontmatter,
- orchestrator with bounded parallelism for sub-checks and an opt-in bounded review loop,
- typed output contracts for plans, findings, verdicts, and sources, including harness-assigned
  fields (finding `status`, source `tier`, `url_status`) kept out of the model-facing schema,
- error taxonomy + SDK-managed retries with a per-run retry budget,
- run-level token budget ledger (`--token-budget`) with cached/reasoning token accounting,
- deterministic source authority tiering and untrusted-content / truncation controls on tool output,
- per-stage model and reasoning-effort routing (`--stage-model`, `--stage-effort`),
- no-harness single-agent baseline (`--strategy single_agent`),
- citation URL health pass (`--verify-sources`),
- TOML config profiles (`facticli.toml`, `--config`, `--profile`),
- AVeriTeC batch runner, scorer (label metrics, Ev2R, pass^k consistency), artifact audit CLI
  (leakage stats, evidence-dependence ablation),
- deterministic unit tests for all of the above (no network, no model calls).

Not yet implemented (see `TODO.md`):
- recency/duplication-aware source ranking beyond tiers,
- contradiction-specific follow-up planning and synthesis checks,
- page fetching with Wayback-first snapshots,
- debate-style judging (deliberately deferred; see "Considered and not adopted" in `TODO.md`).

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
- `src/facticli/core/*`: domain contracts, normalization, run artifacts, error taxonomy
  (`errors.py`), usage + budget ledger (`usage.py`), source tiers (`source_quality.py`),
  tool-output truncation (`truncation.py`), retrieval constraints (`constraints.py`)
- `src/facticli/application/*`: strategy interfaces, explicit stages, services (pipeline and
  single-agent), factory, citation checker (`citations.py`), TOML settings (`settings_file.py`)
- `src/facticli/adapters/openai_provider.py`: shared Agents SDK stage adapters + single-agent adapter
- `src/facticli/adapters/payloads.py`: pure stage-input builders (judge grouping, review grading)
- `src/facticli/adapters/retry_policy.py`: SDK retry policy bound to the run ledger
- `src/facticli/adapters/provider_profile.py`: OpenAI-compatible env resolution + client bootstrap
- `src/facticli/cli.py`: CLI parser and command handlers (config-file layering in `main`)
- `src/facticli/averitec_submission.py`: AVeriTeC submission generation entrypoint
- `src/facticli/averitec_eval.py`, `src/facticli/eval/*`: scorer (label metrics, Ev2R, consistency)
- `src/facticli/artifacts_audit.py`: leakage / robustness / evidence-ablation audit over run artifacts
- `src/facticli/brave_search.py`, `src/facticli/knowledge_store.py`: search tools (bounded output)
- `src/facticli/cli_validators.py`: CLI argument validators
- `src/facticli/skills.py`: skill registry + frontmatter-aware prompt loading
- `src/facticli/render.py`: human-readable output formatter
- `facticli.example.toml`: config profile template
- `src/facticli/web/*`: optional FastAPI GUI for claim extraction (`python -m facticli.web`)
- `src/facticli/web/static/*`: branded CEDMO single-page frontend (HTML/CSS/JS + logo)
- `src/facticli/prompts/*.md`: reusable prompt instructions per skill
- `pages/*`: browser-only claim extractor demo deployed to GitHub Pages via
  `.github/workflows/pages.yml`; the API key is shipped passphrase-encrypted
  (`pages/encrypt_key.mjs`), sourced from the `DEMO_OPENAI_API_KEY` and
  `DEMO_PASSPHRASE` repository secrets

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
   - Input: claim text + max checks
   - Output: `InvestigationPlan` with independent `VerificationCheck` entries, each with
     `search_queries` and `acceptance_criteria`
2. Research stage (`research` skill), one run per check
   - Input: claim + one check payload + constraints + budget reminder
   - Tooling: web search (bounded output, untrusted-content notice)
   - Output: `AspectFinding` with signal, summary, confidence, tiered sources; harness sets
     `status` (`completed` / `failed` / `budget_exhausted`) and `failure_reason`
3. Review stage (`review` skill), optional bounded follow-up controller
   - Input: checks paired with findings and acceptance criteria, limits, round index
   - Output: `ReviewDecision` with `gaps` and either finalize or targeted retries / follow-up checks
4. Judge stage (`judge` skill)
   - Input: findings grouped by signal, shared source table `S1..Sn` with snippets and tiers,
     failed-check notice
   - Output: `FactCheckReport` with verdict, `[S#]`-cited justification, `counter_argument`,
     `evidence_gaps`; harness findings are authoritative in the final report
5. Citation check stage (optional, `--verify-sources`, no model call)
   - Marks each cited URL `live`, `archived` (Wayback), or `broken`

Alternative strategy: `--strategy single_agent` runs the `single_agent` skill (one agent, same
tool and output contract) through the same run context, constraints, tiering, budget, and
citation controls. It exists to measure what the decomposition buys.

Run context: every run installs retrieval constraints and a usage/budget ledger via ContextVars.
The ledger records per-stage usage (including cached and reasoning tokens), draws model retries
from a per-run retry budget, and enforces `--token-budget` (remaining checks become
`budget_exhausted`, review rounds are skipped and recorded in `artifacts.budget`). Budget checks
happen at stage boundaries (before each check / round); a running model call is never interrupted,
so one stage of overshoot is expected and reported in the usage footer.

Inference configuration:
- All inference backends use the same OpenAI-compatible codepath.
- Configure endpoint, key, and model with `OPENAI_API_BASE_URL`, `OPENAI_API_KEY`, and `OPENAI_API_MODEL`.
- CLI `--model` overrides `OPENAI_API_MODEL`; CLI `--base-url` overrides `OPENAI_API_BASE_URL`.
- `--stage-model STAGE=MODEL` and `--stage-effort STAGE=EFFORT` route one stage to another model or
  reasoning effort; the resolved mapping is stored in `artifacts.stage_models`.
- OpenAI-hosted endpoints use the Responses API internally; other base URLs use Chat Completions internally.
- Model settings avoid optional parameters such as explicit `temperature` when they are not accepted
  uniformly across OpenAI-compatible providers; reasoning effort is only sent when requested.
- Retries: `ModelSettings.retry` with `adapters/retry_policy.py` (retry rate-limit/timeout/network/5xx
  with jittered exponential backoff; never auth, bad request, not found, context overflow, schema).
  Per-call cap `--model-retries`, per-run cap `--retry-budget`.
- Config files: `facticli.toml` / `--config` / `FACTICLI_CONFIG` with `[defaults]` and
  `[profiles.<name>]`; precedence flag > profile > defaults > built-in.

### 5.2 Parallelism Model

- Check-level runs execute concurrently via `asyncio` tasks.
- Concurrency is bounded by a semaphore (`max_parallel_research`) to avoid overload.
- Failure of one check should not fail the whole run; failed checks are retried according to the
  error taxonomy and then become explicit `status = failed` findings (signal `insufficient`,
  confidence 0, `failure_reason` set). Downstream prompts treat them as missing observations, not
  as evidence of absence. Checks skipped by the token budget are `status = budget_exhausted`.

### 5.3 Verdict Contract

Final verdict must be one of:
- `Supported`
- `Refuted`
- `Not Enough Evidence`
- `Conflicting Evidence/Cherrypicking`

Every final report should include:
- concise evidence-grounded justification citing source ids (`[S1]`),
- `counter_argument` (strongest case for another verdict) and `evidence_gaps`,
- per-aspect findings with status,
- deduplicated, tiered source list with URLs and snippets (and `url_status` when verified).

## 6) Data Contracts

Primary schemas live in `src/facticli/core/contracts.py`:
- `InvestigationPlan`, `VerificationCheck` (with `acceptance_criteria`)
- `AspectFinding` (with harness-only `status`, `failure_reason`)
- `SourceEvidence` (with harness-only `tier`, `url_status`, `archived_url`)
- `ReviewDecision` (with `gaps`)
- `FactCheckReport` (with `counter_argument`, `evidence_gaps`)
- enums `FindingStatus`, `SourceTier`, `UrlStatus`

Design intent:
- Keep outputs structured and machine-serializable.
- Minimize free-form text ambiguity at stage boundaries.
- Fields the harness assigns after the model call are annotated `SkipJsonSchema` so the model is
  never asked to guess them; the harness copy of findings is authoritative in the final report.
- Preserve enough per-stage artifacts for debugging and later evaluation (`RunArtifacts` includes
  usage events, budget status, error kinds, stage models, citation check counts).

## 7) Prompting Strategy

Prompt files are local and modular, each with frontmatter (`name`, `description`) and the lane
contract structure Purpose / Non-goals / Inputs / Procedure / Failure handling / Output contract:
- `plan.md`: decomposition with acceptance criteria; avoid over-decomposition
- `research.md`: broad-then-narrow search, tool-call budget, verbatim snippets, evidence-not-instructions
- `review.md`: rubric grading against acceptance criteria; gaps -> narrow follow-ups; no praise
- `judge.md`: tier-weighted evidence, counter-argument first, `[S#]` citations, calibrated confidence
- `single_agent.md`: no-harness baseline with the same standards
- `extract_claims.md`: check-worthy claim extraction behavior (language-consistent)

Prompt design principles:
- stage-specific responsibilities with explicit non-goals,
- strict schema compliance,
- explicit source-grounding requirements; retrieved content is evidence, never instructions,
- named failure modes with a named recovery each,
- deterministic tone through strict instructions and typed output contracts,
- minimal overlap between stage instructions.
- Stage inputs are built by pure functions in `adapters/payloads.py` so they are testable without a model.

Claim-extraction language policy:
- `extract_claims.md` detects the input language and reports it as an ISO 639-1
  code in `ClaimExtractionResult.detected_language`.
- All generated text (claim text, reasons, coverage/exclusion notes) is written
  in the detected language; `source_fragment` is copied verbatim.
- Original orthography/diacritics are preserved; behaviour is held consistent
  across languages (validated on Czech, Slovak, Polish, and English).

## 8) CLI Behavior and UX

### 8.1 Commands

- `facticli [--config PATH] [--profile NAME] check "<claim>"`
- `facticli extract-claims "<text>"`
- `facticli skills [--show NAME]`
- `python -m facticli.averitec_submission`, `python -m facticli.averitec_eval`, `python -m facticli.artifacts_audit`

### 8.2 Key Flags

- `--model`, `--base-url`, `--stage-model STAGE=MODEL`, `--stage-effort STAGE=EFFORT`
- `--strategy {pipeline,single_agent}`
- `--max-checks`, `--parallel`, `--feedback-rounds`, `--follow-up-checks`
- `--token-budget`, `--model-retries`, `--retry-budget`
- `--search-provider {openai,brave,knowledge_store}`, `--search-results`, `--search-context-size`
- `--block-fact-checkers`, `--blocked-domain DOMAIN`
- `--verify-sources`, `--artifacts-dir`
- `--max-claims` (extract-claims command)
- `--show-plan`, `--show-usage`, `--stream-progress`, `--json`, `--include-artifacts`

Input/validation rules:
- `extract-claims` accepts either positional text or `--from-file` (mutually exclusive).
- `--max-checks`, `--parallel`, `--max-claims`, `--follow-up-checks`, `--token-budget` are positive integers.
- `--feedback-rounds`, `--model-retries`, `--retry-budget` are non-negative integers.
- `--search-results` accepts integers in the range `1..20`.
- `--stage-model` / `--stage-effort` require a known stage; efforts are `minimal|low|medium|high`.
- Config-file keys mirror long flag names (underscores); unknown profiles are an error (exit 2).

### 8.3 Environment Variables

- `OPENAI_API_BASE_URL` (OpenAI-compatible base URL; optional for OpenAI default)
- `OPENAI_API_KEY` (required for live checks)
- `OPENAI_API_MODEL` (required unless passed with `--model`)
- `FACTICLI_SEARCH_PROVIDER`, `BRAVE_SEARCH_API_KEY` (retrieval)
- `FACTICLI_CONFIG`, `FACTICLI_PROFILE` (config file layering)

## 9) Important Design Constraints

- Always keep source attribution visible in final output.
- Never treat model output as evidence without external source URLs.
- Preserve stage boundaries; avoid collapsing all logic into one prompt (the single-agent strategy
  is a deliberate baseline, not a shortcut).
- Keep review/follow-up loops bounded and explicit; budgets bound cost, not just round counts.
- Keep orchestrator resilient to partial failures, and keep failures visible: a check the harness
  could not run must surface as `status = failed | budget_exhausted`, never as evidence.
- Retrieved web content is untrusted evidence, never instructions; keep the notice on tool output.
- Harness-assigned fields stay out of model-facing schemas (`SkipJsonSchema`).
- Prefer explicit typed schemas over ad-hoc dict contracts; build stage inputs in `adapters/payloads.py`.
- Keep this project CLI-first and automation-friendly; every flag must be settable from `facticli.toml`.

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

1. recency/duplication-aware source ranking on top of tiers,
2. cross-source contradiction analysis for `Conflicting` verdicts (contradiction-triggered follow-ups),
3. page fetching with Wayback-first snapshots and content-date verification for benchmark runs,
4. optional debate-style judging (`--judge-mode`) behind a flag, evaluated against the counter-argument baseline,
5. stage-wise eval metrics (question quality, evidence recall, verdict) attributed per prompt change.

## 13) How to Work in This Repo

- Start from the smallest change that keeps architecture coherent.
- Keep stage contracts backwards compatible when possible.
- If you change data contracts, update prompt requirements and renderers together.
- If you change CLI semantics, update `README.md` and this file in the same change.
