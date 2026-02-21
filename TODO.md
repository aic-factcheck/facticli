# facticli Bootstrap TODOs

- [x] Define package layout and Python packaging metadata (`pyproject.toml`, `src/`).
- [x] Implement typed fact-checking data contracts (plan, findings, verdict, sources).
- [x] Add Codex-inspired prompt modules as reusable local skills (`plan`, `research`, `judge`).
- [x] Build `openai-agents` pipeline:
  - [x] planner agent to decompose claim into parallelizable checks
  - [x] researcher agent (with open web search tool) for each check
  - [x] judge agent to synthesize final veracity verdict + justification + sources
- [x] Implement CLI entrypoint (`facticli check`) and output formatting (text + JSON).
- [x] Document architecture and usage in `README.md`.
- [x] Run lightweight validation (`python -m compileall`, CLI `--help`).

## Quality hardening backlog

- [x] Normalize planner output before research (strip/repair `aspect_id`, question text, and query lists).
- [x] Add bounded retries for per-check research and preserve partial results under flaky retrieval.
- [x] Make Brave query fan-out resilient: keep successful query payloads when others fail.
- [x] Replace silent CLI coercion with strict argument validation for positive/ranged integer flags.
- [x] Reject ambiguous claim-extraction input (`text` + `--from-file`) with explicit error.
- [x] Tighten prompt guidance around source quality, corroboration, and time-sensitive claims.
- [x] Add optional CLI progress streaming for plan + per-check research updates.

- [ ] Add provider-agnostic retry/backoff taxonomy (timeouts, rate limits, transient network, schema mismatch).
- [ ] Add source quality scoring and ranking (authority, recency, primary-source preference, duplication).
- [ ] Add contradiction-focused synthesis checks for `Conflicting Evidence/Cherrypicking`.
- [ ] Expand deterministic tests for prompt/schema drift and renderer behavior.
- [ ] Add dataset-driven regression/evaluation CLI with artifact logging.

## Testing operations

- [x] Add opt-in live smoke test guarded by environment flags.
- [x] Add a routine runner script for compile + test flow with optional live smoke execution.

## Architecture refactor (completed)

- [x] Introduce provider strategy interfaces (`Planner`, `Researcher`, `Judge`) and concrete adapters.
- [x] Introduce explicit stage objects (`PlanStage`, `ResearchStage`, `JudgeStage`, `ClaimExtractionStage`).
- [x] Split code into layered runtime modules (`core`, `application`, `adapters`) while keeping CLI-compatible facades.
- [x] Add first-class run artifacts model and repository wired through the fact-check service and JSON output.
- [x] Consolidate OpenAI/Gemini execution into one OpenAI-compatible adapter path with profile-based key/base-url switching.
