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
- [x] Add an opt-in bounded review loop that can request targeted follow-up research rounds.

- [ ] Add provider-agnostic retry/backoff taxonomy (timeouts, rate limits, transient network, schema mismatch).
- [ ] Add heuristic source-quality triggers to seed review decisions before model review.
- [x] Persist per-round metrics/artifacts for evaluation of follow-up loop effectiveness (FileRunArtifactRepository + per-stage usage/latency events).
- [ ] Add contradiction-specific follow-up planning so mixed findings trigger resolution checks automatically.
- [ ] Add source quality scoring and ranking (authority, recency, primary-source preference, duplication).
- [ ] Add contradiction-focused synthesis checks for `Conflicting Evidence/Cherrypicking`.
- [ ] Expand deterministic tests for prompt/schema drift and renderer behavior.
- [x] Add dataset-driven regression/evaluation CLI with artifact logging (`facticli.averitec_eval`: label metrics + vendored Ev2R judge scorer; submission runner: `--artifacts-dir`, `--run-info`, `--resume`).

## Benchmark instrumentation (2026-07, for the EACL paper experiments)

- [x] Per-stage token usage and latency tracking wired into run artifacts.
- [x] Leakage controls: fact-check-domain blocklist (`--block-fact-checkers`, `--blocked-domain`) and per-claim evidence date cutoff (`--claim-date-field`; Brave freshness cap + post-hoc source filter; raw findings kept in artifacts for the leakage audit).
- [x] Closed-world mode: `knowledge_store` search provider (BM25-lite over per-claim AVeriTeC stores, `--knowledge-store-dir`).
- [x] AVeriTeC eval CLI: accuracy/macro-F1/per-label/confusion/FP-rates + Ev2R QA-recall with configurable judge (`--ev2r`, threshold note: 0.5 local FEVER-8 parity vs 0.44 on the permanent HF leaderboard).
- [ ] Download AVeriTeC dev knowledge stores and validate the knowledge_store provider end-to-end.
- [ ] Leakage audit script over persisted artifacts (fact-check-domain hit rates, post-claim-date evidence rates, leaked-vs-clean accuracy split).
- [ ] No-harness single-agent baseline mode (same model + web tool + token budget, single prompt).
- [ ] Submit final config to the permanent HF leaderboard (https://huggingface.co/spaces/fever/AVeriTeC).

## Testing operations

- [x] Add opt-in live smoke test guarded by environment flags.
- [x] Add a routine runner script for compile + test flow with optional live smoke execution.

## Architecture refactor (completed)

- [x] Introduce provider strategy interfaces (`Planner`, `Researcher`, `Judge`) and concrete adapters.
- [x] Introduce explicit stage objects (`PlanStage`, `ResearchStage`, `ReviewStage`, `JudgeStage`, `ClaimExtractionStage`).
- [x] Split code into layered runtime modules (`core`, `application`, `adapters`) while keeping CLI-compatible facades.
- [x] Add first-class run artifacts model and repository wired through the fact-check service and JSON output.
- [x] Consolidate OpenAI/Gemini execution into one OpenAI-compatible adapter path with profile-based key/base-url switching.
