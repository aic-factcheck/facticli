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

- [ ] Add provider-agnostic retry/backoff taxonomy (timeouts, rate limits, transient network, schema mismatch).
- [ ] Add source quality scoring and ranking (authority, recency, primary-source preference, duplication).
- [ ] Add contradiction-focused synthesis checks for `Conflicting Evidence/Cherrypicking`.
- [ ] Expand deterministic tests for prompt/schema drift and renderer behavior.
- [ ] Add dataset-driven regression/evaluation CLI with artifact logging.
