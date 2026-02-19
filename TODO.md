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
