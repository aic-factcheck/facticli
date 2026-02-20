# facticli

`facticli` is a pip-installable Python CLI for agentic claim verification built on `openai-agents`.

It restructures key ideas from `~/PhD/aic_averitec` (claim decomposition, evidence gathering, verdict synthesis) into a modular command-line multi-agent workflow with:
- open web search,
- orchestrated parallel subroutines,
- final veracity verdict + justification,
- explicit source output.

The architecture is intentionally inspired by Codex-style modular prompting: local skill prompts (`plan`, `research`, `judge`) with a thin orchestrator layer.

## Install

From this repository:

```bash
pip install -e .
```

## Configure

Set your API key:

```bash
export OPENAI_API_KEY=...
```

Optional defaults:

```bash
export FACTICLI_MODEL=gpt-4.1-mini
export FACTICLI_SEARCH_PROVIDER=openai
# only needed when FACTICLI_SEARCH_PROVIDER=brave
export BRAVE_SEARCH_API_KEY=...
```

## Usage

Run a claim check:

```bash
facticli check "The Eiffel Tower was built in 1889 for the World's Fair."
```

Run with Brave Search API retrieval:

```bash
facticli check --search-provider brave "The Eiffel Tower was built in 1889 for the World's Fair."
```

Show the generated plan:

```bash
facticli check --show-plan "The Eiffel Tower was built in 1889 for the World's Fair."
```

Machine-readable output:

```bash
facticli check --json --include-artifacts "The Eiffel Tower was built in 1889 for the World's Fair."
```

List built-in agent skills:

```bash
facticli skills
```

## CLI options

```text
facticli check [--model MODEL] [--max-checks N] [--parallel N]
               [--search-provider {openai,brave}]
               [--search-context-size {low,medium,high}]
               [--show-plan] [--json] [--include-artifacts]
               "<claim>"
```

## Current architecture (bootstrap)

- `plan` skill: decomposes claim into independent checks and search queries.
- `research` skill: runs one check using selectable web retrieval:
  - OpenAI hosted web search tool (`--search-provider openai`)
  - Brave Search API custom tool (`--search-provider brave`)
- `judge` skill: merges findings into one verdict:
  - `Supported`
  - `Refuted`
  - `Not Enough Evidence`
  - `Conflicting Evidence/Cherrypicking`

The orchestrator runs all check-research jobs concurrently (bounded by `--parallel`) and then performs final judgment.

## Repository layout

```text
src/facticli/
  cli.py             # command-line interface
  orchestrator.py    # planner -> parallel research -> judge pipeline
  agents.py          # openai-agents definitions
  skills.py          # skill registry and prompt loading
  types.py           # typed plan/finding/report contracts
  prompts/
    plan.md
    research.md
    judge.md
```

## Contributor guide

- Project contributor/agent guidance lives in `/Users/bertik/PhD/facticli/AGENTS.md`.
- `/Users/bertik/PhD/facticli/CLAUDE.md` is a symlink to the same file.

## Notes

- This is an initial bootstrap and intentionally leaves room for deeper evaluator tooling, benchmark harnesses, and richer source quality scoring.
- If you installed in editable mode, updates in `src/` are reflected immediately.

## License

CC-BY-SA-4.0
