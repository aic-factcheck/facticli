# facticli

`facticli` is a pip-installable Python CLI for agentic claim verification with pluggable inference providers (`openai-agents` and `gemini`).

It restructures key ideas from `~/PhD/aic_averitec` (claim decomposition, evidence gathering, verdict synthesis) into a modular command-line multi-agent workflow with:
- open web search,
- orchestrated parallel subroutines,
- final veracity verdict + justification,
- explicit source output.

The architecture is intentionally inspired by Codex-style modular prompting: local skill prompts (`plan`, `research`, `judge`) with explicit pipeline stages and pluggable provider adapters.

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
export GEMINI_API_KEY=...
export FACTICLI_GEMINI_MODEL=gemini-3-pro
export FACTICLI_INFERENCE_PROVIDER=openai-agents
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

Run with Gemini inference provider:

```bash
facticli check \
  --inference-provider gemini \
  --gemini-model gemini-3-pro \
  --search-provider brave \
  "The Eiffel Tower was built in 1889 for the World's Fair."
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

Extract decontextualized atomic check-worthy claims from arbitrary text:

```bash
facticli extract-claims "In last year’s debate, the minister said inflation fell below 3% while wages rose 10%."
```

Extract claims from a transcript file:

```bash
facticli extract-claims --from-file ./data/debate_excerpt.txt --json
```

## CLI options

```text
facticli check [--model MODEL] [--max-checks N] [--parallel N]
               [--inference-provider {openai-agents,gemini}]
               [--gemini-model GEMINI_MODEL]
               [--search-provider {openai,brave}]
               [--search-results N]
               [--search-context-size {low,medium,high}]
               [--show-plan] [--json] [--include-artifacts]
               "<claim>"

facticli extract-claims [--from-file PATH]
                        [--inference-provider {openai-agents,gemini}]
                        [--model MODEL] [--gemini-model GEMINI_MODEL]
                        [--max-claims N] [--json]
                        [text]
```

Validation notes:
- `--max-checks`, `--parallel`, and `--max-claims` must be integers `>= 1`.
- `--search-results` must be an integer in `1..20`.
- For `extract-claims`, provide either positional `text` or `--from-file`, but not both.

## Current architecture

Layered runtime:
- `core`: typed contracts, normalization helpers, and run artifacts.
- `application`: provider-agnostic interfaces, explicit stages (`PlanStage`, `ResearchStage`, `JudgeStage`, `ClaimExtractionStage`), and services.
- `adapters`: concrete provider strategies (`openai-agents`, `gemini`) and retrievers (Brave Search).

Pipeline behavior:
- `plan` skill decomposes claims into independent checks.
- `research` runs per-check concurrently with bounded parallelism and retry.
- `judge` synthesizes findings into one verdict with merged deduplicated sources.
- claim extraction runs through a dedicated extraction stage/backend.

Inference backends:
- `openai-agents` (default): uses OpenAI Agents SDK (`Runner`, tools, structured output).
- `gemini`: uses `genai.Client` structured prompting and Brave retrieval payloads.

## Repository layout

```text
src/facticli/
  core/
    contracts.py     # typed plan/finding/report/extraction contracts
    normalize.py     # deterministic normalization helpers
    artifacts.py     # run artifact schemas
  application/
    interfaces.py    # planner/research/judge/retriever strategy contracts
    stages.py        # explicit pipeline stages
    services.py      # fact-check and extraction application services
    factory.py       # provider wiring composition root
  adapters/
    openai_provider.py
    gemini_provider.py
    retrievers.py
  cli.py             # command-line interface
  orchestrator.py    # compatibility facade over application service
  claim_extraction.py# compatibility facade over extraction service
  skills.py          # skill registry + prompt loading
  types.py           # compatibility re-export for contracts
  prompts/
    extract_claims.md
    plan.md
    research.md
    judge.md
```

## Demo notebooks

Interactive demos live in `/Users/bertik/PhD/facticli/notebooks`:

- `01_planner_subroutine_demo.ipynb`
- `02_research_subroutine_demo.ipynb`
- `03_judge_subroutine_demo.ipynb`
- `04_full_checker_demo.ipynb`
- `05_claim_extraction_demo.ipynb`

Each notebook includes:
- auto-reload setup (`%load_ext autoreload`, `%autoreload 2`),
- emoji-based headings for quick navigation,
- multiple example claims as commented-out variable redefinitions.

## Testing

Run the integrated unit tests:

```bash
python3 -m unittest discover -s tests -p "test_*.py" -v
```

## Contributor guide

- Project contributor/agent guidance lives in `/Users/bertik/PhD/facticli/AGENTS.md`.
- `/Users/bertik/PhD/facticli/CLAUDE.md` is a symlink to the same file.

## Notes

- This is an initial bootstrap and intentionally leaves room for deeper evaluator tooling, benchmark harnesses, and richer source quality scoring.
- If you installed in editable mode, updates in `src/` are reflected immediately.

## License

CC-BY-SA-4.0
