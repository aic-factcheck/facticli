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

- [x] Add provider-agnostic retry/backoff taxonomy (timeouts, rate limits, transient network, schema mismatch) — `core/errors.py` + SDK `ModelRetrySettings` policy (2026-09).
- [ ] Add heuristic source-quality triggers to seed review decisions before model review.
- [x] Persist per-round metrics/artifacts for evaluation of follow-up loop effectiveness (FileRunArtifactRepository + per-stage usage/latency events).
- [ ] Add contradiction-specific follow-up planning so mixed findings trigger resolution checks automatically.
- [x] Add source quality scoring and ranking (authority, recency, primary-source preference, duplication) — deterministic tiers in `core/source_quality.py`; judge weighs tiers (2026-09). Recency/duplication ranking beyond tiers still open.
- [ ] Add contradiction-focused synthesis checks for `Conflicting Evidence/Cherrypicking`.
- [x] Expand deterministic tests for prompt/schema drift and renderer behavior (schema exclusion, lane-contract sections, renderer, payload builders; 2026-09).
- [x] Add dataset-driven regression/evaluation CLI with artifact logging (`facticli.averitec_eval`: label metrics + vendored Ev2R judge scorer; submission runner: `--artifacts-dir`, `--run-info`, `--resume`).

## Benchmark instrumentation (2026-07, for the EACL paper experiments)

- [x] Per-stage token usage and latency tracking wired into run artifacts.
- [x] Leakage controls: fact-check-domain blocklist (`--block-fact-checkers`, `--blocked-domain`) and per-claim evidence date cutoff (`--claim-date-field`; Brave freshness cap + post-hoc source filter; raw findings kept in artifacts for the leakage audit).
- [x] Closed-world mode: `knowledge_store` search provider (BM25-lite over per-claim AVeriTeC stores, `--knowledge-store-dir`).
- [x] AVeriTeC eval CLI: accuracy/macro-F1/per-label/confusion/FP-rates + Ev2R QA-recall with configurable judge (`--ev2r`, threshold note: 0.5 local FEVER-8 parity vs 0.44 on the permanent HF leaderboard).
- [ ] Download AVeriTeC dev knowledge stores and validate the knowledge_store provider end-to-end.
- [x] Leakage audit script over persisted artifacts (fact-check-domain hit rates, post-claim-date evidence rates, leaked-vs-clean accuracy split) — `python -m facticli.artifacts_audit` (2026-09).
- [x] No-harness single-agent baseline mode (same model + web tool + token budget, single prompt) — `--strategy single_agent` (2026-09).
- [ ] Submit final config to the permanent HF leaderboard (https://huggingface.co/spaces/fever/AVeriTeC).

## Harness modernization (2026-09, from Codex / pi / OpenClaw / 2026 harness + fact-checking survey)

Source survey: Codex CLI (typed context fragments, rollout token budgets, reviewer rubric + schema,
Guardian trust taxonomy, middle-elision truncation, per-role sub-agent TOML, web-search trust modes),
pi (~1k-token prompts, tools declare their own prompt snippets, typed terminal tool, tree sessions,
fixed compaction template), OpenClaw (tool policy filters schemas before the call, untrusted-content
wrapping, failover taxonomy, role-specific models, lane contracts with non-goals), and 2026 papers
(unverified-vs-insufficient findings, evidence ablation, cherry-pick override, citation faithfulness,
authority tiers, temporal leakage, alignment bottleneck, per-step effort routing, pass^k evals).

### A. Contracts and evidence quality
- [x] `AspectFinding.status` (`completed` | `failed` | `budget_exhausted`) + `failure_reason`: a check
      that crashed/timed out is a missing observation, not evidence of absence; judge prompt and renderer
      treat the two differently (Claude Code `/deep-research`, silent-failure taxonomy, Codex
      `guardian_context_omission`).
- [x] `SourceEvidence.tier` deterministic authority tiering (primary/official, reference, major outlet,
      fact-checker, other, user-generated) computed in the research stage from a domain table and
      surfaced to judge + renderer (AuthorityBench: explicit authority metadata in the prompt helps).
- [x] `VerificationCheck.acceptance_criteria`: planner emits "done when ..." rubric per check; reviewer
      grades checks against the rubric instead of vibes (Codex review rubric, rubric-guided verification).
- [x] Judge input restructuring: findings grouped by signal with counts, stable source ids (`S1..Sn`),
      verbatim snippets next to summaries; justification cites source ids; mandatory
      `counter_argument` (strongest case for a different verdict) and `evidence_gaps`
      (cherry-pick override; "Who is the Agent to Blame"; Anthropic generator/evaluator split).
- [x] `ReviewDecision.gaps` + per-follow-up `priority`: reviewer emits gap descriptions that seed
      targeted follow-up queries (PROClaim progressive RAG, WKGFC adaptive expansion).

### B. Harness robustness
- [x] Retry/backoff taxonomy: SDK `ModelRetrySettings` with a facticli policy (retry 408/409/429/5xx,
      network, timeout; never 400/401/403/404/context overflow), exponential backoff with jitter,
      per-run retry cap, and `error_kind` classification recorded in artifacts
      (OpenClaw failover semantics, Codex `request_max_retries`).
- [x] Run-level token budget ledger (`--token-budget`): planner + N researchers + review rounds + judge
      share one ledger; when exhausted, remaining checks become `budget_exhausted`, follow-up rounds are
      skipped, and the report says so (Codex rollout budget / `<rollout_budget>` reminders).
- [x] Tool-output truncation with header + middle elision for Brave / knowledge-store results, and
      explicit untrusted-content markers on every search payload (Codex output-truncation, OpenClaw
      `EXTERNAL_UNTRUSTED_CONTENT`, Guardian "tool outputs are untrusted evidence").
- [x] Per-stage model and reasoning-effort routing: `--stage-model STAGE=MODEL`,
      `--stage-effort STAGE=low|medium|high` (OpenClaw `utilityModel`/`subagents.model`, ARES per-step
      effort routing, ClaimCheck small research models + frontier judge).
- [x] Cached/reasoning token accounting in usage events (prompt-cache hit rate is the cost metric;
      static instructions first, volatile payload last).

### C. Prompts as lane contracts
- [x] Rewrite all skill prompts as contracts: Purpose / Non-goals / Inputs / Procedure / Failure handling
      / Output contract (OpenClaw lane contracts, NLAH failure taxonomy, Codex terse formatting rules).
- [x] Researcher: broad-then-narrow search, tool-call budget, evolving-note compression before answering,
      evidence-not-instructions rule, verbatim snippets, "do not adjudicate the claim"
      (Anthropic multi-agent research, Open Deep Research compress-then-return).
- [x] Planner: avoid over-decomposition when evidence cannot be aligned to sub-checks
      (alignment bottleneck), explicit temporal check policy, acceptance criteria.
- [x] SKILL.md-style frontmatter (`name`, `description`) on prompt files; `facticli skills` reads
      descriptions from the files; `facticli skills --show NAME` prints a prompt (pi / Codex skills).

### D. Observability, evaluation, baselines
- [x] Citation URL health pass (`--verify-sources`): HEAD/GET each cited URL, Wayback fallback,
      `url_status` live/archived/broken recorded per source ("Detecting and Correcting Reference
      Hallucinations", "Cited but Not Verified").
- [x] Artifact audit CLI (`python -m facticli.artifacts_audit`): leakage stats over persisted runs
      (fact-check-domain hits, post-claim-date sources, removed-source counts) and evidence-dependence
      ablation (re-judge with findings removed; flag verdicts that survive) (FAE/REAL, BrowseComp eval
      awareness).
- [x] Eval CLI: pass^k / per-claim verdict consistency across repeated submissions
      (Anthropic "Demystifying evals").
- [x] No-harness single-agent baseline (`--strategy single_agent`): one agent, same model + search tool,
      one prompt, same output contract; for the harness-vs-no-harness experiment.
- [x] Text renderer: source tiers, failed-check markers, counter-argument, usage/budget footer
      (`--show-usage`).

### E. Configuration
- [x] `facticli.toml` config file with `[defaults]` and `[profiles.<name>]` (`--config`, `--profile`,
      `FACTICLI_CONFIG`); CLI flags override profile values (Codex `config.toml` profiles, pi settings
      layering).

### Considered and deliberately not adopted (for now)
- Persistent memory tiers / dreaming / standing intents (OpenClaw): fact-checks are stateless per claim;
  revisit if a monitoring service is built.
- Heartbeat / cron scheduling: outside the CLI's scope; GitHub Actions covers periodic smoke runs.
- Free-form sub-agent spawning tools (Codex multi-agent v2): the fixed plan -> research -> review ->
  judge graph is the point of the research; the bounded review loop already covers adaptive depth.
- Context compaction: per-check research runs are short (<= max_turns); compaction would add cost
  without benefit. Tool-output truncation covers the actual bloat source.
- Debate-style judging (DebateCV / PROClaim): survey shows a single strong judge call often matches or
  beats debate; counter-argument field captures the cheap half. Left as a future `--judge-mode` flag.
- Page fetching + Wayback-first snapshots: retrieval is snippet-based today; a fetch tool would need a
  scraper dependency and its own leakage controls. Tracked under benchmark instrumentation.

## Testing operations

- [x] Add opt-in live smoke test guarded by environment flags.
- [x] Add a routine runner script for compile + test flow with optional live smoke execution.

## Architecture refactor (completed)

- [x] Introduce provider strategy interfaces (`Planner`, `Researcher`, `Judge`) and concrete adapters.
- [x] Introduce explicit stage objects (`PlanStage`, `ResearchStage`, `ReviewStage`, `JudgeStage`, `ClaimExtractionStage`).
- [x] Split code into layered runtime modules (`core`, `application`, `adapters`) while keeping CLI-compatible facades.
- [x] Add first-class run artifacts model and repository wired through the fact-check service and JSON output.
- [x] Consolidate OpenAI/Gemini execution into one OpenAI-compatible adapter path with profile-based key/base-url switching.
