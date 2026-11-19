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
- [ ] Add heuristic source-quality triggers to seed review decisions before model review (tried in 2026-11 as
      deterministic reviewer flags; not adopted without a measured review run, see below).
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

## Harness modernization II (2026-11, second survey)

Second pass over Codex, pi, OpenClaw, Hermes, Claude Code, DeerFlow, opencode, deepagents, and
GPT-Researcher. Since 2026-09 these harnesses added little agent structure; the new work is
deterministic bookkeeping around the model: checking what the model claims against an execution
record, naming every way a run can end, billing failed attempts, replaying retrieval. The survey
was read by research agents and the individual harness citations were not re-verified.

### Adopted
- [x] Retrieval journal + source grounding: each research run records what its search tools returned;
      every cited source is labelled `quote_matched` / `url_matched` / `unmatched` by string comparison
      (`core/retrieval.py`, `core/grounding.py`). The judge and reviewer are told about `unmatched` only.
- [x] Hard search cap enforced by the tools at twice the advertised `--tool-call-budget`.
- [x] Wrap-up call: a research run at its turn limit gets one tools-off call instead of failing.
- [x] Usage ledger counts every attempt (failed, timed out, wrap-up) and records cache-write tokens.
- [x] Record/replay cache for Brave responses (`--search-cache-dir`).
- [x] Error taxonomy: `quota` vs `rate_limit`, `forbidden` vs `auth`, `refusal`; fatal kinds (auth,
      quota, unknown model, rejected Brave key) abort the run and the AVeriTeC batch (exit code 3).
- [x] Reference date (claim date, else today) in the research, review, judge, and single-agent payloads,
      not the planner's; reasoning efforts up to `max`.
- [x] AVeriTeC evidence rows each state a different fact (one row per finding first, then snippets).
- [x] Fixed along the way: `--search-results` had no effect; `blocked_domains` and `search_results`
      in `facticli.toml` were ignored and file values skipped validation; Brave results reached the
      model with `<strong>` markup; an unparseable claim date silently disabled the evidence cutoff.

### Measured (gpt-5.6-terra, first 10 claims of `dev_20.json`, Brave, fact-checkers blocked, claim-date cutoff)
Single runs per arm unless stated, so a difference of one or two claims is noise.
- Old code 9/10 correct at 90.5k tokens per claim. This code 8/10 at 98.3k, agreeing with old on 9 of
  10 claims. The one difference (claim 18) is not a regression: rerun three times each, old and new
  both answer `Refuted` every time.
- Provenance: 57 of 57 cited sources `quote_matched` in the final run. An earlier run had 62 of 64, one
  `url_matched` (table text re-typed with separators) and one `unmatched` (the researcher wrote
  `..._JUDICIARY.pdf` where the search had returned `...-JUDICIARY.pdf`). With hosted OpenAI search,
  24 of 24 `url_matched`. In one review-enabled run the judge mis-copied a URL into its own source list.
- Reference date in the planner input: on claim 270 ("House Speaker Nancy Pelosi said ...") the planner
  then added a check on whether she was Speaker on that date; it came back `supports` and the verdict
  went from `Refuted` to `Not Enough Evidence` in 3 of 3 runs (old code: 1 of 3). Withholding the date
  from the planner only: 1 of 3, and no such check.
- Search cap enforced at the advertised budget (1x): calls refused in 6 of 20 checks, 7/10 correct. At
  2x: one check in about 20 reaches it (19 searches attempted against a budget of 8); it still returned
  a finding.
- Turn limit forced to 3 (4 claims): the old code lost a check and its ledger showed 4k tokens for a
  claim that spent about 120k; with the wrap-up call no check failed.
- 40 to 50% of input tokens were billed cache writes, which the old ledger did not record.
- Evidence packing, 40 saved reports scored twice with Ev2R (gpt-5.2): mean recall 0.72 with distinct
  rows vs 0.63 with one summary per source; better on 8 of 10 claims, worse on 1. The old packing put
  8.6 rows per claim into the submission but only 2.9 distinct answers.
- Harness audit (2026-11-19, same setup, 10 claims): 8/10 correct at 103k tokens per claim, research 92%
  of tokens (plan 1.8k, judge 6.9k). 20 checks, 3.8 model requests and 8.6 searches each; no failed
  attempt, timeout, wrap-up call, or cap refusal. The retry, timeout, wrap-up, and cap machinery costs
  nothing when it does not fire, so the spend is evidence tokens.

### Built, measured, and not adopted
- Lower search budget (`--tool-call-budget 5` instead of 8, 2026-11-19): 32% fewer tokens (69.7k per
  claim) but 6/10 correct instead of 8/10, both losses turning into `Not Enough Evidence`. Rerun twice
  at budget 8 on the same code: claim 307 `Refuted` both times, so that loss is the budget; claim 270
  `Not Enough Evidence` once (it flips at budget 8 too, see above). Fewer searches cost verdicts, as
  with the 1x cap.
- Page reading (`fetch_page` on search results, Wayback snapshot under a claim date, visible-text
  extraction, passage selection): 42% more tokens, no verdict change, and only 13 of 40 reads found a
  snapshot from before the claim date. It also adds a fetch surface (SSRF checks, a DNS-rebinding
  gap). Worth another look for undated claims, where pages are read live.
- Schema-repair call after invalid structured output: never fired in a real run, and on tool-using
  stages it turned "the model answered in prose" into a tools-off answer that looked like a normal
  finding. A failed check that is retried with tools is the better outcome.
- Deterministic reviewer flags (single source, low tier only, conflicting checks, ...): most restate
  fields already in the review payload, and no review-enabled run was measured.
- A separate `--tool-call-cap` flag: the only other setting tried (1x) cost a verdict.

### Open
- [ ] A run that raises (planner or judge failure, fatal abort) saves no artifact, so its spend is
      missing from the audit and from the batch manifest.
- [ ] Brave monthly-quota exhaustion is not told apart from throttling (response shape unverified);
      it shows up as failed searches in the receipts, not as a fatal error.
- [ ] Near-duplicate clustering of syndicated sources before the judge.
- [ ] Evaluate the review loop with and without `unmatched` / `stop_reason` in its payload.
- [ ] Hosted OpenAI search: blocklist is applied post hoc; the API's `filters` only allow-list.
- [x] `facticli check` offered `--search-provider knowledge_store` with no way to point at a store, so
      every check came back insufficient; it now refuses it (2026-11-19).

## Testing operations

- [x] Add opt-in live smoke test guarded by environment flags.
- [x] Add a routine runner script for compile + test flow with optional live smoke execution.

## Architecture refactor (completed)

- [x] Introduce provider strategy interfaces (`Planner`, `Researcher`, `Judge`) and concrete adapters.
- [x] Introduce explicit stage objects (`PlanStage`, `ResearchStage`, `ReviewStage`, `JudgeStage`, `ClaimExtractionStage`).
- [x] Split code into layered runtime modules (`core`, `application`, `adapters`) while keeping CLI-compatible facades.
- [x] Add first-class run artifacts model and repository wired through the fact-check service and JSON output.
- [x] Consolidate OpenAI/Gemini execution into one OpenAI-compatible adapter path with profile-based key/base-url switching.
