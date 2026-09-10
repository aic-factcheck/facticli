from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any

from agents import Agent, ModelSettings, Runner, Tool, WebSearchTool
from openai.types.shared import Reasoning

from facticli.application.interfaces import (
    ClaimExtractionBackend,
    Judge,
    Planner,
    Researcher,
    Reviewer,
    SingleAgentChecker,
)
from facticli.brave_search import build_brave_web_search_tool
from facticli.core.constraints import get_constraints
from facticli.core.contracts import (
    AspectFinding,
    ClaimExtractionResult,
    FactCheckReport,
    InvestigationPlan,
    ReviewDecision,
    VerificationCheck,
)
from facticli.core.usage import get_usage_log, record_stage_usage
from facticli.knowledge_store import build_knowledge_store_search_tool
from facticli.skills import load_skill_prompt

from .payloads import judge_payload, research_payload, review_payload, single_agent_payload
from .retry_policy import build_retry_settings

# Default per-check search budget surfaced to the researcher prompt. Anthropic's
# multi-agent research numbers: simple fact-finding needs 3-10 tool calls.
DEFAULT_TOOL_CALL_BUDGET = 8


def build_model_settings(
    *,
    parallel_tool_calls: bool | None,
    reasoning_effort: str | None = None,
    retry_attempts: int = 0,
) -> ModelSettings:
    """Shared ModelSettings: optional reasoning effort and SDK-managed retries.

    Temperature and similar knobs stay unset because OpenAI-compatible
    providers do not accept them uniformly. Reasoning effort is opt-in per
    stage; when unset the provider default applies.
    """
    return ModelSettings(
        parallel_tool_calls=parallel_tool_calls,
        reasoning=Reasoning(effort=reasoning_effort) if reasoning_effort else None,  # type: ignore[arg-type]
        retry=build_retry_settings(retry_attempts),
    )


def build_search_tools(
    *,
    search_provider: str,
    search_context_size: str,
    max_chars_per_field: int,
    max_total_chars: int,
) -> list[Tool]:
    """Search tools for evidence-gathering agents; only these stages carry tool schemas."""
    if search_provider == "openai":
        return [WebSearchTool(search_context_size=search_context_size)]  # type: ignore[arg-type]
    if search_provider == "brave":
        return [build_brave_web_search_tool(max_chars_per_field=max_chars_per_field, max_total_chars=max_total_chars)]
    if search_provider == "knowledge_store":
        return [
            build_knowledge_store_search_tool(
                max_chars_per_field=max_chars_per_field, max_total_chars=max_total_chars
            )
        ]
    raise ValueError(f"Unsupported search provider: {search_provider}")


async def _run_tracked(agent: Agent[None], payload: str, *, max_turns: int, stage: str, model: str) -> Any:
    """Run an agent while recording token usage and latency for the stage."""
    started_at = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    try:
        result = await Runner.run(agent, payload, max_turns=max_turns)
    finally:
        duration = time.monotonic() - start
    usage = getattr(getattr(result, "context_wrapper", None), "usage", None)
    record_stage_usage(
        stage=stage,
        model=model,
        usage=usage,
        duration_seconds=duration,
        started_at=started_at,
    )
    return result


def _dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False)


class CompatiblePlannerAdapter(Planner):
    """Agents SDK planner adapter for OpenAI-compatible chat providers."""

    def __init__(self, model: str, max_turns: int, *, reasoning_effort: str | None = None, retry_attempts: int = 0):
        self._agent: Agent[None] = Agent(
            name="claim_planner",
            instructions=load_skill_prompt("plan"),
            output_type=InvestigationPlan,
            model=model,
            model_settings=build_model_settings(
                parallel_tool_calls=False, reasoning_effort=reasoning_effort, retry_attempts=retry_attempts
            ),
        )
        self._max_turns = max_turns
        self._model = model

    async def plan(self, claim: str, max_checks: int) -> InvestigationPlan:
        """Ask the planning skill to produce at most the configured number of checks."""
        payload = {
            "claim": claim,
            "requirements": {
                "max_checks": max_checks,
                "acceptance_criteria_per_check": True,
                "checks_must_be_independent": True,
            },
        }
        result = await _run_tracked(self._agent, _dumps(payload), max_turns=self._max_turns, stage="plan", model=self._model)
        return result.final_output_as(InvestigationPlan, raise_if_incorrect_type=True)


class CompatibleResearchAdapter(Researcher):
    """Research adapter that executes one check with configured search tooling."""

    def __init__(
        self,
        model: str,
        max_turns: int,
        search_context_size: str,
        search_provider: str,
        *,
        reasoning_effort: str | None = None,
        retry_attempts: int = 0,
        max_chars_per_field: int = 1500,
        max_total_chars: int = 12000,
        tool_call_budget: int = DEFAULT_TOOL_CALL_BUDGET,
    ):
        tools = build_search_tools(
            search_provider=search_provider,
            search_context_size=search_context_size,
            max_chars_per_field=max_chars_per_field,
            max_total_chars=max_total_chars,
        )
        self._agent: Agent[None] = Agent(
            name="check_researcher",
            instructions=load_skill_prompt("research"),
            tools=tools,
            output_type=AspectFinding,
            model=model,
            model_settings=build_model_settings(
                parallel_tool_calls=True, reasoning_effort=reasoning_effort, retry_attempts=retry_attempts
            ),
        )
        self._max_turns = max_turns
        self._search_provider = search_provider
        self._model = model
        self._tool_call_budget = tool_call_budget

    async def research(self, claim: str, check: VerificationCheck) -> AspectFinding:
        """Collect evidence for one check and backfill missing identity fields."""
        payload = research_payload(
            claim=claim,
            check=check,
            search_provider=self._search_provider,
            constraints=get_constraints(),
            usage_log=get_usage_log(),
            tool_call_budget=self._tool_call_budget,
        )
        result = await _run_tracked(
            self._agent,
            _dumps(payload),
            max_turns=self._max_turns,
            stage="research",
            model=self._model,
        )
        finding = result.final_output_as(AspectFinding, raise_if_incorrect_type=True)

        updates: dict[str, str] = {}
        if not finding.aspect_id.strip() or finding.aspect_id != check.aspect_id:
            updates["aspect_id"] = check.aspect_id
        if not finding.question.strip():
            updates["question"] = check.question
        return finding.model_copy(update=updates) if updates else finding


class CompatibleJudgeAdapter(Judge):
    """Judge adapter that synthesizes a final report from structured findings."""

    def __init__(self, model: str, max_turns: int, *, reasoning_effort: str | None = None, retry_attempts: int = 0):
        self._agent: Agent[None] = Agent(
            name="veracity_judge",
            instructions=load_skill_prompt("judge"),
            output_type=FactCheckReport,
            model=model,
            model_settings=build_model_settings(
                parallel_tool_calls=False, reasoning_effort=reasoning_effort, retry_attempts=retry_attempts
            ),
        )
        self._max_turns = max_turns
        self._model = model

    async def judge(
        self,
        claim: str,
        plan: InvestigationPlan,
        findings: list[AspectFinding],
    ) -> FactCheckReport:
        """Request final verdict synthesis from grouped findings and a shared source table."""
        payload = judge_payload(claim=claim, plan=plan, findings=findings)
        result = await _run_tracked(
            self._agent,
            _dumps(payload),
            max_turns=self._max_turns,
            stage="judge",
            model=self._model,
        )
        return result.final_output_as(FactCheckReport, raise_if_incorrect_type=True)


class CompatibleReviewAdapter(Reviewer):
    """Review adapter that grades findings against acceptance criteria."""

    def __init__(self, model: str, max_turns: int, *, reasoning_effort: str | None = None, retry_attempts: int = 0):
        self._agent: Agent[None] = Agent(
            name="evidence_review",
            instructions=load_skill_prompt("review"),
            output_type=ReviewDecision,
            model=model,
            model_settings=build_model_settings(
                parallel_tool_calls=False, reasoning_effort=reasoning_effort, retry_attempts=retry_attempts
            ),
        )
        self._max_turns = max_turns
        self._model = model

    async def review(
        self,
        claim: str,
        plan: InvestigationPlan,
        findings: list[AspectFinding],
        *,
        max_follow_up_checks: int = 2,
        round_index: int = 1,
    ) -> ReviewDecision:
        """Ask the review skill whether extra evidence gathering is required."""
        payload = review_payload(
            claim=claim,
            plan=plan,
            findings=findings,
            max_follow_up_checks=max_follow_up_checks,
            round_index=round_index,
        )
        result = await _run_tracked(
            self._agent,
            _dumps(payload),
            max_turns=self._max_turns,
            stage="review",
            model=self._model,
        )
        return result.final_output_as(ReviewDecision, raise_if_incorrect_type=True)


class CompatibleSingleAgentAdapter(SingleAgentChecker):
    """No-harness baseline: one agent with the search tool produces the whole report."""

    def __init__(
        self,
        model: str,
        max_turns: int,
        search_context_size: str,
        search_provider: str,
        *,
        reasoning_effort: str | None = None,
        retry_attempts: int = 0,
        max_chars_per_field: int = 1500,
        max_total_chars: int = 12000,
    ):
        tools = build_search_tools(
            search_provider=search_provider,
            search_context_size=search_context_size,
            max_chars_per_field=max_chars_per_field,
            max_total_chars=max_total_chars,
        )
        self._agent: Agent[None] = Agent(
            name="single_agent_fact_checker",
            instructions=load_skill_prompt("single_agent"),
            tools=tools,
            output_type=FactCheckReport,
            model=model,
            model_settings=build_model_settings(
                parallel_tool_calls=True, reasoning_effort=reasoning_effort, retry_attempts=retry_attempts
            ),
        )
        self._max_turns = max_turns
        self._search_provider = search_provider
        self._model = model

    async def check(self, claim: str, max_checks: int) -> FactCheckReport:
        payload = single_agent_payload(
            claim=claim,
            max_checks=max_checks,
            search_provider=self._search_provider,
            constraints=get_constraints(),
            usage_log=get_usage_log(),
        )
        result = await _run_tracked(
            self._agent,
            _dumps(payload),
            max_turns=self._max_turns,
            stage="single_agent",
            model=self._model,
        )
        return result.final_output_as(FactCheckReport, raise_if_incorrect_type=True)


class CompatibleClaimExtractionAdapter(ClaimExtractionBackend):
    """Claim extraction adapter for turning prose into check-worthy atomic claims."""

    def __init__(self, model: str, max_turns: int, *, reasoning_effort: str | None = None, retry_attempts: int = 0):
        self._agent: Agent[None] = Agent(
            name="checkworthy_claim_extractor",
            instructions=load_skill_prompt("extract_claims"),
            output_type=ClaimExtractionResult,
            model=model,
            model_settings=build_model_settings(
                parallel_tool_calls=False, reasoning_effort=reasoning_effort, retry_attempts=retry_attempts
            ),
        )
        self._max_turns = max_turns
        self._model = model

    async def extract(self, input_text: str, max_claims: int) -> ClaimExtractionResult:
        """Run extraction instructions with strict limits and coverage requirements."""
        payload = {
            "input_text": input_text,
            "requirements": {
                "max_claims": max_claims,
                "decontextualized": True,
                "atomic_claims": True,
                "maximize_checkworthy_coverage": True,
                "only_directly_mentioned_facts": True,
                "detect_and_report_language": True,
                "write_output_in_input_language": True,
                "preserve_original_diacritics": True,
            },
        }
        result = await _run_tracked(
            self._agent,
            _dumps(payload),
            max_turns=self._max_turns,
            stage="extract_claims",
            model=self._model,
        )
        return result.final_output_as(ClaimExtractionResult, raise_if_incorrect_type=True)
