from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from agents import Agent, ModelSettings, RunHooks, Runner, Tool, WebSearchTool
from agents.models.interface import Model
from agents.run_error_handlers import RunErrorHandlerInput, RunErrorHandlers
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
    StopReason,
    VerificationCheck,
)
from facticli.core.retrieval import OPEN_PAGE, RetrievalJournal, RetrievalRecord, get_journal
from facticli.core.search_cache import SearchCache
from facticli.core.usage import StageUsageEvent, apply_usage, get_usage_log, track_usage
from facticli.knowledge_store import build_knowledge_store_search_tool
from facticli.skills import load_skill_prompt

from .payloads import dated, judge_payload, research_payload, review_payload, single_agent_payload
from .retry_policy import build_retry_settings

HOSTED_SEARCH_PROVIDER = "openai"
# Ask the Responses API for the complete list of URLs each hosted search
# consulted, so cited URLs can be checked against it afterwards.
HOSTED_SEARCH_INCLUDE = "web_search_call.action.sources"

WRAP_UP_NOTICE = (
    "HARNESS NOTICE: The turn limit for this task has been reached and the search tools are now disabled. "
    "Do not request further tool calls. Return your final structured output now, based only on the evidence "
    "already retrieved above. Where that evidence does not settle a question, say so: use the `insufficient` "
    "signal or lower your confidence, and record what is still missing. Do not fill gaps from memory."
)
CUT_SHORT_GAP = (
    "The single-agent run was cut short at {limit}; the report was written from the evidence gathered up to that point."
)


def build_model_settings(
    *,
    parallel_tool_calls: bool | None,
    reasoning_effort: str | None = None,
    retry_attempts: int = 0,
    response_include: list[str] | None = None,
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
        response_include=response_include,  # type: ignore[arg-type]
    )


def build_search_tools(
    *,
    search_provider: str,
    search_context_size: str,
    max_chars_per_field: int,
    max_total_chars: int,
    search_results_per_query: int,
    cache: SearchCache | None = None,
) -> list[Tool]:
    """Search tools for evidence-gathering agents; only these stages carry tool schemas."""
    if search_provider == HOSTED_SEARCH_PROVIDER:
        return [WebSearchTool(search_context_size=search_context_size)]  # type: ignore[arg-type]
    if search_provider == "brave":
        return [
            build_brave_web_search_tool(
                max_chars_per_field=max_chars_per_field,
                max_total_chars=max_total_chars,
                default_count=search_results_per_query,
                cache=cache,
            )
        ]
    if search_provider == "knowledge_store":
        return [
            build_knowledge_store_search_tool(
                max_chars_per_field=max_chars_per_field,
                max_total_chars=max_total_chars,
                default_count=search_results_per_query,
            )
        ]
    raise ValueError(f"Unsupported search provider: {search_provider}")


def record_hosted_search_calls(response: Any, journal: RetrievalJournal) -> None:
    """Copy the hosted web-search calls of one model response into the retrieval journal.

    Hosted search runs on the provider's side, so the harness never sees page
    text, only (when requested with ``HOSTED_SEARCH_INCLUDE``) the URLs each
    call consulted. That is enough for URL-level provenance; quotes cannot be
    checked. A search that reports no source list leaves the journal incomplete.
    """
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) != "web_search_call":
            continue
        action = getattr(item, "action", None)
        action_type = getattr(action, "type", None)
        if action_type == "search":
            sources = getattr(action, "sources", None)
            if sources is None:
                journal.complete = False
            queries = getattr(action, "queries", None) or [getattr(action, "query", None)]
            journal.add(
                RetrievalRecord(
                    provider=HOSTED_SEARCH_PROVIDER,
                    query=" | ".join(query for query in queries if query) or None,
                    urls=[source.url for source in sources or [] if getattr(source, "url", None)],
                )
            )
        elif action_type in ("open_page", "find_in_page"):
            url = getattr(action, "url", None)
            journal.add(RetrievalRecord(provider=HOSTED_SEARCH_PROVIDER, kind=OPEN_PAGE, urls=[url] if url else []))


class _LedgerHooks(RunHooks[Any]):
    """Keep the usage ledger and the retrieval journal current as model responses arrive.

    The SDK hands usage and raw responses back only with a successful result.
    Copying them per response means a run that later exceeds its turn limit,
    fails validation, or is cancelled by a timeout still pays into the ledger
    and leaves its hosted-search receipts.
    """

    def __init__(self, event: StageUsageEvent | None) -> None:
        self._event = event

    async def on_llm_end(self, context: Any, agent: Any, response: Any) -> None:
        if self._event is not None:
            apply_usage(self._event, context.usage)
        journal = get_journal()
        if journal is not None:
            record_hosted_search_calls(response, journal)


class _StageRunner:
    """One stage's agent, with usage tracking and a soft landing at the turn limit.

    A tool-using run that reaches its turn limit gets one more model call with
    tools disabled, so the evidence it already paid for becomes a finding
    written on less evidence instead of a failed check. The call is recorded as
    its own usage event; if it fails as well, the turn-limit error propagates.
    """

    def __init__(
        self,
        *,
        name: str,
        stage: str,
        output_type: type[Any],
        model: str | Model,
        max_turns: int,
        reasoning_effort: str | None = None,
        retry_attempts: int = 0,
        tools: Sequence[Tool] = (),
    ) -> None:
        hosted_search = any(isinstance(tool, WebSearchTool) for tool in tools)
        settings = build_model_settings(
            parallel_tool_calls=bool(tools),
            reasoning_effort=reasoning_effort,
            retry_attempts=retry_attempts,
            response_include=[HOSTED_SEARCH_INCLUDE] if hosted_search else None,
        )
        self._agent: Agent[None] = Agent(
            name=name,
            instructions=load_skill_prompt(stage),
            tools=list(tools),
            output_type=output_type,
            model=model,
            model_settings=settings,
        )
        # Same instructions and tool schemas, so the prompt prefix stays
        # cache-identical; the model is only forbidden to call a tool.
        self._wrap_up_agent = (
            self._agent.clone(model_settings=replace(settings, tool_choice="none", parallel_tool_calls=None))
            if tools
            else None
        )
        self._output_type = output_type
        self._max_turns = max_turns
        self._stage = stage
        self._model_label = model if isinstance(model, str) else type(model).__name__

    async def run(self, payload: dict[str, Any]) -> tuple[Any, bool]:
        """Run the stage; return the typed output and whether the wrap-up call produced it."""
        message = json.dumps(payload, indent=2, ensure_ascii=False)
        wrap_up_agent = self._wrap_up_agent
        if wrap_up_agent is None:
            return await self._tracked_run(self._agent, message, max_turns=self._max_turns), False

        journal = get_journal()
        if journal is not None:
            # Function tools journal every result and hosted search reports its
            # sources, so from here on a URL the journal lacks was not retrieved.
            journal.complete = True
        wrapped_up = False

        async def wrap_up(data: RunErrorHandlerInput[None]) -> Any | None:
            nonlocal wrapped_up
            try:
                output = await self._tracked_run(
                    wrap_up_agent,
                    [*data.run_data.history, {"role": "user", "content": WRAP_UP_NOTICE}],
                    max_turns=1,
                    wrap_up=True,
                )
            except Exception:
                return None  # the SDK raises the turn-limit error; the failed call stays in the ledger
            wrapped_up = True
            return output

        output = await self._tracked_run(
            self._agent, message, max_turns=self._max_turns, error_handlers={"max_turns": wrap_up}
        )
        return output, wrapped_up

    async def _tracked_run(
        self,
        agent: Agent[None],
        message: str | list[Any],
        *,
        max_turns: int,
        wrap_up: bool = False,
        error_handlers: RunErrorHandlers[None] | None = None,
    ) -> Any:
        with track_usage(stage=self._stage, model=self._model_label, wrap_up=wrap_up) as event:
            result = await Runner.run(
                agent, message, max_turns=max_turns, hooks=_LedgerHooks(event), error_handlers=error_handlers
            )
            return result.final_output_as(self._output_type, raise_if_incorrect_type=True)


class CompatiblePlannerAdapter(Planner):
    """Agents SDK planner adapter for OpenAI-compatible chat providers."""

    def __init__(
        self,
        model: str | Model,
        max_turns: int,
        *,
        reasoning_effort: str | None = None,
        retry_attempts: int = 0,
    ):
        self._runner = _StageRunner(
            name="claim_planner",
            stage="plan",
            output_type=InvestigationPlan,
            model=model,
            max_turns=max_turns,
            reasoning_effort=reasoning_effort,
            retry_attempts=retry_attempts,
        )

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
        plan, _ = await self._runner.run(payload)
        return plan


class CompatibleResearchAdapter(Researcher):
    """Research adapter that executes one check with the given search tools."""

    def __init__(
        self,
        model: str | Model,
        max_turns: int,
        tools: Sequence[Tool],
        search_provider: str,
        *,
        tool_call_budget: int,
        reasoning_effort: str | None = None,
        retry_attempts: int = 0,
    ):
        self._runner = _StageRunner(
            name="check_researcher",
            stage="research",
            output_type=AspectFinding,
            model=model,
            max_turns=max_turns,
            reasoning_effort=reasoning_effort,
            retry_attempts=retry_attempts,
            tools=tools,
        )
        self._search_provider = search_provider
        self._tool_call_budget = tool_call_budget

    async def research(self, claim: str, check: VerificationCheck) -> AspectFinding:
        """Collect evidence for one check and backfill missing identity fields."""
        constraints = get_constraints()
        payload = research_payload(
            claim=claim,
            check=check,
            search_provider=self._search_provider,
            constraints=constraints,
            usage_log=get_usage_log(),
            tool_call_budget=self._tool_call_budget,
        )
        finding, wrapped_up = await self._runner.run(dated(payload, constraints))
        finding = finding.without_harness_fields()

        updates: dict[str, Any] = {}
        if not finding.aspect_id.strip() or finding.aspect_id != check.aspect_id:
            updates["aspect_id"] = check.aspect_id
        if not finding.question.strip():
            updates["question"] = check.question
        journal = get_journal()
        if wrapped_up:
            updates["stop_reason"] = StopReason.TURN_LIMIT
        elif journal is not None and journal.refusals:
            updates["stop_reason"] = StopReason.SEARCH_CAP
        return finding.model_copy(update=updates) if updates else finding


class CompatibleJudgeAdapter(Judge):
    """Judge adapter that synthesizes a final report from structured findings."""

    def __init__(
        self,
        model: str | Model,
        max_turns: int,
        *,
        reasoning_effort: str | None = None,
        retry_attempts: int = 0,
    ):
        self._runner = _StageRunner(
            name="veracity_judge",
            stage="judge",
            output_type=FactCheckReport,
            model=model,
            max_turns=max_turns,
            reasoning_effort=reasoning_effort,
            retry_attempts=retry_attempts,
        )

    async def judge(
        self,
        claim: str,
        plan: InvestigationPlan,
        findings: list[AspectFinding],
    ) -> FactCheckReport:
        """Request final verdict synthesis from grouped findings and a shared source table."""
        payload = judge_payload(claim=claim, plan=plan, findings=findings)
        report, _ = await self._runner.run(dated(payload, get_constraints()))
        return report.without_harness_fields()


class CompatibleReviewAdapter(Reviewer):
    """Review adapter that grades findings against acceptance criteria."""

    def __init__(
        self,
        model: str | Model,
        max_turns: int,
        *,
        reasoning_effort: str | None = None,
        retry_attempts: int = 0,
    ):
        self._runner = _StageRunner(
            name="evidence_review",
            stage="review",
            output_type=ReviewDecision,
            model=model,
            max_turns=max_turns,
            reasoning_effort=reasoning_effort,
            retry_attempts=retry_attempts,
        )

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
        decision, _ = await self._runner.run(dated(payload, get_constraints()))
        return decision


class CompatibleSingleAgentAdapter(SingleAgentChecker):
    """No-harness baseline: one agent with the search tools produces the whole report."""

    def __init__(
        self,
        model: str | Model,
        max_turns: int,
        tools: Sequence[Tool],
        search_provider: str,
        *,
        tool_call_budget: int,
        reasoning_effort: str | None = None,
        retry_attempts: int = 0,
    ):
        self._runner = _StageRunner(
            name="single_agent_fact_checker",
            stage="single_agent",
            output_type=FactCheckReport,
            model=model,
            max_turns=max_turns,
            reasoning_effort=reasoning_effort,
            retry_attempts=retry_attempts,
            tools=tools,
        )
        self._search_provider = search_provider
        self._tool_call_budget = tool_call_budget

    async def check(self, claim: str, max_checks: int) -> FactCheckReport:
        constraints = get_constraints()
        payload = single_agent_payload(
            claim=claim,
            max_checks=max_checks,
            search_provider=self._search_provider,
            constraints=constraints,
            usage_log=get_usage_log(),
            tool_call_budget=self._tool_call_budget,
        )
        report, wrapped_up = await self._runner.run(dated(payload, constraints))
        report = report.without_harness_fields()
        journal = get_journal()
        limit = "its turn limit" if wrapped_up else "the search cap" if journal is not None and journal.refusals else None
        if limit:
            return report.model_copy(update={"evidence_gaps": [*report.evidence_gaps, CUT_SHORT_GAP.format(limit=limit)]})
        return report


class CompatibleClaimExtractionAdapter(ClaimExtractionBackend):
    """Claim extraction adapter for turning prose into check-worthy atomic claims."""

    def __init__(
        self,
        model: str | Model,
        max_turns: int,
        *,
        reasoning_effort: str | None = None,
        retry_attempts: int = 0,
    ):
        self._runner = _StageRunner(
            name="checkworthy_claim_extractor",
            stage="extract_claims",
            output_type=ClaimExtractionResult,
            model=model,
            max_turns=max_turns,
            reasoning_effort=reasoning_effort,
            retry_attempts=retry_attempts,
        )

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
        result, _ = await self._runner.run(payload)
        return result
