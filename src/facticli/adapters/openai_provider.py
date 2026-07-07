from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any

from agents import Agent, ModelSettings, Runner, WebSearchTool

from facticli.application.interfaces import ClaimExtractionBackend, Judge, Planner, Researcher, Reviewer
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
from facticli.core.usage import record_stage_usage
from facticli.knowledge_store import build_knowledge_store_search_tool
from facticli.skills import load_skill_prompt


async def _run_tracked(agent: Agent[None], payload: str, *, max_turns: int, stage: str, model: str) -> Any:
    """Run an agent while recording token usage and latency for the stage."""
    started_at = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    result = await Runner.run(agent, payload, max_turns=max_turns)
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


class CompatiblePlannerAdapter(Planner):
    """Agents SDK planner adapter for OpenAI-compatible chat providers."""
    def __init__(self, model: str, max_turns: int):
        self._agent: Agent[None] = Agent(
            name="claim_planner",
            instructions=load_skill_prompt("plan"),
            output_type=InvestigationPlan,
            model=model,
            model_settings=ModelSettings(
                parallel_tool_calls=False,
            ),
        )
        self._max_turns = max_turns
        self._model = model

    async def plan(self, claim: str, max_checks: int) -> InvestigationPlan:
        """Ask the planning skill to produce at most the configured number of checks."""
        payload = (
            "Build a fact-checking plan for this claim.\n\n"
            f"Claim:\n{claim}\n\n"
            f"Output at most {max_checks} checks."
        )
        result = await _run_tracked(self._agent, payload, max_turns=self._max_turns, stage="plan", model=self._model)
        return result.final_output_as(InvestigationPlan, raise_if_incorrect_type=True)


class CompatibleResearchAdapter(Researcher):
    """Research adapter that executes one check with configured search tooling."""
    def __init__(
        self,
        model: str,
        max_turns: int,
        search_context_size: str,
        search_provider: str,
    ):
        if search_provider == "openai":
            tools = [WebSearchTool(search_context_size=search_context_size)]
        elif search_provider == "brave":
            tools = [build_brave_web_search_tool()]
        elif search_provider == "knowledge_store":
            tools = [build_knowledge_store_search_tool()]
        else:
            raise ValueError(f"Unsupported search provider: {search_provider}")

        self._agent: Agent[None] = Agent(
            name="check_researcher",
            instructions=load_skill_prompt("research"),
            tools=tools,
            output_type=AspectFinding,
            model=model,
            model_settings=ModelSettings(
                parallel_tool_calls=True,
            ),
        )
        self._max_turns = max_turns
        self._search_provider = search_provider
        self._model = model

    async def research(self, claim: str, check: VerificationCheck) -> AspectFinding:
        """Collect evidence for one check and backfill missing identity fields."""
        payload = {
            "claim": claim,
            "check": check.model_dump(),
            "requirements": {
                "min_sources": 2,
                "must_use_search_tool": True,
                "preferred_provider": self._search_provider,
            },
        }
        constraints = get_constraints()
        if constraints is not None and (constraints.claim_date or constraints.blocked_domains):
            payload["constraints"] = {
                "evidence_cutoff_date": constraints.claim_date,
                "blocked_domains": constraints.blocked_domains,
                "instruction": (
                    "Only use evidence published on or before the cutoff date (if set). "
                    "Never cite or rely on sources from the blocked domains; these are "
                    "fact-checking sites excluded to keep the verdict independent."
                ),
            }
        result = await _run_tracked(
            self._agent,
            json.dumps(payload, indent=2),
            max_turns=self._max_turns,
            stage="research",
            model=self._model,
        )
        finding = result.final_output_as(AspectFinding, raise_if_incorrect_type=True)

        updates: dict[str, str] = {}
        if not finding.aspect_id.strip():
            updates["aspect_id"] = check.aspect_id
        if not finding.question.strip():
            updates["question"] = check.question
        return finding.model_copy(update=updates) if updates else finding


class CompatibleJudgeAdapter(Judge):
    """Judge adapter that synthesizes a final report from structured findings."""
    def __init__(self, model: str, max_turns: int):
        self._agent: Agent[None] = Agent(
            name="veracity_judge",
            instructions=load_skill_prompt("judge"),
            output_type=FactCheckReport,
            model=model,
            model_settings=ModelSettings(
                parallel_tool_calls=False,
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
        """Request final verdict synthesis from claim, plan, and findings."""
        payload = {
            "claim": claim,
            "plan": plan.model_dump(),
            "findings": [finding.model_dump() for finding in findings],
        }
        result = await _run_tracked(
            self._agent,
            json.dumps(payload, indent=2),
            max_turns=self._max_turns,
            stage="judge",
            model=self._model,
        )
        return result.final_output_as(FactCheckReport, raise_if_incorrect_type=True)


class CompatibleReviewAdapter(Reviewer):
    """Review adapter that requests targeted retries or follow-up checks."""
    def __init__(self, model: str, max_turns: int):
        self._agent: Agent[None] = Agent(
            name="evidence_review",
            instructions=load_skill_prompt("review"),
            output_type=ReviewDecision,
            model=model,
            model_settings=ModelSettings(
                parallel_tool_calls=False,
            ),
        )
        self._max_turns = max_turns
        self._model = model

    async def review(
        self,
        claim: str,
        plan: InvestigationPlan,
        findings: list[AspectFinding],
    ) -> ReviewDecision:
        """Ask the review skill whether extra evidence gathering is required."""
        payload = {
            "claim": claim,
            "plan": plan.model_dump(),
            "findings": [finding.model_dump() for finding in findings],
        }
        result = await _run_tracked(
            self._agent,
            json.dumps(payload, indent=2),
            max_turns=self._max_turns,
            stage="review",
            model=self._model,
        )
        return result.final_output_as(ReviewDecision, raise_if_incorrect_type=True)


class CompatibleClaimExtractionAdapter(ClaimExtractionBackend):
    """Claim extraction adapter for turning prose into check-worthy atomic claims."""
    def __init__(self, model: str, max_turns: int):
        self._agent: Agent[None] = Agent(
            name="checkworthy_claim_extractor",
            instructions=load_skill_prompt("extract_claims"),
            output_type=ClaimExtractionResult,
            model=model,
            model_settings=ModelSettings(
                parallel_tool_calls=False,
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
            json.dumps(payload, indent=2),
            max_turns=self._max_turns,
            stage="extract_claims",
            model=self._model,
        )
        return result.final_output_as(ClaimExtractionResult, raise_if_incorrect_type=True)
