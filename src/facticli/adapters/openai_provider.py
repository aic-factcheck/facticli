from __future__ import annotations

import json

from agents import Agent, ModelSettings, Runner, WebSearchTool

from facticli.application.interfaces import ClaimExtractionBackend, Judge, Planner, Researcher, Reviewer
from facticli.brave_search import build_brave_web_search_tool
from facticli.core.contracts import (
    AspectFinding,
    ClaimExtractionResult,
    FactCheckReport,
    InvestigationPlan,
    ReviewDecision,
    VerificationCheck,
)
from facticli.skills import load_skill_prompt


class CompatiblePlannerAdapter(Planner):
    """Agents SDK planner adapter for OpenAI-compatible chat providers."""
    def __init__(self, model: str, max_turns: int):
        self._agent: Agent[None] = Agent(
            name="claim_planner",
            instructions=load_skill_prompt("plan"),
            output_type=InvestigationPlan,
            model=model,
            model_settings=ModelSettings(
                temperature=0.15,
                parallel_tool_calls=False,
            ),
        )
        self._max_turns = max_turns

    async def plan(self, claim: str, max_checks: int) -> InvestigationPlan:
        """Ask the planning skill to produce at most the configured number of checks."""
        payload = (
            "Build a fact-checking plan for this claim.\n\n"
            f"Claim:\n{claim}\n\n"
            f"Output at most {max_checks} checks."
        )
        result = await Runner.run(self._agent, payload, max_turns=self._max_turns)
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
        else:
            raise ValueError(f"Unsupported search provider: {search_provider}")

        self._agent: Agent[None] = Agent(
            name="check_researcher",
            instructions=load_skill_prompt("research"),
            tools=tools,
            output_type=AspectFinding,
            model=model,
            model_settings=ModelSettings(
                temperature=0.2,
                parallel_tool_calls=True,
            ),
        )
        self._max_turns = max_turns
        self._search_provider = search_provider

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
        result = await Runner.run(self._agent, json.dumps(payload, indent=2), max_turns=self._max_turns)
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
                temperature=0.1,
                parallel_tool_calls=False,
            ),
        )
        self._max_turns = max_turns

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
        result = await Runner.run(
            self._agent,
            json.dumps(payload, indent=2),
            max_turns=self._max_turns,
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
                temperature=0.1,
                parallel_tool_calls=False,
            ),
        )
        self._max_turns = max_turns

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
        result = await Runner.run(
            self._agent,
            json.dumps(payload, indent=2),
            max_turns=self._max_turns,
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
                temperature=0.1,
                parallel_tool_calls=False,
            ),
        )
        self._max_turns = max_turns

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
            },
        }
        result = await Runner.run(
            self._agent,
            json.dumps(payload, indent=2),
            max_turns=self._max_turns,
        )
        return result.final_output_as(ClaimExtractionResult, raise_if_incorrect_type=True)
