from __future__ import annotations

import json

from agents import Agent, ModelSettings, Runner

from facticli.agents import build_judge_agent, build_planner_agent, build_research_agent
from facticli.application.interfaces import ClaimExtractionBackend, Judge, Planner, Researcher
from facticli.core.contracts import (
    AspectFinding,
    ClaimExtractionResult,
    FactCheckReport,
    InvestigationPlan,
    VerificationCheck,
)
from facticli.skills import load_skill_prompt


class OpenAIPlannerAdapter(Planner):
    def __init__(self, model: str, max_turns: int):
        self._agent = build_planner_agent(model=model)
        self._max_turns = max_turns

    async def plan(self, claim: str, max_checks: int) -> InvestigationPlan:
        payload = (
            "Build a fact-checking plan for this claim.\n\n"
            f"Claim:\n{claim}\n\n"
            f"Output at most {max_checks} checks."
        )
        result = await Runner.run(self._agent, payload, max_turns=self._max_turns)
        return result.final_output_as(InvestigationPlan, raise_if_incorrect_type=True)


class OpenAIResearchAdapter(Researcher):
    def __init__(
        self,
        model: str,
        max_turns: int,
        search_context_size: str,
        search_provider: str,
    ):
        self._agent = build_research_agent(
            model=model,
            search_context_size=search_context_size,
            search_provider=search_provider,
        )
        self._max_turns = max_turns
        self._search_provider = search_provider

    async def research(self, claim: str, check: VerificationCheck) -> AspectFinding:
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


class OpenAIJudgeAdapter(Judge):
    def __init__(self, model: str, max_turns: int, judge_extra_turns: int):
        self._agent = build_judge_agent(model=model)
        self._max_turns = max_turns
        self._judge_extra_turns = judge_extra_turns

    async def judge(
        self,
        claim: str,
        plan: InvestigationPlan,
        findings: list[AspectFinding],
    ) -> FactCheckReport:
        payload = {
            "claim": claim,
            "plan": plan.model_dump(),
            "findings": [finding.model_dump() for finding in findings],
        }
        result = await Runner.run(
            self._agent,
            json.dumps(payload, indent=2),
            max_turns=self._max_turns + self._judge_extra_turns,
        )
        return result.final_output_as(FactCheckReport, raise_if_incorrect_type=True)


class OpenAIClaimExtractionAdapter(ClaimExtractionBackend):
    def __init__(self, model: str, max_turns: int):
        instructions = load_skill_prompt("extract_claims")
        self._agent: Agent[None] = Agent(
            name="checkworthy_claim_extractor",
            instructions=instructions,
            output_type=ClaimExtractionResult,
            model=model,
            model_settings=ModelSettings(
                temperature=0.1,
                parallel_tool_calls=False,
            ),
        )
        self._max_turns = max_turns

    async def extract(self, input_text: str, max_claims: int) -> ClaimExtractionResult:
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
