from __future__ import annotations

from agents import Agent, ModelSettings, WebSearchTool

from .skills import load_skill_prompt
from .types import AspectFinding, FactCheckReport, InvestigationPlan


def build_planner_agent(model: str) -> Agent[None]:
    return Agent(
        name="claim_planner",
        instructions=load_skill_prompt("plan"),
        output_type=InvestigationPlan,
        model=model,
        model_settings=ModelSettings(
            temperature=0.15,
            parallel_tool_calls=False,
        ),
    )


def build_research_agent(model: str, search_context_size: str) -> Agent[None]:
    return Agent(
        name="check_researcher",
        instructions=load_skill_prompt("research"),
        tools=[WebSearchTool(search_context_size=search_context_size)],
        output_type=AspectFinding,
        model=model,
        model_settings=ModelSettings(
            temperature=0.2,
            parallel_tool_calls=True,
        ),
    )


def build_judge_agent(model: str) -> Agent[None]:
    return Agent(
        name="veracity_judge",
        instructions=load_skill_prompt("judge"),
        output_type=FactCheckReport,
        model=model,
        model_settings=ModelSettings(
            temperature=0.1,
            parallel_tool_calls=False,
        ),
    )

