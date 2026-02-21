from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files

from pydantic import BaseModel

from .types import AspectFinding, ClaimExtractionResult, FactCheckReport, InvestigationPlan


@dataclass(frozen=True)
class SkillSpec:
    name: str
    description: str
    prompt_file: str
    output_model: type[BaseModel]
    uses_web_search: bool = False
    public: bool = True


SKILLS: dict[str, SkillSpec] = {
    "plan": SkillSpec(
        name="plan",
        description="Decompose claim into independent, parallelizable verification checks.",
        prompt_file="plan.md",
        output_model=InvestigationPlan,
        uses_web_search=False,
    ),
    "research": SkillSpec(
        name="research",
        description="Investigate one check with open web search and evidence extraction.",
        prompt_file="research.md",
        output_model=AspectFinding,
        uses_web_search=True,
    ),
    "judge": SkillSpec(
        name="judge",
        description="Synthesize findings into a final veracity verdict with justification.",
        prompt_file="judge.md",
        output_model=FactCheckReport,
        uses_web_search=False,
    ),
    "extract_claims": SkillSpec(
        name="extract_claims",
        description="Extract decontextualized atomic check-worthy claims from arbitrary text.",
        prompt_file="extract_claims.md",
        output_model=ClaimExtractionResult,
        uses_web_search=False,
    ),
}


@lru_cache(maxsize=None)
def load_skill_prompt(skill_name: str) -> str:
    if skill_name not in SKILLS:
        raise KeyError(f"Unknown skill: {skill_name}")

    prompt_file = SKILLS[skill_name].prompt_file
    prompt_path = files("facticli.prompts").joinpath(prompt_file)
    return prompt_path.read_text(encoding="utf-8")


def list_skills() -> list[SkillSpec]:
    return [skill for skill in SKILLS.values() if skill.public]
