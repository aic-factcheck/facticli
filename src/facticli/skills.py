from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files

from pydantic import BaseModel

from .core.contracts import AspectFinding, ClaimExtractionResult, FactCheckReport, InvestigationPlan, ReviewDecision


@dataclass(frozen=True)
class SkillSpec:
    """One prompt skill: a markdown instruction file bound to an output contract.

    Prompt files carry SKILL.md-style YAML frontmatter (``name``,
    ``description``). The description shown by ``facticli skills`` comes from
    the file so the prompt stays the single source of truth; the registry only
    binds the file to its typed output model and tool needs.
    """
    name: str
    prompt_file: str
    output_model: type[BaseModel]
    uses_web_search: bool = False
    public: bool = True
    fallback_description: str = ""

    @property
    def description(self) -> str:
        meta = load_skill_metadata(self.name)
        return meta.get("description") or self.fallback_description


SKILLS: dict[str, SkillSpec] = {
    "plan": SkillSpec(
        name="plan",
        prompt_file="plan.md",
        output_model=InvestigationPlan,
        fallback_description="Decompose claim into independent, parallelizable verification checks.",
    ),
    "research": SkillSpec(
        name="research",
        prompt_file="research.md",
        output_model=AspectFinding,
        uses_web_search=True,
        fallback_description="Investigate one check with web search and evidence extraction.",
    ),
    "review": SkillSpec(
        name="review",
        prompt_file="review.md",
        output_model=ReviewDecision,
        fallback_description="Grade findings against acceptance criteria and request bounded follow-up.",
    ),
    "judge": SkillSpec(
        name="judge",
        prompt_file="judge.md",
        output_model=FactCheckReport,
        fallback_description="Synthesize findings into a final veracity verdict with justification.",
    ),
    "single_agent": SkillSpec(
        name="single_agent",
        prompt_file="single_agent.md",
        output_model=FactCheckReport,
        uses_web_search=True,
        fallback_description="No-harness baseline: one agent researches and judges the claim in one run.",
    ),
    "extract_claims": SkillSpec(
        name="extract_claims",
        prompt_file="extract_claims.md",
        output_model=ClaimExtractionResult,
        fallback_description="Extract decontextualized atomic check-worthy claims from arbitrary text.",
    ),
}


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Split ``---`` delimited ``key: value`` frontmatter from a markdown body.

    Returns ``({}, text)`` when the file has no frontmatter. Only flat string
    values are supported, which is all SKILL.md-style headers need.
    """
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    meta: dict[str, str] = {}
    for index in range(1, len(lines)):
        line = lines[index]
        if line.strip() == "---":
            body = "\n".join(lines[index + 1 :]).lstrip("\n")
            return meta, body
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if ":" not in line:
            return {}, text
        key, value = line.split(":", 1)
        meta[key.strip().lower()] = value.strip().strip('"').strip("'")
    return {}, text


@lru_cache(maxsize=None)
def _read_skill_file(skill_name: str) -> tuple[dict[str, str], str]:
    if skill_name not in SKILLS:
        raise KeyError(f"Unknown skill: {skill_name}")
    prompt_file = SKILLS[skill_name].prompt_file
    prompt_path = files("facticli.prompts").joinpath(prompt_file)
    return parse_frontmatter(prompt_path.read_text(encoding="utf-8"))


def load_skill_prompt(skill_name: str) -> str:
    """Return the instruction body of a skill (frontmatter stripped)."""
    return _read_skill_file(skill_name)[1]


def load_skill_metadata(skill_name: str) -> dict[str, str]:
    """Return the frontmatter of a skill prompt file (may be empty)."""
    return _read_skill_file(skill_name)[0]


def list_skills() -> list[SkillSpec]:
    return [skill for skill in SKILLS.values() if skill.public]


def get_skill(skill_name: str) -> SkillSpec:
    if skill_name not in SKILLS:
        raise KeyError(f"Unknown skill: {skill_name}")
    return SKILLS[skill_name]
