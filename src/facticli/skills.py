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
    output_model: type[BaseModel]
    uses_web_search: bool = False

    @property
    def description(self) -> str:
        return load_skill_metadata(self.name).get("description", "")


SKILLS: dict[str, SkillSpec] = {
    "plan": SkillSpec(name="plan", output_model=InvestigationPlan),
    "research": SkillSpec(name="research", output_model=AspectFinding, uses_web_search=True),
    "review": SkillSpec(name="review", output_model=ReviewDecision),
    "judge": SkillSpec(name="judge", output_model=FactCheckReport),
    "single_agent": SkillSpec(name="single_agent", output_model=FactCheckReport, uses_web_search=True),
    "extract_claims": SkillSpec(name="extract_claims", output_model=ClaimExtractionResult),
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
    prompt_path = files("facticli.prompts").joinpath(f"{skill_name}.md")
    return parse_frontmatter(prompt_path.read_text(encoding="utf-8"))


def load_skill_prompt(skill_name: str) -> str:
    """Return the instruction body of a skill (frontmatter stripped)."""
    return _read_skill_file(skill_name)[1]


def load_skill_metadata(skill_name: str) -> dict[str, str]:
    """Return the frontmatter of a skill prompt file (may be empty)."""
    return _read_skill_file(skill_name)[0]


def list_skills() -> list[SkillSpec]:
    return list(SKILLS.values())


def get_skill(skill_name: str) -> SkillSpec:
    if skill_name not in SKILLS:
        raise KeyError(f"Unknown skill: {skill_name}")
    return SKILLS[skill_name]
