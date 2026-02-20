from __future__ import annotations

from dataclasses import dataclass, field

from .contracts import AspectFinding, FactCheckReport, InvestigationPlan, VerificationCheck


@dataclass
class ResearchCheckArtifact:
    check: VerificationCheck
    attempts: int = 0
    errors: list[str] = field(default_factory=list)
    finding: AspectFinding | None = None


@dataclass
class RunArtifacts:
    claim: str
    normalized_claim: str
    plan_raw: InvestigationPlan | None = None
    plan_normalized: InvestigationPlan | None = None
    research_checks: list[ResearchCheckArtifact] = field(default_factory=list)
    report_raw: FactCheckReport | None = None
    report_final: FactCheckReport | None = None

    def get_or_create_check(self, check: VerificationCheck) -> ResearchCheckArtifact:
        for artifact in self.research_checks:
            if artifact.check.aspect_id == check.aspect_id and artifact.check.question == check.question:
                return artifact
        artifact = ResearchCheckArtifact(check=check)
        self.research_checks.append(artifact)
        return artifact
