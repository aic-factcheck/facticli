from __future__ import annotations

from dataclasses import dataclass, field

from .contracts import AspectFinding, FactCheckReport, InvestigationPlan, ReviewDecision, VerificationCheck


@dataclass
class ResearchCheckArtifact:
    check: VerificationCheck
    attempts: int = 0
    errors: list[str] = field(default_factory=list)
    finding: AspectFinding | None = None


@dataclass
class ReviewRoundArtifact:
    round_index: int
    input_plan: InvestigationPlan
    input_findings: list[AspectFinding]
    decision: ReviewDecision | None = None
    follow_up_plan: InvestigationPlan | None = None


@dataclass
class RunArtifacts:
    claim: str
    normalized_claim: str
    plan_raw: InvestigationPlan | None = None
    plan_normalized: InvestigationPlan | None = None
    research_checks: list[ResearchCheckArtifact] = field(default_factory=list)
    review_rounds: list[ReviewRoundArtifact] = field(default_factory=list)
    report_raw: FactCheckReport | None = None
    report_final: FactCheckReport | None = None

    def get_or_create_check(self, check: VerificationCheck) -> ResearchCheckArtifact:
        for artifact in self.research_checks:
            if artifact.check.aspect_id == check.aspect_id and artifact.check.question == check.question:
                return artifact
        artifact = ResearchCheckArtifact(check=check)
        self.research_checks.append(artifact)
        return artifact

    def add_review_round(
        self,
        *,
        round_index: int,
        input_plan: InvestigationPlan,
        input_findings: list[AspectFinding],
    ) -> ReviewRoundArtifact:
        artifact = ReviewRoundArtifact(
            round_index=round_index,
            input_plan=input_plan,
            input_findings=list(input_findings),
        )
        self.review_rounds.append(artifact)
        return artifact
