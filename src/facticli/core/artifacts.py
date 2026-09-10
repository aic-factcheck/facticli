from __future__ import annotations

from pydantic import BaseModel, Field

from .constraints import ResearchConstraints
from .contracts import AspectFinding, FactCheckReport, InvestigationPlan, ReviewDecision, SourceEvidence, VerificationCheck
from .usage import BudgetStatus, StageUsageEvent, UsageSummary


class ResearchCheckArtifact(BaseModel):
    """Debug artifact for one research check, including attempts and errors."""
    check: VerificationCheck
    attempts: int = 0
    errors: list[str] = Field(default_factory=list)
    error_kinds: list[str] = Field(
        default_factory=list,
        description="ErrorKind value per failed attempt, aligned with `errors`.",
    )
    finding: AspectFinding | None = None
    removed_sources: list[SourceEvidence] = Field(default_factory=list)


class ReviewRoundArtifact(BaseModel):
    """Snapshot of one review round and its follow-up decision."""
    round_index: int
    input_plan: InvestigationPlan
    input_findings: list[AspectFinding]
    decision: ReviewDecision | None = None
    follow_up_plan: InvestigationPlan | None = None


class RunArtifacts(BaseModel):
    """Aggregated per-run artifacts used for inspection and replayability."""
    claim: str
    normalized_claim: str
    claim_id: str | None = None
    constraints: ResearchConstraints | None = None
    plan_raw: InvestigationPlan | None = None
    plan_normalized: InvestigationPlan | None = None
    research_checks: list[ResearchCheckArtifact] = Field(default_factory=list)
    review_rounds: list[ReviewRoundArtifact] = Field(default_factory=list)
    report_raw: FactCheckReport | None = None
    report_final: FactCheckReport | None = None
    started_at: str | None = None
    duration_seconds: float | None = None
    usage_events: list[StageUsageEvent] = Field(default_factory=list)
    usage_summary: UsageSummary | None = None
    budget: BudgetStatus | None = None
    strategy: str = "pipeline"
    stage_models: dict[str, str] = Field(
        default_factory=dict,
        description="Resolved model per stage when per-stage routing was used.",
    )
    citation_check: dict[str, int] | None = Field(
        default=None,
        description="Counts per url_status from the optional citation health pass.",
    )

    def get_or_create_check(self, check: VerificationCheck) -> ResearchCheckArtifact:
        """Return existing check artifact by identity or create a new slot."""
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
        """Append and return a review-round artifact with input snapshots."""
        artifact = ReviewRoundArtifact(
            round_index=round_index,
            input_plan=input_plan,
            input_findings=list(input_findings),
        )
        self.review_rounds.append(artifact)
        return artifact
