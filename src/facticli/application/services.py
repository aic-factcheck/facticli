from __future__ import annotations

from dataclasses import dataclass

from facticli.core.artifacts import RunArtifacts
from facticli.core.contracts import AspectFinding, ClaimExtractionResult, FactCheckReport, InvestigationPlan

from .repository import RunArtifactRepository
from .stages import ClaimExtractionStage, JudgeStage, PlanStage, ResearchStage


@dataclass
class FactCheckRun:
    claim: str
    plan: InvestigationPlan
    findings: list[AspectFinding]
    report: FactCheckReport
    artifacts: RunArtifacts


@dataclass(frozen=True)
class FactCheckService:
    plan_stage: PlanStage
    research_stage: ResearchStage
    judge_stage: JudgeStage
    artifact_repository: RunArtifactRepository | None = None

    async def check_claim(self, claim: str) -> FactCheckRun:
        normalized_claim = claim.strip()
        if not normalized_claim:
            raise ValueError("Claim is empty.")

        artifacts = RunArtifacts(claim=claim, normalized_claim=normalized_claim)
        plan = await self.plan_stage.execute(claim=normalized_claim, artifacts=artifacts)
        findings = await self.research_stage.execute(
            claim=normalized_claim,
            plan=plan,
            artifacts=artifacts,
        )
        report = await self.judge_stage.execute(
            claim=normalized_claim,
            plan=plan,
            findings=findings,
            artifacts=artifacts,
        )

        if self.artifact_repository is not None:
            self.artifact_repository.save(artifacts)

        return FactCheckRun(
            claim=normalized_claim,
            plan=plan,
            findings=findings,
            report=report,
            artifacts=artifacts,
        )


@dataclass(frozen=True)
class ClaimExtractionService:
    extraction_stage: ClaimExtractionStage

    async def extract_claims(self, input_text: str) -> ClaimExtractionResult:
        return await self.extraction_stage.execute(input_text)
