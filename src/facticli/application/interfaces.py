from __future__ import annotations

from typing import Protocol

from facticli.core.contracts import (
    AspectFinding,
    ClaimExtractionResult,
    FactCheckReport,
    InvestigationPlan,
    ReviewDecision,
    VerificationCheck,
)


class Planner(Protocol):
    async def plan(self, claim: str, max_checks: int) -> InvestigationPlan:
        ...


class Researcher(Protocol):
    async def research(self, claim: str, check: VerificationCheck) -> AspectFinding:
        ...


class Judge(Protocol):
    async def judge(
        self,
        claim: str,
        plan: InvestigationPlan,
        findings: list[AspectFinding],
    ) -> FactCheckReport:
        ...


class Reviewer(Protocol):
    async def review(
        self,
        claim: str,
        plan: InvestigationPlan,
        findings: list[AspectFinding],
    ) -> ReviewDecision:
        ...


class ClaimExtractionBackend(Protocol):
    async def extract(self, input_text: str, max_claims: int) -> ClaimExtractionResult:
        ...
