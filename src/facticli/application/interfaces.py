from __future__ import annotations

from typing import Protocol

from facticli.core.contracts import (
    AspectFinding,
    ClaimExtractionResult,
    FactCheckReport,
    InvestigationPlan,
    VerificationCheck,
)


class Planner(Protocol):
    async def plan(self, claim: str, max_checks: int) -> InvestigationPlan:
        ...


class Researcher(Protocol):
    async def research(self, claim: str, check: VerificationCheck) -> AspectFinding:
        ...


class Retriever(Protocol):
    async def search(self, queries: list[str], results_per_query: int) -> list[dict[str, object]]:
        ...


class Judge(Protocol):
    async def judge(
        self,
        claim: str,
        plan: InvestigationPlan,
        findings: list[AspectFinding],
    ) -> FactCheckReport:
        ...


class ClaimExtractionBackend(Protocol):
    async def extract(self, input_text: str, max_claims: int) -> ClaimExtractionResult:
        ...
