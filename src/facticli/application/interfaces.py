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
    """Builds an investigation plan from a user claim."""

    async def plan(self, claim: str, max_checks: int) -> InvestigationPlan:
        """Return bounded, independent checks that structure downstream research."""
        ...


class Researcher(Protocol):
    """Executes one verification check using external evidence."""

    async def research(self, claim: str, check: VerificationCheck) -> AspectFinding:
        """Produce an aspect-level finding with signal, summary, and sources."""
        ...


class Judge(Protocol):
    """Synthesizes a final verdict from plan context and findings."""

    async def judge(
        self,
        claim: str,
        plan: InvestigationPlan,
        findings: list[AspectFinding],
    ) -> FactCheckReport:
        """Return the final fact-check report with verdict and justification."""
        ...


class Reviewer(Protocol):
    """Decides whether follow-up research is needed before judging."""

    async def review(
        self,
        claim: str,
        plan: InvestigationPlan,
        findings: list[AspectFinding],
    ) -> ReviewDecision:
        """Request targeted retries/new checks or finalize the current evidence."""
        ...


class ClaimExtractionBackend(Protocol):
    """Extracts check-worthy factual claims from free-form text."""

    async def extract(self, input_text: str, max_claims: int) -> ClaimExtractionResult:
        """Return decontextualized atomic claims suitable for fact checking."""
        ...
