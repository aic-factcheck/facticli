from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class VeracityVerdict(str, Enum):
    SUPPORTED = "Supported"
    REFUTED = "Refuted"
    NOT_ENOUGH_EVIDENCE = "Not Enough Evidence"
    CONFLICTING = "Conflicting Evidence/Cherrypicking"


class EvidenceSignal(str, Enum):
    SUPPORTS = "supports"
    REFUTES = "refutes"
    MIXED = "mixed"
    INSUFFICIENT = "insufficient"


class SourceEvidence(BaseModel):
    title: str = Field(description="Human-readable title of the source.")
    url: str = Field(description="URL used during fact-checking.")
    snippet: str = Field(description="Short text span from the source backing the claim.")
    publisher: str | None = Field(default=None)
    published_at: str | None = Field(default=None)


class VerificationCheck(BaseModel):
    aspect_id: str = Field(description="Stable check identifier, e.g. 'timeline_1'.")
    question: str = Field(description="Precise verification question for one claim aspect.")
    rationale: str = Field(description="Why this question matters for claim validation.")
    search_queries: list[str] = Field(
        default_factory=list,
        description="Targeted web queries to be used by the investigator.",
    )


class InvestigationPlan(BaseModel):
    claim: str
    checks: list[VerificationCheck] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


class AspectFinding(BaseModel):
    aspect_id: str
    question: str
    signal: EvidenceSignal
    summary: str = Field(description="What the collected evidence says for this aspect.")
    confidence: float = Field(description="0 to 1 confidence score for this aspect.")
    sources: list[SourceEvidence] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


class FactCheckReport(BaseModel):
    claim: str
    verdict: VeracityVerdict
    verdict_confidence: float = Field(description="0 to 1 confidence in final verdict.")
    justification: str = Field(description="Tight synthesis of why the verdict is assigned.")
    key_points: list[str] = Field(default_factory=list)
    findings: list[AspectFinding] = Field(default_factory=list)
    sources: list[SourceEvidence] = Field(default_factory=list)

