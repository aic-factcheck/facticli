from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator


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


class CheckworthyClaim(BaseModel):
    claim_id: str = Field(description="Stable identifier for the extracted claim.")
    claim_text: str = Field(
        description=(
            "Standalone, decontextualized, atomic factual claim suitable for independent checking."
        )
    )
    source_fragment: str = Field(
        description="Short fragment from the input text that directly grounds this claim."
    )
    checkworthy_reason: str = Field(
        description="Why this claim is check-worthy (impact, specificity, verifiability)."
    )


class ClaimExtractionResult(BaseModel):
    input_text: str = Field(description="Original input text that was processed.")
    claims: list[CheckworthyClaim] = Field(default_factory=list)
    coverage_notes: list[str] = Field(
        default_factory=list,
        description="How extraction covers the factual content from the input.",
    )
    excluded_nonfactual: list[str] = Field(
        default_factory=list,
        description="Statements intentionally excluded as non-factual/opinion/rhetoric.",
    )


class SourceEvidence(BaseModel):
    title: str = Field(description="Human-readable title of the source.")
    url: str = Field(description="URL used during fact-checking.")
    snippet: str = Field(description="Short text span from the source backing the claim.")
    publisher: str | None = Field(default=None)
    published_at: str | None = Field(default=None)

    @field_validator("url")
    @classmethod
    def url_must_be_http(cls, v: str) -> str:
        stripped = v.strip()
        if stripped and not stripped.startswith(("http://", "https://")):
            raise ValueError(f"url must start with http:// or https://, got: {stripped!r}")
        return stripped


class VerificationCheck(BaseModel):
    aspect_id: str = Field(description="Stable check identifier, e.g. 'timeline_1'.")
    question: str = Field(description="Precise verification question for one claim aspect.")
    rationale: str = Field(description="Why this question matters for claim validation.")
    search_queries: list[str] = Field(
        default_factory=list,
        description="Targeted web queries to be used by the investigator.",
    )


class InvestigationPlan(BaseModel):
    claim: str = Field(description="Exact claim text being investigated.")
    checks: list[VerificationCheck] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


class AspectFinding(BaseModel):
    aspect_id: str
    question: str
    signal: EvidenceSignal
    summary: str = Field(description="What the collected evidence says for this aspect.")
    confidence: float = Field(ge=0.0, le=1.0, description="0 to 1 confidence score for this aspect.")
    sources: list[SourceEvidence] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


class FactCheckReport(BaseModel):
    claim: str
    verdict: VeracityVerdict
    verdict_confidence: float = Field(ge=0.0, le=1.0, description="0 to 1 confidence in final verdict.")
    justification: str = Field(description="Tight synthesis of why the verdict is assigned.")
    key_points: list[str] = Field(default_factory=list)
    findings: list[AspectFinding] = Field(default_factory=list)
    sources: list[SourceEvidence] = Field(default_factory=list)
