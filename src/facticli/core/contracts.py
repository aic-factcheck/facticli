from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator
from pydantic.json_schema import SkipJsonSchema

# Fields annotated with ``SkipJsonSchema`` are assigned by the harness after
# the model call (tiering, grounding, failure status, stop reason).
# They stay out of the model-facing output schema so the model is never asked
# to guess them. A provider that does not enforce the schema strictly still
# accepts them when a model emits them, so model output goes through
# ``without_harness_fields()`` before the harness assigns anything.


def _harness_defaults(model: BaseModel) -> dict[str, Any]:
    """Default value of every ``SkipJsonSchema`` field, so a newly added one is reset too."""
    return {
        name: field.get_default(call_default_factory=True)
        for name, field in type(model).model_fields.items()
        if any(isinstance(item, SkipJsonSchema) for item in field.metadata)
    }


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


class FindingStatus(str, Enum):
    """Whether a check produced an observation or the harness failed to run it.

    A ``failed`` check is a missing observation, not evidence of absence;
    downstream stages must not read it as ``insufficient`` evidence about the
    world.
    """

    COMPLETED = "completed"
    FAILED = "failed"


class SourceTier(str, Enum):
    """Deterministic authority tier assigned from the source domain."""

    PRIMARY = "primary"  # official, governmental, institutional, scientific, legal
    REFERENCE = "reference"  # encyclopedias, archives, curated datasets
    NEWS = "news"  # established news organizations and wire services
    FACT_CHECKER = "fact_checker"  # dedicated fact-checking organizations
    USER_GENERATED = "user_generated"  # social media, forums, blogging platforms
    OTHER = "other"


class SourceGrounding(str, Enum):
    """Whether a cited source can be traced to what the search tools returned.

    Assigned by a deterministic comparison against the run's retrieval journal.
    It records provenance only: that the URL (and the quoted wording) really
    came out of a tool call. It says nothing about whether the source is right
    or whether the snippet supports the finding.
    """

    QUOTE_MATCHED = "quote_matched"  # snippet found verbatim in the text retrieved for that URL
    URL_MATCHED = "url_matched"  # URL was retrieved, snippet wording was not found there
    UNMATCHED = "unmatched"  # URL never appeared in any tool output of the run


class StopReason(str, Enum):
    """Why a completed research run ended before the model chose to stop.

    The finding is still an observation; it was produced from the evidence
    gathered up to the limit and should be read with that in mind.
    """

    TURN_LIMIT = "turn_limit"  # hit max turns; a tools-off wrap-up call produced the finding
    SEARCH_CAP = "search_cap"  # the hard search cap refused further calls


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
    """Structured output from claim extraction with coverage metadata."""
    input_text: str = Field(description="Original input text that was processed.")
    detected_language: str = Field(
        default="",
        description=(
            "ISO 639-1 code of the input's dominant language (e.g. 'cs', 'sk', 'pl', 'en'). "
            "All generated claim text is written in this language."
        ),
    )
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
    """Source snippet attached to findings and final verdict justification."""
    title: str = Field(description="Human-readable title of the source.")
    url: str = Field(description="URL used during fact-checking.")
    snippet: str = Field(description="Short text span from the source backing the claim.")
    publisher: str | None = Field(default=None)
    published_at: str | None = Field(default=None)
    tier: SkipJsonSchema[SourceTier | None] = None
    grounding: SkipJsonSchema[SourceGrounding | None] = None

    @field_validator("url")
    @classmethod
    def url_must_be_http(cls, value: str) -> str:
        stripped = value.strip()
        if stripped and not stripped.startswith(("http://", "https://")):
            raise ValueError(f"url must start with http:// or https://, got: {stripped!r}")
        return stripped

    def without_harness_fields(self) -> "SourceEvidence":
        """Copy with every harness-assigned field reset to its default."""
        return self.model_copy(update=_harness_defaults(self))


class VerificationCheck(BaseModel):
    """Single independent verification question generated by planning."""
    aspect_id: str = Field(description="Stable check identifier, e.g. 'timeline_1'.")
    question: str = Field(description="Precise verification question for one claim aspect.")
    rationale: str = Field(description="Why this question matters for claim validation.")
    search_queries: list[str] = Field(
        default_factory=list,
        description="Targeted web queries to be used by the investigator.",
    )
    acceptance_criteria: list[str] = Field(
        default_factory=list,
        description=(
            "Concrete 'done when ...' conditions that settle this check, e.g. "
            "'the figure is confirmed by an official statistics release'. Used by the reviewer."
        ),
    )


class InvestigationPlan(BaseModel):
    """Planner output that enumerates checks and explicit assumptions."""
    claim: str = Field(description="Exact claim text being investigated.")
    checks: list[VerificationCheck] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


class AspectFinding(BaseModel):
    """Research outcome for one planned aspect of the claim."""
    aspect_id: str
    question: str
    signal: EvidenceSignal
    summary: str = Field(description="What the collected evidence says for this aspect.")
    confidence: float = Field(ge=0.0, le=1.0, description="0 to 1 confidence score for this aspect.")
    sources: list[SourceEvidence] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    status: SkipJsonSchema[FindingStatus] = FindingStatus.COMPLETED
    failure_reason: SkipJsonSchema[str | None] = None
    stop_reason: SkipJsonSchema[StopReason | None] = None

    @property
    def is_observation(self) -> bool:
        """True when the finding reflects actual research, not a harness failure."""
        return self.status == FindingStatus.COMPLETED

    def without_harness_fields(self) -> "AspectFinding":
        """Copy with every harness-assigned field reset, on the finding and its sources."""
        return self.model_copy(
            update={
                **_harness_defaults(self),
                "sources": [source.without_harness_fields() for source in self.sources],
            }
        )


class FactCheckReport(BaseModel):
    """Final verdict artifact combining synthesis, findings, and deduped sources."""
    claim: str
    verdict: VeracityVerdict
    verdict_confidence: float = Field(ge=0.0, le=1.0, description="0 to 1 confidence in final verdict.")
    justification: str = Field(
        description=(
            "Tight synthesis of why the verdict is assigned. Every evidential sentence cites "
            "source ids like [S1]; inferences are marked as such."
        )
    )
    key_points: list[str] = Field(default_factory=list)
    counter_argument: str = Field(
        default="",
        description=(
            "The strongest honest case for a different verdict, grounded in the same findings, "
            "and why it was not adopted."
        ),
    )
    evidence_gaps: list[str] = Field(
        default_factory=list,
        description="Specific evidence that would have changed or firmed up the verdict but was not found.",
    )
    findings: list[AspectFinding] = Field(default_factory=list)
    sources: list[SourceEvidence] = Field(default_factory=list)

    def without_harness_fields(self) -> "FactCheckReport":
        """Copy with harness-assigned fields reset on every finding and source."""
        return self.model_copy(
            update={
                "findings": [finding.without_harness_fields() for finding in self.findings],
                "sources": [source.without_harness_fields() for source in self.sources],
            }
        )

