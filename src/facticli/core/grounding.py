from __future__ import annotations

import re
import unicodedata
from collections import Counter

from .contracts import AspectFinding, SourceEvidence, SourceGrounding
from .retrieval import RetrievalJournal
from .truncation import clean_result_text

# Deterministic provenance check for cited sources: no model call, only string
# comparison against the retrieval journal. It answers "did this URL, and this
# wording, really come out of a search tool?" and nothing more. A source that
# passes can still be wrong, and a snippet that passes can still fail to
# support the finding; the field is called grounding, not verification, for
# that reason.

# Shorter quotes are too easy to match by accident to count as evidence that a
# passage was copied. Counted without whitespace; 15 still admits a short
# factual sentence ("GDP grew 2.1% in 2020") and excludes a bare number or name.
MIN_QUOTE_CHARS = 15

_ELLIPSIS = re.compile(r"\[\s*(?:\.{3,}|…)\s*\](?:\s*\(\d+ chars omitted\))?|\.{3,}|…")
_WHITESPACE = re.compile(r"\s+")
_PUNCTUATION_FOLD = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "‚": "'",
        "“": '"',
        "”": '"',
        "„": '"',
        "«": '"',
        "»": '"',
        "‐": "-",
        "‑": "-",
        "‒": "-",
        "–": "-",
        "—": "-",
        "−": "-",
        "­": "",
    }
)
_BRACKETED = re.compile(r"\[[^\[\]]*\]")
_SOURCE_ID = re.compile(r"\bS(\d+)\b")


def normalize_evidence_text(text: str) -> str:
    """Fold a passage to a comparison form with all whitespace removed.

    Models strip markup, straighten quotes, and re-space punctuation when they
    copy a passage. Removing markup, unifying typographic characters,
    case-folding, and dropping whitespace makes a faithful copy compare equal
    without making a paraphrase match.
    """
    folded = unicodedata.normalize("NFKC", clean_result_text(text)).translate(_PUNCTUATION_FOLD)
    return _WHITESPACE.sub("", folded).casefold()


def quote_segments(snippet: str) -> list[str]:
    """Split a snippet at ellipses into normalized segments long enough to check."""
    segments: list[str] = []
    for part in _ELLIPSIS.split(snippet or ""):
        # Strip wrapping quotation marks after folding, so curly ones go too.
        normalized = normalize_evidence_text(part).strip("\"'")
        if len(normalized) >= MIN_QUOTE_CHARS:
            segments.append(normalized)
    return segments


def snippet_in_text(snippet: str, retrieved_text: str) -> bool:
    """True when every checkable segment of the snippet occurs verbatim in the text."""
    segments = quote_segments(snippet)
    if not segments:
        return False
    haystack = normalize_evidence_text(retrieved_text)
    return all(segment in haystack for segment in segments)


def ground_source(source: SourceEvidence, journal: RetrievalJournal | None) -> SourceGrounding | None:
    """Classify one source against the journal; None when grounding cannot be assessed."""
    if journal is None or not journal.complete:
        return None
    texts = journal.sightings(source.url)
    if not texts:
        return SourceGrounding.UNMATCHED
    if snippet_in_text(source.snippet, "\n".join(texts)):
        return SourceGrounding.QUOTE_MATCHED
    return SourceGrounding.URL_MATCHED


def ground_sources(sources: list[SourceEvidence], journal: RetrievalJournal | None) -> list[SourceEvidence]:
    """Return copies of ``sources`` with their grounding level set by the harness.

    The field is always overwritten, to None when grounding cannot be
    assessed, so a value that arrived with the model's output never survives.
    """
    return [source.model_copy(update={"grounding": ground_source(source, journal)}) for source in sources]


def ground_finding(finding: AspectFinding, journal: RetrievalJournal | None) -> AspectFinding:
    """Set the grounding level of every source of a finding."""
    if not finding.sources:
        return finding
    return finding.model_copy(update={"sources": ground_sources(finding.sources, journal)})


def summarize_grounding(sources: list[SourceEvidence]) -> dict[str, int]:
    """Count sources per grounding level; empty when none was assessed."""
    return dict(Counter(source.grounding.value for source in sources if source.grounding is not None))


def unresolved_citation_ids(texts: list[str], source_count: int) -> list[int]:
    """Source numbers cited in square brackets that point at no row of the source list.

    Any ``S<n>`` inside brackets counts, so ``[S1, S9]`` and ``[S1][S9]`` are
    read alike. Resolution is bookkeeping, not verification: a resolved id
    shows that the cited source exists, not that it supports the sentence it
    is attached to.
    """
    cited = dict.fromkeys(
        int(number)
        for text in texts
        for group in _BRACKETED.findall(text or "")
        for number in _SOURCE_ID.findall(group)
    )
    return [number for number in cited if not 1 <= number <= source_count]
