from __future__ import annotations

from collections.abc import Sequence
from contextvars import ContextVar, Token
from typing import Any
from urllib.parse import parse_qsl, unquote, urlsplit

from pydantic import BaseModel, Field

# The retrieval journal is the harness's own record of what the search tools
# returned during one research run. Two things hang off it:
#
#   1. Provenance. A model is told to cite only sources it saw. With a journal
#      that becomes checkable by string comparison, with no model call: was
#      this URL in a tool result, and does the quoted snippet occur in the
#      text returned for it (``core/grounding.py``).
#   2. A hard cap. The journal counts search calls, so the tool itself can
#      refuse the call that exceeds the cap and tell the model to answer.
#
# Retrieved text is held in memory for the comparison. Run artifacts keep only
# the receipts.

_TRACKING_PARAMS: frozenset[str] = frozenset(
    {"fbclid", "gclid", "gbraid", "wbraid", "msclkid", "mc_cid", "mc_eid", "igshid", "srsltid", "yclid", "dclid"}
)

SEARCH = "search"
OPEN_PAGE = "open_page"


class RetrievalRecord(BaseModel):
    """Receipt for one tool call made while researching.

    A receipt is execution evidence only: it records that a call happened and
    which URLs came back. It does not validate anything the model wrote later.
    """

    provider: str
    kind: str = SEARCH
    query: str | None = None
    urls: list[str] = Field(default_factory=list)
    cached: bool = False
    refused: bool = False
    error: str | None = None


class RetrievalSummary(BaseModel):
    """Counts over the receipts of one whole run."""

    searches: int = 0
    pages_opened: int = 0
    cache_hits: int = 0
    refused: int = 0
    errors: int = 0
    distinct_urls: int = 0


def summarize_retrieval(records: list[RetrievalRecord]) -> RetrievalSummary:
    summary = RetrievalSummary()
    urls: set[tuple[str, frozenset[tuple[str, str]]]] = set()
    for record in records:
        if record.refused:
            summary.refused += 1
            continue
        if record.kind == OPEN_PAGE:
            summary.pages_opened += 1
        else:
            summary.searches += 1
        summary.cache_hits += record.cached
        summary.errors += record.error is not None
        urls.update(url_key(url) for url in record.urls)
    summary.distinct_urls = sum(1 for location, _ in urls if location)
    return summary


def url_key(url: str) -> tuple[str, frozenset[tuple[str, str]]]:
    """Reduce a URL to ``(host + path, query pairs)`` for provenance matching.

    Scheme, ``www.``, fragments, trailing slashes, percent-encoding, and
    tracking parameters are normalized away because models and search tools
    disagree on all of them for the same page. Query parameters that can
    carry identity (``?id=``, ``?p=``) are kept.
    """
    stripped = (url or "").strip()
    if not stripped:
        return "", frozenset()
    if "://" not in stripped:
        stripped = "http://" + stripped
    try:
        parts = urlsplit(stripped)
    except ValueError:
        return "", frozenset()
    host = (parts.hostname or "").lower().rstrip(".").removeprefix("www.")
    if not host:
        return "", frozenset()
    query = frozenset(
        (key.lower(), value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if not key.lower().startswith("utm_") and key.lower() not in _TRACKING_PARAMS
    )
    return host + unquote(parts.path).rstrip("/"), query


class RetrievalJournal:
    """Per-research-run record of tool calls, returned URLs, and returned text."""

    def __init__(self, *, search_cap: int | None = None) -> None:
        self.records: list[RetrievalRecord] = []
        self.search_cap = search_cap
        # True when every URL the tools returned is in the journal, so a cited
        # URL that is missing from it was not retrieved. The adapter that runs
        # the tools sets it; a hosted search call that does not report its
        # sources clears it. While False, cited sources are left unassessed,
        # which is different from "nothing matched".
        self.complete = False
        self._searches = 0
        self._seen: dict[str, list[tuple[frozenset[tuple[str, str]], str]]] = {}

    def admit_search(self) -> bool:
        """Count one search call; False once the cap is used up."""
        if self.search_cap is not None and self._searches >= self.search_cap:
            return False
        self._searches += 1
        return True

    @property
    def refusals(self) -> int:
        return sum(1 for record in self.records if record.refused)

    def add(self, record: RetrievalRecord, texts: Sequence[str] = ()) -> None:
        """Append a receipt and index its URLs; ``texts[i]`` is the text returned for ``record.urls[i]``.

        Pass the text as retrieved, before any truncation for the model, so a
        snippet quoted across an elision still matches.
        """
        self.records.append(record)
        for index, url in enumerate(record.urls):
            location, query = url_key(url)
            if location:
                self._seen.setdefault(location, []).append((query, texts[index] if index < len(texts) else ""))

    def sightings(self, url: str) -> list[str]:
        """Text of every returned result this URL can refer to; empty when the tools never returned it.

        An entry is the empty string when only the URL was reported. A citation
        may drop query parameters of the returned URL but not add any: an
        extra ``?id=`` names a page the tools never returned.
        """
        location, query = url_key(url)
        return [text for known, text in self._seen.get(location, []) if query <= known]


_active_journal: ContextVar[RetrievalJournal | None] = ContextVar("facticli_retrieval_journal", default=None)


def activate_journal(journal: RetrievalJournal) -> Token:
    """Install a journal for the current async context (one research run)."""
    return _active_journal.set(journal)


def deactivate_journal(token: Token) -> None:
    """Restore the previous journal context."""
    _active_journal.reset(token)


def get_journal() -> RetrievalJournal | None:
    """Return the journal active in this context, if any."""
    return _active_journal.get()


def refuse_search_over_cap(provider: str, query: str) -> dict[str, Any] | None:
    """Count one search against the active journal's cap; None when the call may run.

    Past the cap the refusal is recorded and returned as an ordinary tool
    result that tells the model to answer from what it has, so reaching the
    cap never fails the run.
    """
    journal = get_journal()
    if journal is None or journal.admit_search():
        return None
    journal.add(RetrievalRecord(provider=provider, query=query, refused=True))
    return {
        "provider": provider,
        "query": query,
        "error": "search_limit_reached",
        "instruction": (
            f"All {journal.search_cap} searches allowed here have been used, so this search was not run. "
            "Do not call the search tool again. Return your final structured output now, based only on the "
            "evidence already retrieved. Where that evidence does not settle a question, say so and record "
            "what is still missing."
        ),
        "result_count": 0,
        "results": [],
    }
