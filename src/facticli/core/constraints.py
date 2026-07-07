from __future__ import annotations

from contextvars import ContextVar, Token
from datetime import date
from urllib.parse import urlparse

from pydantic import BaseModel, Field

# Domains of dedicated fact-checking organizations. Retrieving these while
# evaluating on fact-checking benchmarks constitutes label leakage: the gold
# verdicts of datasets such as AVeriTeC are derived from these very articles
# (cf. AVerImaTeC shared task exclusion policy, arXiv:2602.11221).
FACT_CHECK_DOMAINS: tuple[str, ...] = (
    "politifact.com",
    "snopes.com",
    "factcheck.org",
    "fullfact.org",
    "leadstories.com",
    "factcheck.afp.com",
    "checkyourfact.com",
    "truthorfiction.com",
    "factly.in",
    "boomlive.in",
    "altnews.in",
    "verafiles.org",
    "africacheck.org",
    "poynter.org",
    "misbar.com",
    "logicallyfacts.com",
    "logically.ai",
    "factcheckni.org",
    "demagog.cz",
    "demagog.org.pl",
    "faktabaari.fi",
    "correctiv.org",
    "maldita.es",
    "newtral.es",
    "pagellapolitica.it",
    "facta.news",
    "teyit.org",
    "rappler.com/newsbreak/fact-check",
    "factcrescendo.com",
    "vishvasnews.com",
    "newschecker.in",
    "thequint.com/news/webqoof",
    "healthfeedback.org",
    "sciencefeedback.co",
    "climatefeedback.org",
    "washingtonpost.com/news/fact-checker",
    "apnews.com/hub/ap-fact-check",
    "reuters.com/fact-check",
    "factcheck.kz",
    "istinomer.rs",
    "faktograf.hr",
)


class ResearchConstraints(BaseModel):
    """Per-run retrieval constraints propagated to search tools and filters."""
    claim_id: str | None = None
    claim_date: str | None = None  # ISO YYYY-MM-DD once normalized
    blocked_domains: list[str] = Field(default_factory=list)
    knowledge_store_dir: str | None = None


_active_constraints: ContextVar[ResearchConstraints | None] = ContextVar(
    "facticli_research_constraints", default=None
)


def activate_constraints(constraints: ResearchConstraints) -> Token:
    """Install constraints for the current async context."""
    return _active_constraints.set(constraints)


def deactivate_constraints(token: Token) -> None:
    """Restore the previous constraints context."""
    _active_constraints.reset(token)


def get_constraints() -> ResearchConstraints | None:
    """Return the constraints active in this context, if any."""
    return _active_constraints.get()


def parse_date_loose(value: str | None) -> date | None:
    """Parse ISO (YYYY-MM-DD) or AVeriTeC-style (D-M-YYYY) date strings."""
    if not value:
        return None
    candidate = value.strip()[:10].replace("/", "-")
    parts = candidate.split("-")
    if len(parts) != 3:
        return None
    try:
        if len(parts[0]) == 4:
            year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
        else:
            day, month, year = int(parts[0]), int(parts[1]), int(parts[2])
        return date(year, month, day)
    except ValueError:
        return None


def normalize_claim_date(value: str | None) -> str | None:
    """Normalize any supported date string to ISO format, or None."""
    parsed = parse_date_loose(value)
    return parsed.isoformat() if parsed else None


def url_domain(url: str) -> str:
    """Extract a lowercase host (without www.) from a URL."""
    try:
        host = urlparse(url).netloc.lower()
    except ValueError:
        return ""
    return host.removeprefix("www.")


def is_blocked_url(url: str, blocked_domains: list[str] | tuple[str, ...]) -> bool:
    """True when the URL's host or host/path prefix matches a blocked entry.

    Entries may be bare domains (matched against the host, including
    subdomains) or domain/path prefixes (matched against host + path).
    """
    host = url_domain(url)
    if not host:
        return False
    try:
        path = urlparse(url).path.lower()
    except ValueError:
        path = ""
    host_and_path = host + path
    for entry in blocked_domains:
        entry = entry.lower().strip().removeprefix("www.")
        if not entry:
            continue
        if "/" in entry:
            if host_and_path.startswith(entry):
                return True
            continue
        if host == entry or host.endswith("." + entry):
            return True
    return False


def violates_date_cutoff(published_at: str | None, claim_date_iso: str | None) -> bool:
    """True when a source has a parseable publication date after the cutoff."""
    if not claim_date_iso:
        return False
    published = parse_date_loose(published_at)
    cutoff = parse_date_loose(claim_date_iso)
    if published is None or cutoff is None:
        return False
    return published > cutoff
