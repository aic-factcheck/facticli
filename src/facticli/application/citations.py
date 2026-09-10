from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import httpx

from facticli.core.contracts import SourceEvidence, UrlStatus

# Deep-research agents hallucinate roughly one in ten URLs; a HEAD check with a
# Wayback fallback in the loop brings confirmed-broken citations under 1%
# ("Detecting and Correcting Reference Hallucinations", 2026). This pass is
# deterministic (no model call) and runs after the judge so that the report
# can mark each source as live, archived, or broken.

WAYBACK_AVAILABILITY_URL = "https://archive.org/wayback/available"
_USER_AGENT = "facticli/0.1 (+https://github.com/aic-factcheck/facticli) citation-check"

Fetcher = Callable[[str], Awaitable[int]]


@dataclass
class CitationHealthChecker:
    """Check cited URLs for reachability, falling back to Wayback snapshots."""

    timeout_seconds: float = 8.0
    max_parallel: int = 8
    use_wayback: bool = True
    client_factory: Callable[[], httpx.AsyncClient] | None = None

    def _client(self) -> httpx.AsyncClient:
        if self.client_factory is not None:
            return self.client_factory()
        return httpx.AsyncClient(
            timeout=self.timeout_seconds,
            follow_redirects=True,
            headers={"User-Agent": _USER_AGENT, "Accept": "*/*"},
        )

    async def _status(self, client: httpx.AsyncClient, url: str) -> int | None:
        """Return the HTTP status for ``url`` (HEAD, then GET), or None on error."""
        try:
            response = await client.head(url)
            if response.status_code < 400 or response.status_code in (401, 402, 403, 429):
                return response.status_code
            if response.status_code not in (404, 405, 410, 501):
                return response.status_code
        except httpx.HTTPError:
            pass
        try:
            async with client.stream("GET", url) as response:
                return response.status_code
        except httpx.HTTPError:
            return None

    async def _wayback(self, client: httpx.AsyncClient, url: str) -> str | None:
        try:
            response = await client.get(WAYBACK_AVAILABILITY_URL, params={"url": url})
            if response.status_code >= 400:
                return None
            payload: dict[str, Any] = response.json()
        except (httpx.HTTPError, ValueError):
            return None
        closest = (payload.get("archived_snapshots") or {}).get("closest") or {}
        if closest.get("available") and closest.get("url"):
            return str(closest["url"])
        return None

    async def check_url(self, client: httpx.AsyncClient, url: str) -> tuple[UrlStatus, str | None]:
        """Classify one URL; paywalls / bot walls (401/403/429) count as live."""
        status = await self._status(client, url)
        if status is not None and (status < 400 or status in (401, 402, 403, 429)):
            return UrlStatus.LIVE, None
        if self.use_wayback:
            archived = await self._wayback(client, url)
            if archived:
                return UrlStatus.ARCHIVED, archived
        return UrlStatus.BROKEN, None

    async def check_sources(self, sources: list[SourceEvidence]) -> list[SourceEvidence]:
        """Return copies of ``sources`` annotated with ``url_status`` / ``archived_url``.

        Identical URLs are checked once. Sources already annotated are kept.
        """
        pending = {source.url for source in sources if source.url_status is None and source.url}
        results: dict[str, tuple[UrlStatus, str | None]] = {}
        if pending:
            semaphore = asyncio.Semaphore(max(1, self.max_parallel))
            async with self._client() as client:

                async def run(url: str) -> None:
                    async with semaphore:
                        results[url] = await self.check_url(client, url)

                await asyncio.gather(*(run(url) for url in sorted(pending)))

        annotated: list[SourceEvidence] = []
        for source in sources:
            if source.url_status is not None or source.url not in results:
                annotated.append(source)
                continue
            status, archived_url = results[source.url]
            annotated.append(source.model_copy(update={"url_status": status, "archived_url": archived_url}))
        return annotated


def summarize_url_statuses(sources: list[SourceEvidence]) -> dict[str, int]:
    """Count sources per url_status value (``unchecked`` when absent)."""
    counts: dict[str, int] = {}
    for source in sources:
        key = source.url_status.value if source.url_status is not None else "unchecked"
        counts[key] = counts.get(key, 0) + 1
    return counts
