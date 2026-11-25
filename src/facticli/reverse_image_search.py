"""Reverse image search: earlier web appearances of a claim image.

Google Lens (through Serper) finds pages that show the same or a visually
similar image. It returns titles and links but no dates and no page text, so
every candidate page is then read through Exa's contents endpoint, which
supplies the publication date and a few sentences of text.

Leakage controls are the same as for the text search tools: blocked domains
are removed before anything else, and under a claim-date cutoff only pages
with a publication date on or before the cutoff are kept. A page Exa cannot
date is dropped under a cutoff, as undated Exa search results are, because
Lens matches of a viral image are mostly the later fact-checks of it.

Lens needs an image URL it can fetch, so the tool only works for images that
have a public URL (see ``facticli.core.images``).
"""

from __future__ import annotations

import asyncio
import json
import os
import random
from typing import Any

import httpx
from agents import FunctionTool, function_tool

from facticli.core.constraints import get_constraints, is_blocked_url
from facticli.core.errors import SearchAuthError
from facticli.core.retrieval import REVERSE_IMAGE, RetrievalRecord, get_journal, refuse_search_over_cap
from facticli.core.search_cache import SearchCache
from facticli.core.truncation import UNTRUSTED_CONTENT_NOTICE, clean_result_text, truncate_search_results
from facticli.exa_search import EXA_CONTENTS_URL, _request_exa, _tool_failure
from facticli.core.images import get_image

PROVIDER = "reverse_image_search"
SERPER_LENS_URL = "https://google.serper.dev/lens"
DEFAULT_RESULT_COUNT = 8
# Pages read per returned result: under a cutoff many matches are dropped
# after dating, so more candidates are read than are returned.
CANDIDATE_FACTOR = 2
HIGHLIGHT_SENTENCES = 3

_RATE_LIMIT_RETRIES = 2


def normalize_lens_matches(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Reduce a Serper Lens response to title, link, and site name per match."""
    matches: list[dict[str, Any]] = []
    for item in payload.get("organic", []) or []:
        url = str(item.get("link") or "")
        if url.startswith(("http://", "https://")):
            matches.append(
                {
                    "title": clean_result_text(str(item.get("title") or "")),
                    "url": url,
                    "site": str(item.get("source") or "") or None,
                }
            )
    return matches


def normalize_exa_pages(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Exa contents results keyed by the requested URL (``id``): date, author, highlights."""
    pages: dict[str, dict[str, Any]] = {}
    for item in payload.get("results", []) or []:
        key = str(item.get("id") or item.get("url") or "")
        highlights = [clean_result_text(str(text)) for text in (item.get("highlights") or []) if text]
        pages[key] = {
            "published": item.get("publishedDate"),
            "author": item.get("author"),
            "snippet": "\n...\n".join(highlights),
        }
    return pages


async def _request_lens(image_url: str, api_key: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=60) as client:
        for attempt in range(_RATE_LIMIT_RETRIES + 1):
            response = await client.post(
                SERPER_LENS_URL,
                headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
                json={"url": image_url},
            )
            if response.status_code != 429 or attempt == _RATE_LIMIT_RETRIES:
                break
            await asyncio.sleep(2**attempt + random.random())
    if response.status_code in (401, 403):
        raise SearchAuthError(f"Serper rejected the API key (HTTP {response.status_code}).")
    if response.status_code == 402:
        raise SearchAuthError("Serper reports exhausted credits (HTTP 402).")
    response.raise_for_status()
    return response.json()


def _notice(image_id: str, error: str, instruction: str) -> dict[str, Any]:
    return {"provider": PROVIDER, "image_id": image_id, "error": error, "instruction": instruction, "result_count": 0, "results": []}


def _published_on_or_before(published: str | None, claim_date: str) -> bool:
    day = str(published or "")[:10]
    return bool(day) and day <= claim_date


async def run_reverse_image_search(
    image_id: str,
    focus: str | None = None,
    count: int | None = None,
    *,
    max_chars_per_field: int = 1500,
    max_total_chars: int = 12000,
    default_count: int = DEFAULT_RESULT_COUNT,
    cache: SearchCache | None = None,
    lens_memo: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Find pages showing the claim image, dated and filtered, as bounded tool output.

    ``lens_memo`` keeps Lens matches per image URL for the life of the tool:
    researchers call the tool again on the same image with a new ``focus``,
    and Lens would return the same matches. Only the page reads differ.
    """
    image = get_image(image_id)
    if image is None:
        return _notice(image_id, "unknown_image_id", "Use one of the image ids listed in the task input.")
    if not image.public_url:
        return _notice(
            image.image_id,
            "no_public_url",
            "This image has no public URL, so it cannot be searched by image. Search the web for what it shows instead.",
        )
    serper_key = os.getenv("SERPER_API_KEY")
    if not serper_key:
        raise SearchAuthError("SERPER_API_KEY is not set.")
    exa_key = os.getenv("EXA_API_KEY")
    if not exa_key:
        raise SearchAuthError("EXA_API_KEY is not set; reverse image search uses it to date the pages it finds.")

    refusal = refuse_search_over_cap(PROVIDER, image.image_id)
    if refusal is not None:
        return refusal

    journal = get_journal()
    constraints = get_constraints()
    claim_date = constraints.claim_date if constraints else None
    blocked_domains = list(constraints.blocked_domains) if constraints else []
    safe_count = min(max(count if count is not None else default_count, 1), 20)
    focus = " ".join((focus or "").split()) or None

    lens_request = {"engine": "serper_lens", "url": image.public_url}
    matches = lens_memo.get(image.public_url) if lens_memo is not None else None
    if matches is None and cache is not None:
        matches = cache.get(lens_request)
    cached = matches is not None
    if not cached:
        try:
            matches = normalize_lens_matches(await _request_lens(image.public_url, serper_key))
        except (httpx.HTTPError, ValueError) as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            error = f"reverse image search failed ({f'HTTP {status}' if status else type(exc).__name__})"
            if journal is not None:
                journal.add(RetrievalRecord(provider=PROVIDER, kind=REVERSE_IMAGE, query=image.image_id, error=error))
            return _notice(
                image.image_id,
                error,
                "Reverse image search failed. Do not retry it; search the web for what the image shows instead.",
            )
        if cache is not None:
            cache.put(lens_request, matches)
    if lens_memo is not None:
        lens_memo[image.public_url] = matches

    allowed = [match for match in matches if not is_blocked_url(match["url"], blocked_domains)]
    blocked = len(matches) - len(allowed)
    candidates = allowed[: safe_count * CANDIDATE_FACTOR]

    pages: dict[str, dict[str, Any]] = {}
    page_error: str | None = None
    if candidates:
        highlights: dict[str, Any] = {"numSentences": HIGHLIGHT_SENTENCES}
        if focus:
            highlights["query"] = focus
        body = {"urls": [match["url"] for match in candidates], "highlights": highlights}
        contents_request = {"engine": "exa_contents", **body}
        recorded = cache.get(contents_request) if cache is not None else None
        if recorded is not None:
            pages = recorded
        else:
            try:
                pages = normalize_exa_pages(await _request_exa(body, exa_key, url=EXA_CONTENTS_URL))
            except (httpx.HTTPError, ValueError) as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                page_error = f"page dating failed ({f'HTTP {status}' if status else type(exc).__name__})"
            else:
                if cache is not None:
                    cache.put(contents_request, pages)

    results: list[dict[str, Any]] = []
    for match in candidates:
        page = pages.get(match["url"], {})
        results.append(
            {
                "title": match["title"],
                "url": match["url"],
                "site": match["site"],
                "published": page.get("published"),
                "author": page.get("author"),
                "snippet": page.get("snippet", ""),
            }
        )
    undated_dropped = 0
    after_cutoff_dropped = 0
    if claim_date:
        undated_dropped = sum(1 for item in results if not item["published"])
        after_cutoff_dropped = sum(
            1 for item in results if item["published"] and not _published_on_or_before(item["published"], claim_date)
        )
        results = [item for item in results if _published_on_or_before(item["published"], claim_date)]

    bounded_results, truncation_header = truncate_search_results(
        results[:safe_count],
        max_results=safe_count,
        max_chars_per_field=max_chars_per_field,
        max_total_chars=max_total_chars,
    )
    result: dict[str, Any] = {
        "provider": PROVIDER,
        "image_id": image.image_id,
        "content_policy": UNTRUSTED_CONTENT_NOTICE,
        "match_note": (
            "Pages showing the same or a visually similar image. A match does not prove the page shows this exact "
            "image; compare titles and text with the claim."
        ),
        "matches_found": len(matches),
        "result_count": len(bounded_results),
        "results": bounded_results,
    }
    result.update(truncation_header)
    if claim_date:
        result["published_before"] = claim_date
    if blocked:
        result["blocked_result_count"] = blocked
    if undated_dropped:
        result["undated_result_count_removed"] = undated_dropped
    if after_cutoff_dropped:
        result["after_cutoff_result_count_removed"] = after_cutoff_dropped
    if page_error:
        result["page_error"] = page_error

    if journal is not None:
        journal.add(
            RetrievalRecord(
                provider=PROVIDER,
                kind=REVERSE_IMAGE,
                query=image.image_id + (f" | {focus}" if focus else ""),
                urls=[item["url"] for item in bounded_results],
                cached=cached,
                error=page_error,
            ),
            texts=["\n".join(part for part in (item["title"], item["snippet"]) if part) for item in results[: len(bounded_results)]],
        )
    return result


def build_reverse_image_search_tool(
    *,
    max_chars_per_field: int = 1500,
    max_total_chars: int = 12000,
    default_count: int = DEFAULT_RESULT_COUNT,
    cache: SearchCache | None = None,
) -> FunctionTool:
    lens_memo: dict[str, list[dict[str, Any]]] = {}

    @function_tool(failure_error_function=_tool_failure)
    async def reverse_image_search(image_id: str, focus: str | None = None, count: int | None = None) -> str:
        """
        Find web pages that show one of the claim's images, with their publication dates.

        Use it to learn where and when an image appeared before the claim and what it was said to show
        there. Each call counts against your search budget.

        Args:
            image_id: Id of a claim image from the task input, e.g. "image_1".
            focus: Optional words describing what to look for in the matching pages; the returned
                snippets are chosen to match them.
            count: Number of pages to return (1-20). Omit to use the default.

        Returns:
            A JSON string with matching pages: title, URL, site, publication date, and a snippet.
        """
        result = await run_reverse_image_search(
            image_id,
            focus,
            count,
            max_chars_per_field=max_chars_per_field,
            max_total_chars=max_total_chars,
            default_count=default_count,
            cache=cache,
            lens_memo=lens_memo,
        )
        return json.dumps(result, ensure_ascii=False)

    return reverse_image_search
