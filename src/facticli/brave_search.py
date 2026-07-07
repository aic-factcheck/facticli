from __future__ import annotations

import json
import os
from typing import Any

import httpx
from agents import FunctionTool, function_tool

from facticli.core.constraints import get_constraints, is_blocked_url


async def run_brave_web_search(
    query: str,
    count: int = 5,
    country: str = "us",
    search_lang: str = "en",
) -> dict[str, Any]:
    api_key = os.getenv("BRAVE_SEARCH_API_KEY")
    if not api_key:
        raise RuntimeError("BRAVE_SEARCH_API_KEY is not set.")

    constraints = get_constraints()
    claim_date = constraints.claim_date if constraints else None
    blocked_domains = list(constraints.blocked_domains) if constraints else []

    safe_count = min(max(count, 1), 20)
    params: dict[str, Any] = {
        "q": query,
        "count": safe_count,
        "country": country,
        "search_lang": search_lang,
        "extra_snippets": "true",
    }
    if claim_date:
        # Restrict result freshness to documents published up to the claim date.
        params["freshness"] = f"1900-01-01to{claim_date}"

    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": api_key,
            },
            params=params,
        )
    response.raise_for_status()

    payload: dict[str, Any] = response.json()
    web_results = payload.get("web", {}).get("results", [])

    normalized_results: list[dict[str, Any]] = []
    blocked_count = 0
    for item in web_results:
        url = item.get("url", "")
        if blocked_domains and is_blocked_url(url, blocked_domains):
            blocked_count += 1
            continue
        normalized_results.append(
            {
                "title": item.get("title", ""),
                "url": url,
                "description": item.get("description", ""),
                "age": item.get("age"),
                "page_age": item.get("page_age"),
                "extra_snippets": item.get("extra_snippets", [])[:3],
            }
        )

    result: dict[str, Any] = {
        "provider": "brave",
        "query": query,
        "result_count": len(normalized_results),
        "results": normalized_results,
    }
    if claim_date:
        result["freshness_cutoff"] = claim_date
    if blocked_count:
        result["blocked_result_count"] = blocked_count
    return result


def build_brave_web_search_tool() -> FunctionTool:
    @function_tool
    async def brave_web_search(
        query: str,
        count: int = 5,
        country: str = "us",
        search_lang: str = "en",
    ) -> str:
        """
        Search the public web using Brave Search API and return concise JSON results.

        Args:
            query: Search query string.
            count: Number of web results to return (1-20).
            country: Two-letter country code for search localization.
            search_lang: Language code for search filtering.

        Returns:
            A JSON string with query metadata and normalized web results.
        """
        result = await run_brave_web_search(
            query=query,
            count=count,
            country=country,
            search_lang=search_lang,
        )
        return json.dumps(result, ensure_ascii=False)

    return brave_web_search
