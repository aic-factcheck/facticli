from __future__ import annotations

import asyncio
import json
import os
import random
from typing import Any

import httpx
from agents import FunctionTool, RunContextWrapper, default_tool_error_function, function_tool

from facticli.core.constraints import get_constraints, is_blocked_url
from facticli.core.errors import SearchAuthError
from facticli.core.retrieval import RetrievalRecord, get_journal, refuse_search_over_cap
from facticli.core.truncation import UNTRUSTED_CONTENT_NOTICE, clean_result_text, truncate_search_results

DEFAULT_MAX_CHARS_PER_FIELD = 1500
DEFAULT_MAX_TOTAL_CHARS = 12000
DEFAULT_RESULT_COUNT = 5
BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
PROVIDER = "brave"

# A throttled request (HTTP 429) succeeds a moment later; parallel researchers
# would otherwise each see a tool error and spend model turns reacting to it.
_RATE_LIMIT_RETRIES = 2


def normalize_brave_results(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Reduce a Brave response to the fields the researcher needs, with markup removed."""
    normalized: list[dict[str, Any]] = []
    for item in payload.get("web", {}).get("results", []) or []:
        normalized.append(
            {
                "title": clean_result_text(str(item.get("title") or "")),
                "url": str(item.get("url") or ""),
                "description": clean_result_text(str(item.get("description") or "")),
                "age": item.get("age"),
                "page_age": item.get("page_age"),
                "extra_snippets": [clean_result_text(str(snippet)) for snippet in (item.get("extra_snippets") or [])[:3]],
            }
        )
    return normalized


def _rejects_api_key(response: httpx.Response) -> bool:
    """Brave answers an invalid key with 422 and this code, not with 401."""
    return response.status_code == 401 or (
        response.status_code == 422 and "SUBSCRIPTION_TOKEN_INVALID" in response.text
    )


def _tool_failure(context: RunContextWrapper[Any], error: Exception) -> str:
    """Let a rejected API key fail the run; anything else is reported to the model as usual."""
    if isinstance(error, SearchAuthError):
        raise error
    return default_tool_error_function(context, error)


async def _request_brave(params: dict[str, Any], api_key: str) -> list[dict[str, Any]]:
    async with httpx.AsyncClient(timeout=20) as client:
        for attempt in range(_RATE_LIMIT_RETRIES + 1):
            response = await client.get(
                BRAVE_SEARCH_URL,
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                    "X-Subscription-Token": api_key,
                },
                params=params,
            )
            if response.status_code != 429 or attempt == _RATE_LIMIT_RETRIES:
                break
            await asyncio.sleep(2**attempt + random.random())
    if _rejects_api_key(response):
        raise SearchAuthError(f"Brave Search rejected the API key (HTTP {response.status_code}).")
    response.raise_for_status()
    return normalize_brave_results(response.json())


def _result_text(item: dict[str, Any]) -> str:
    return "\n".join(part for part in (item["title"], item["description"], *item["extra_snippets"]) if part)


async def run_brave_web_search(
    query: str,
    count: int | None = None,
    country: str = "us",
    search_lang: str = "en",
    *,
    max_chars_per_field: int = DEFAULT_MAX_CHARS_PER_FIELD,
    max_total_chars: int = DEFAULT_MAX_TOTAL_CHARS,
    default_count: int = DEFAULT_RESULT_COUNT,
) -> dict[str, Any]:
    """Run one search and return the bounded tool output.

    A failure the model can work around (throttling that outlasts the retries,
    a server error, a rejected parameter) comes back as a tool result that says
    so. A rejected or missing API key raises ``SearchAuthError``: no query can
    succeed, so the run must fail visibly instead of reporting every check as
    insufficient.
    """
    api_key = os.getenv("BRAVE_SEARCH_API_KEY")
    if not api_key:
        raise SearchAuthError("BRAVE_SEARCH_API_KEY is not set.")

    refusal = refuse_search_over_cap(PROVIDER, query)
    if refusal is not None:
        return refusal

    journal = get_journal()
    constraints = get_constraints()
    claim_date = constraints.claim_date if constraints else None
    blocked_domains = list(constraints.blocked_domains) if constraints else []

    safe_count = min(max(count if count is not None else default_count, 1), 20)
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

    try:
        raw_results = await _request_brave(params, api_key)
    except (httpx.HTTPError, ValueError) as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        error = f"search request failed ({f'HTTP {status}' if status else type(exc).__name__})"
        if journal is not None:
            journal.add(RetrievalRecord(provider=PROVIDER, query=query, error=error))
        return {
            "provider": PROVIDER,
            "query": query,
            "error": error,
            "instruction": (
                "This search failed. Try once more with a different, simpler query; if searches keep "
                "failing, finish with the evidence you already have and note the failure in the caveats."
            ),
            "result_count": 0,
            "results": [],
        }

    results = [item for item in raw_results if not is_blocked_url(item["url"], blocked_domains)]
    bounded_results, truncation_header = truncate_search_results(
        results,
        max_results=safe_count,
        max_chars_per_field=max_chars_per_field,
        max_total_chars=max_total_chars,
    )

    result: dict[str, Any] = {
        "provider": PROVIDER,
        "query": query,
        "content_policy": UNTRUSTED_CONTENT_NOTICE,
        "result_count": len(bounded_results),
        "results": bounded_results,
    }
    result.update(truncation_header)
    if claim_date:
        result["freshness_cutoff"] = claim_date
    if len(results) < len(raw_results):
        result["blocked_result_count"] = len(raw_results) - len(results)

    if journal is not None:
        # Index the results the model was shown, with their text before truncation.
        shown = results[: len(bounded_results)]
        journal.add(
            RetrievalRecord(provider=PROVIDER, query=query, urls=[item["url"] for item in shown]),
            texts=[_result_text(item) for item in shown],
        )
    return result


def build_brave_web_search_tool(
    *,
    max_chars_per_field: int = DEFAULT_MAX_CHARS_PER_FIELD,
    max_total_chars: int = DEFAULT_MAX_TOTAL_CHARS,
    default_count: int = DEFAULT_RESULT_COUNT,
) -> FunctionTool:
    @function_tool(failure_error_function=_tool_failure)
    async def brave_web_search(
        query: str,
        count: int | None = None,
        country: str = "us",
        search_lang: str = "en",
    ) -> str:
        """
        Search the public web using Brave Search API and return concise JSON results.

        Args:
            query: Search query string.
            count: Number of web results to return (1-20). Omit to use the configured default.
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
            max_chars_per_field=max_chars_per_field,
            max_total_chars=max_total_chars,
            default_count=default_count,
        )
        return json.dumps(result, ensure_ascii=False)

    return brave_web_search
