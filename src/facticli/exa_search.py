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
from facticli.core.search_cache import SearchCache
from facticli.core.truncation import UNTRUSTED_CONTENT_NOTICE, clean_result_text, truncate_search_results

DEFAULT_MAX_CHARS_PER_FIELD = 1500
DEFAULT_MAX_TOTAL_CHARS = 12000
DEFAULT_RESULT_COUNT = 5
EXA_SEARCH_URL = "https://api.exa.ai/search"
PROVIDER = "exa"
# "fast" answered in about 0.2 s against about 5 s for "auto" at the same price
# (one probe, 2026-11-19) and led the OpenBenchmarks company-news board.
SEARCH_TYPE = "fast"
HIGHLIGHT_SENTENCES = 3

_RATE_LIMIT_RETRIES = 2


def normalize_exa_results(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Reduce an Exa response to the fields the researcher needs."""
    normalized: list[dict[str, Any]] = []
    for item in payload.get("results", []) or []:
        highlights = [clean_result_text(str(text)) for text in (item.get("highlights") or []) if text]
        normalized.append(
            {
                "title": clean_result_text(str(item.get("title") or "")),
                "url": str(item.get("url") or ""),
                "published": item.get("publishedDate"),
                "author": item.get("author"),
                "snippet": "\n...\n".join(highlights),
            }
        )
    return normalized


def _tool_failure(context: RunContextWrapper[Any], error: Exception) -> str:
    """Let a rejected API key fail the run; anything else is reported to the model as usual."""
    if isinstance(error, SearchAuthError):
        raise error
    return default_tool_error_function(context, error)


async def _request_exa(body: dict[str, Any], api_key: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=30) as client:
        for attempt in range(_RATE_LIMIT_RETRIES + 1):
            response = await client.post(
                EXA_SEARCH_URL,
                headers={"x-api-key": api_key, "Content-Type": "application/json"},
                json=body,
            )
            if response.status_code != 429 or attempt == _RATE_LIMIT_RETRIES:
                break
            await asyncio.sleep(2**attempt + random.random())
    if response.status_code in (401, 403):
        raise SearchAuthError(f"Exa rejected the API key (HTTP {response.status_code}).")
    if response.status_code == 402:
        raise SearchAuthError("Exa reports exhausted credits (HTTP 402).")
    response.raise_for_status()
    return response.json()


def _result_text(item: dict[str, Any]) -> str:
    return "\n".join(part for part in (item["title"], item["snippet"]) if part)


def _published_after(item: dict[str, Any], claim_date: str) -> bool:
    published = str(item.get("published") or "")[:10]
    return bool(published) and published > claim_date


async def run_exa_web_search(
    query: str,
    count: int | None = None,
    *,
    max_chars_per_field: int = DEFAULT_MAX_CHARS_PER_FIELD,
    max_total_chars: int = DEFAULT_MAX_TOTAL_CHARS,
    default_count: int = DEFAULT_RESULT_COUNT,
    cache: SearchCache | None = None,
) -> dict[str, Any]:
    """Run one Exa search and return the bounded tool output.

    Leakage controls are applied by the engine before the model sees anything:
    fact-check domains go into ``excludeDomains`` and the claim date into
    ``endPublishedDate``. Exa lets results without a publication date through
    that filter (a 2020 fact-check came back for a 2011 cutoff in a probe), so
    under a cutoff undated results are dropped here and counted. Domain/path
    entries Exa cannot express are filtered here as well.
    """
    api_key = os.getenv("EXA_API_KEY")
    if not api_key:
        raise SearchAuthError("EXA_API_KEY is not set.")

    refusal = refuse_search_over_cap(PROVIDER, query)
    if refusal is not None:
        return refusal

    journal = get_journal()
    constraints = get_constraints()
    claim_date = constraints.claim_date if constraints else None
    blocked_domains = list(constraints.blocked_domains) if constraints else []

    safe_count = min(max(count if count is not None else default_count, 1), 20)
    body: dict[str, Any] = {
        "query": query,
        "numResults": safe_count,
        "type": SEARCH_TYPE,
        "contents": {"highlights": {"numSentences": HIGHLIGHT_SENTENCES}},
    }
    bare_domains = sorted({entry for entry in blocked_domains if "/" not in entry})
    if bare_domains:
        body["excludeDomains"] = bare_domains
    if claim_date:
        body["endPublishedDate"] = f"{claim_date}T23:59:59.999Z"

    request = {**body, "query": " ".join(query.split()).casefold()}
    raw_results = cache.get(request) if cache is not None else None
    cached = raw_results is not None
    if not cached:
        try:
            raw_results = normalize_exa_results(await _request_exa(body, api_key))
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
        if cache is not None:
            cache.put(request, raw_results)

    results = [item for item in raw_results if not is_blocked_url(item["url"], blocked_domains)]
    blocked = len(raw_results) - len(results)
    undated_dropped = 0
    if claim_date:
        kept = [item for item in results if item.get("published") and not _published_after(item, claim_date)]
        undated_dropped = sum(1 for item in results if not item.get("published"))
        results = kept

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
        result["published_before"] = claim_date
    if blocked:
        result["blocked_result_count"] = blocked
    if undated_dropped:
        result["undated_result_count_removed"] = undated_dropped

    if journal is not None:
        shown = results[: len(bounded_results)]
        journal.add(
            RetrievalRecord(provider=PROVIDER, query=query, urls=[item["url"] for item in shown], cached=cached),
            texts=[_result_text(item) for item in shown],
        )
    return result


def build_exa_web_search_tool(
    *,
    max_chars_per_field: int = DEFAULT_MAX_CHARS_PER_FIELD,
    max_total_chars: int = DEFAULT_MAX_TOTAL_CHARS,
    default_count: int = DEFAULT_RESULT_COUNT,
    cache: SearchCache | None = None,
) -> FunctionTool:
    @function_tool(failure_error_function=_tool_failure)
    async def exa_web_search(query: str, count: int | None = None) -> str:
        """
        Search the public web with the Exa search API and return concise JSON results with highlights.

        Write the query in the language the evidence is most likely published in; the engine has no
        language or country parameter.

        Args:
            query: Search query string.
            count: Number of web results to return (1-20). Omit to use the configured default.

        Returns:
            A JSON string with query metadata and normalized web results.
        """
        result = await run_exa_web_search(
            query=query,
            count=count,
            max_chars_per_field=max_chars_per_field,
            max_total_chars=max_total_chars,
            default_count=default_count,
            cache=cache,
        )
        return json.dumps(result, ensure_ascii=False)

    return exa_web_search
