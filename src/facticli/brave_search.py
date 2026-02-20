from __future__ import annotations

import json
import os
from typing import Any

import requests
from agents import FunctionTool, function_tool


def build_brave_web_search_tool() -> FunctionTool:
    @function_tool
    def brave_web_search(
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
        api_key = os.getenv("BRAVE_SEARCH_API_KEY")
        if not api_key:
            raise RuntimeError("BRAVE_SEARCH_API_KEY is not set.")

        safe_count = min(max(count, 1), 20)
        response = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": api_key,
            },
            params={
                "q": query,
                "count": safe_count,
                "country": country,
                "search_lang": search_lang,
                "extra_snippets": "true",
            },
            timeout=20,
        )
        response.raise_for_status()

        payload: dict[str, Any] = response.json()
        web_results = payload.get("web", {}).get("results", [])

        normalized_results: list[dict[str, Any]] = []
        for item in web_results:
            normalized_results.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "description": item.get("description", ""),
                    "age": item.get("age"),
                    "extra_snippets": item.get("extra_snippets", [])[:3],
                }
            )

        return json.dumps(
            {
                "provider": "brave",
                "query": query,
                "result_count": len(normalized_results),
                "results": normalized_results,
            },
            ensure_ascii=False,
        )

    return brave_web_search

