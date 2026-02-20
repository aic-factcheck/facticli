from __future__ import annotations

import asyncio
from dataclasses import dataclass

from facticli.application.interfaces import Retriever
from facticli.brave_search import run_brave_web_search


@dataclass(frozen=True)
class BraveSearchRetriever(Retriever):
    async def search(self, queries: list[str], results_per_query: int) -> list[dict[str, object]]:
        if not queries:
            return []

        tasks = [
            asyncio.to_thread(run_brave_web_search, query, results_per_query) for query in queries
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        payloads: list[dict[str, object]] = []
        for query, result in zip(queries, results, strict=False):
            if isinstance(result, Exception):
                payloads.append(
                    {
                        "provider": "brave",
                        "query": query,
                        "result_count": 0,
                        "results": [],
                        "error": f"{type(result).__name__}: {result}",
                    }
                )
                continue
            payloads.append(result)
        return payloads
