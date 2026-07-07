from __future__ import annotations

import json
import math
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any

from agents import FunctionTool, function_tool

from facticli.core.constraints import get_constraints

_CHUNK_TARGET_CHARS = 2000
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())


class _KnowledgeStoreIndex:
    """BM25-scored chunk index over one claim's pre-scraped knowledge store."""

    def __init__(self, chunks: list[dict[str, str]]):
        self.chunks = chunks
        self._chunk_tokens = [_tokenize(chunk["text"]) for chunk in chunks]
        self._doc_freq: dict[str, int] = {}
        for tokens in self._chunk_tokens:
            for term in set(tokens):
                self._doc_freq[term] = self._doc_freq.get(term, 0) + 1
        self._avg_len = (
            sum(len(tokens) for tokens in self._chunk_tokens) / len(self._chunk_tokens)
            if self._chunk_tokens
            else 0.0
        )

    def search(self, query: str, count: int) -> list[dict[str, Any]]:
        query_terms = _tokenize(query)
        if not query_terms or not self.chunks:
            return []
        n_docs = len(self.chunks)
        k1, b = 1.5, 0.75
        scores: list[tuple[float, int]] = []
        for index, tokens in enumerate(self._chunk_tokens):
            if not tokens:
                continue
            term_freq: dict[str, int] = {}
            for term in tokens:
                term_freq[term] = term_freq.get(term, 0) + 1
            score = 0.0
            for term in query_terms:
                tf = term_freq.get(term, 0)
                if tf == 0:
                    continue
                df = self._doc_freq.get(term, 0)
                idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
                denom = tf + k1 * (1 - b + b * len(tokens) / (self._avg_len or 1.0))
                score += idf * tf * (k1 + 1) / denom
            if score > 0:
                scores.append((score, index))
        scores.sort(key=lambda item: (-item[0], item[1]))
        results = []
        for score, index in scores[:count]:
            chunk = self.chunks[index]
            results.append(
                {
                    "title": chunk["url"],
                    "url": chunk["url"],
                    "description": chunk["text"],
                    "score": round(score, 3),
                }
            )
        return results


_index_cache: OrderedDict[str, _KnowledgeStoreIndex] = OrderedDict()
_INDEX_CACHE_SIZE = 8


def _iter_store_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        payload = json.loads(text)
        return payload if isinstance(payload, list) else []
    records = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def _build_chunks(records: list[dict[str, Any]]) -> list[dict[str, str]]:
    chunks: list[dict[str, str]] = []
    for record in records:
        url = str(record.get("url", "")).strip()
        body = record.get("url2text", record.get("text", ""))
        if isinstance(body, list):
            sentences = [str(item) for item in body]
        else:
            sentences = [str(body)]
        buffer = ""
        for sentence in sentences:
            candidate = f"{buffer} {sentence}".strip() if buffer else sentence.strip()
            if len(candidate) >= _CHUNK_TARGET_CHARS:
                chunks.append({"url": url, "text": candidate})
                buffer = ""
            else:
                buffer = candidate
        if buffer.strip():
            chunks.append({"url": url, "text": buffer.strip()})
    return chunks


def _load_index(store_dir: str, claim_id: str) -> _KnowledgeStoreIndex | None:
    cache_key = f"{store_dir}::{claim_id}"
    if cache_key in _index_cache:
        _index_cache.move_to_end(cache_key)
        return _index_cache[cache_key]

    base = Path(store_dir)
    path = None
    for suffix in (".json", ".jsonl"):
        candidate = base / f"{claim_id}{suffix}"
        if candidate.is_file():
            path = candidate
            break
    if path is None:
        return None

    index = _KnowledgeStoreIndex(_build_chunks(_iter_store_records(path)))
    _index_cache[cache_key] = index
    while len(_index_cache) > _INDEX_CACHE_SIZE:
        _index_cache.popitem(last=False)
    return index


async def run_knowledge_store_search(query: str, count: int = 5) -> dict[str, Any]:
    constraints = get_constraints()
    if constraints is None or not constraints.knowledge_store_dir or constraints.claim_id is None:
        return {
            "provider": "knowledge_store",
            "query": query,
            "error": (
                "No knowledge store is configured for this claim. "
                "Report the check as insufficient instead of guessing."
            ),
            "result_count": 0,
            "results": [],
        }

    index = _load_index(constraints.knowledge_store_dir, str(constraints.claim_id))
    if index is None:
        return {
            "provider": "knowledge_store",
            "query": query,
            "error": f"Knowledge store file for claim {constraints.claim_id} not found.",
            "result_count": 0,
            "results": [],
        }

    safe_count = min(max(count, 1), 20)
    results = index.search(query, safe_count)
    return {
        "provider": "knowledge_store",
        "query": query,
        "result_count": len(results),
        "results": results,
    }


def build_knowledge_store_search_tool() -> FunctionTool:
    @function_tool
    async def knowledge_store_search(query: str, count: int = 5) -> str:
        """
        Search the offline evidence knowledge store for the claim under investigation.

        This searches pre-scraped web documents collected for this specific claim;
        there is no live internet access. Issue several differently-phrased queries
        to cover the aspect you are verifying.

        Args:
            query: Search query string.
            count: Number of text chunks to return (1-20).

        Returns:
            A JSON string with query metadata and scored document chunks.
        """
        result = await run_knowledge_store_search(query=query, count=count)
        return json.dumps(result, ensure_ascii=False)

    return knowledge_store_search
