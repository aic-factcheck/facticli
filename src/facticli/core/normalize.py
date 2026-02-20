from __future__ import annotations

import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .contracts import VerificationCheck


def sanitize_aspect_id(raw_aspect_id: str, fallback_index: int) -> str:
    lowered = raw_aspect_id.strip().lower()
    cleaned = re.sub(r"[^a-z0-9_]+", "_", lowered).strip("_")
    return cleaned or f"check_{fallback_index}"


def normalize_query_list(
    queries: list[str],
    fallback: list[str] | None = None,
    max_queries: int = 5,
) -> list[str]:
    candidates = [*queries, *(fallback or [])]
    normalized: list[str] = []
    seen: set[str] = set()

    for candidate in candidates:
        query = candidate.strip()
        if not query:
            continue
        query_key = query.casefold()
        if query_key in seen:
            continue
        seen.add(query_key)
        normalized.append(query)
        if len(normalized) >= max(1, max_queries):
            break

    return normalized


def normalize_plan_checks(
    claim: str,
    checks: list[VerificationCheck],
    max_checks: int,
    max_search_queries_per_check: int,
) -> list[VerificationCheck]:
    normalized_checks: list[VerificationCheck] = []
    used_aspect_ids: set[str] = set()

    for index, check in enumerate(checks, start=1):
        question = check.question.strip()
        if not question:
            continue

        base_aspect_id = sanitize_aspect_id(check.aspect_id, fallback_index=index)
        aspect_id = base_aspect_id
        suffix = 2
        while aspect_id in used_aspect_ids:
            aspect_id = f"{base_aspect_id}_{suffix}"
            suffix += 1
        used_aspect_ids.add(aspect_id)

        normalized_checks.append(
            check.model_copy(
                update={
                    "aspect_id": aspect_id,
                    "question": question,
                    "rationale": check.rationale.strip(),
                    "search_queries": normalize_query_list(
                        check.search_queries,
                        fallback=[question, claim],
                        max_queries=max_search_queries_per_check,
                    ),
                }
            )
        )
        if len(normalized_checks) >= max(1, max_checks):
            break

    return normalized_checks


def normalize_source_url(url: str) -> str:
    stripped = url.strip()
    if not stripped:
        return ""

    try:
        parts = urlsplit(stripped)
    except ValueError:
        return stripped.lower()

    filtered_query = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if not key.lower().startswith("utm_")
    ]
    normalized_query = urlencode(filtered_query, doseq=True)
    normalized_path = parts.path.rstrip("/")

    return urlunsplit(
        (
            parts.scheme.lower(),
            parts.netloc.lower(),
            normalized_path,
            normalized_query,
            "",
        )
    )
