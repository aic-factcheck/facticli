from __future__ import annotations

import re
from collections.abc import Iterable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .contracts import VerificationCheck


def sanitize_aspect_id(raw_aspect_id: str, fallback_index: int) -> str:
    lowered = raw_aspect_id.strip().lower()
    cleaned = re.sub(r"[^a-z0-9_]+", "_", lowered).strip("_")
    return cleaned or f"check_{fallback_index}"


def normalize_text_list(values: list[str], max_items: int = 8) -> list[str]:
    """Strip, drop empties, dedupe case-insensitively, and cap a list of short texts."""
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = " ".join(value.split())
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(text)
        if len(normalized) >= max(1, max_items):
            break
    return normalized


def normalize_plan_checks(
    claim: str,
    checks: list[VerificationCheck],
    max_checks: int,
    max_search_queries_per_check: int,
    taken_aspect_ids: Iterable[str] = (),
) -> list[VerificationCheck]:
    """Repair and cap checks; every aspect_id is unique, also against ``taken_aspect_ids``."""
    normalized_checks: list[VerificationCheck] = []
    used_aspect_ids: set[str] = set(taken_aspect_ids)

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
                    "search_queries": normalize_text_list(
                        [*check.search_queries, question, claim], max_items=max_search_queries_per_check
                    ),
                    "acceptance_criteria": normalize_text_list(check.acceptance_criteria),
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
