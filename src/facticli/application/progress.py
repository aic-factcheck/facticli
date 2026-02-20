from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable


@dataclass(frozen=True)
class ProgressEvent:
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)


ProgressCallback = Callable[[ProgressEvent], Awaitable[None] | None]


async def emit_progress(
    callback: ProgressCallback | None,
    kind: str,
    payload: dict[str, Any] | None = None,
) -> None:
    if callback is None:
        return

    result = callback(ProgressEvent(kind=kind, payload=payload or {}))
    if inspect.isawaitable(result):
        await result
