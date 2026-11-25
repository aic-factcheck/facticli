from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class StageUsageEvent(BaseModel):
    """One tracked model run's token usage and latency, attributed to a pipeline stage.

    ``completed`` is False for a run that raised or was cancelled: it still
    spent tokens, and the ledger counts every billed attempt, not only the
    ones that produced a result. ``wrap_up`` marks the tools-off call that
    salvages a run at its turn limit.
    """
    stage: str
    model: str | None = None
    requests: int = 0
    input_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    duration_seconds: float = 0.0
    started_at: str = ""
    completed: bool = True
    wrap_up: bool = False


class UsageSummary(BaseModel):
    """Aggregated usage across one run, totalled and broken down per stage.

    ``failed_attempts`` / ``failed_attempt_tokens`` cover runs that raised or
    were cancelled (timeouts) after spending tokens.
    """
    requests: int = 0
    input_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    llm_seconds: float = 0.0
    cache_hit_rate: float = 0.0
    failed_attempts: int = 0
    failed_attempt_tokens: int = 0
    wrap_up_calls: int = 0
    per_stage: dict[str, dict[str, float]] = Field(default_factory=dict)


class UsageLog:
    """Per-run collector of usage events, shared across stages via ContextVar.

    Events are appended when a model run starts and updated in place after
    every response, so runs that are still in flight or that end in an error
    are counted too.
    """

    def __init__(self) -> None:
        self.events: list[StageUsageEvent] = []

    @property
    def tokens_used(self) -> int:
        return sum(event.total_tokens for event in self.events)


_active_usage_log: ContextVar[UsageLog | None] = ContextVar("facticli_usage_log", default=None)


def activate_usage_log() -> tuple[UsageLog, Token]:
    """Install a fresh usage log for the current async context."""
    log = UsageLog()
    token = _active_usage_log.set(log)
    return log, token


def deactivate_usage_log(token: Token) -> None:
    """Restore the previous usage-log context."""
    _active_usage_log.reset(token)


def get_usage_log() -> UsageLog | None:
    """Return the active usage log, if any."""
    return _active_usage_log.get()


def _detail(usage: Any, container: str, attribute: str) -> int:
    details = getattr(usage, container, None)
    if details is None:
        return 0
    value = getattr(details, attribute, None)
    if value is None and isinstance(details, dict):
        value = details.get(attribute)
    return int(value or 0)


def apply_usage(event: StageUsageEvent, usage: Any) -> None:
    """Overwrite an event's token counters from a cumulative SDK usage object."""
    if usage is None:
        return
    event.requests = int(getattr(usage, "requests", 0) or 0)
    event.input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    event.cached_input_tokens = _detail(usage, "input_tokens_details", "cached_tokens")
    event.cache_write_tokens = _detail(usage, "input_tokens_details", "cache_write_tokens")
    event.output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    event.reasoning_tokens = _detail(usage, "output_tokens_details", "reasoning_tokens")
    event.total_tokens = int(getattr(usage, "total_tokens", 0) or 0)


@contextmanager
def track_usage(*, stage: str, model: str | None, wrap_up: bool = False) -> Iterator[StageUsageEvent | None]:
    """Open a usage event around one model run; yields None outside a tracked run.

    The event joins the ledger before the run starts, and the caller refreshes
    it with ``apply_usage`` as responses arrive, so tokens spent by a run that
    raises, times out, or is cancelled are still counted. It is marked
    completed only when the block exits normally.
    """
    log = _active_usage_log.get()
    if log is None:
        yield None
        return
    event = StageUsageEvent(
        stage=stage,
        model=model,
        started_at=datetime.now(timezone.utc).isoformat(),
        completed=False,
        wrap_up=wrap_up,
    )
    log.events.append(event)
    start = time.monotonic()
    try:
        yield event
        event.completed = True
    finally:
        event.duration_seconds = time.monotonic() - start


_TOKEN_FIELDS: tuple[str, ...] = (
    "requests",
    "input_tokens",
    "cached_input_tokens",
    "cache_write_tokens",
    "output_tokens",
    "reasoning_tokens",
    "total_tokens",
)
_SUMMED_FIELDS: tuple[str, ...] = (*_TOKEN_FIELDS, "llm_seconds", "failed_attempts", "failed_attempt_tokens")


def summarize_usage(events: list[StageUsageEvent]) -> UsageSummary:
    """Aggregate the events per stage name and total the stages."""
    summary = UsageSummary()
    for event in events:
        bucket = summary.per_stage.setdefault(event.stage, dict.fromkeys(("calls", *_SUMMED_FIELDS), 0))
        bucket["calls"] += 1
        for name in _TOKEN_FIELDS:
            bucket[name] += getattr(event, name)
        if event.wrap_up:
            # Runs inside its primary run, whose duration already covers it.
            summary.wrap_up_calls += 1
        else:
            bucket["llm_seconds"] += event.duration_seconds
        if not event.completed:
            bucket["failed_attempts"] += 1
            bucket["failed_attempt_tokens"] += event.total_tokens
    for name in _SUMMED_FIELDS:
        setattr(summary, name, sum(bucket[name] for bucket in summary.per_stage.values()))
    if summary.input_tokens:
        summary.cache_hit_rate = round(summary.cached_input_tokens / summary.input_tokens, 4)
    return summary
