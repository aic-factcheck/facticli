from __future__ import annotations

from contextvars import ContextVar, Token
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class StageUsageEvent(BaseModel):
    """One LLM call's token usage and latency, attributed to a pipeline stage."""
    stage: str
    model: str | None = None
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    duration_seconds: float = 0.0
    started_at: str = ""


class UsageSummary(BaseModel):
    """Aggregated usage across one run, totalled and broken down per stage."""
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    llm_seconds: float = 0.0
    per_stage: dict[str, dict[str, float]] = Field(default_factory=dict)


class UsageLog:
    """Mutable per-run collector shared across stage adapters via ContextVar."""
    def __init__(self) -> None:
        self.events: list[StageUsageEvent] = []


_active_usage_log: ContextVar[UsageLog | None] = ContextVar("facticli_usage_log", default=None)


def activate_usage_log() -> tuple[UsageLog, Token]:
    """Install a fresh usage log for the current async context."""
    log = UsageLog()
    token = _active_usage_log.set(log)
    return log, token


def deactivate_usage_log(token: Token) -> None:
    """Restore the previous usage-log context."""
    _active_usage_log.reset(token)


def record_stage_usage(
    *,
    stage: str,
    model: str | None,
    usage: Any,
    duration_seconds: float,
    started_at: str | None = None,
) -> None:
    """Append one usage event to the active log; no-op outside a tracked run."""
    log = _active_usage_log.get()
    if log is None:
        return
    log.events.append(
        StageUsageEvent(
            stage=stage,
            model=model,
            requests=int(getattr(usage, "requests", 0) or 0),
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
            duration_seconds=duration_seconds,
            started_at=started_at or datetime.now(timezone.utc).isoformat(),
        )
    )


def summarize_usage(events: list[StageUsageEvent]) -> UsageSummary:
    """Total the events and aggregate them per stage name."""
    summary = UsageSummary()
    for event in events:
        summary.requests += event.requests
        summary.input_tokens += event.input_tokens
        summary.output_tokens += event.output_tokens
        summary.total_tokens += event.total_tokens
        summary.llm_seconds += event.duration_seconds
        bucket = summary.per_stage.setdefault(
            event.stage,
            {"calls": 0, "requests": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "llm_seconds": 0.0},
        )
        bucket["calls"] += 1
        bucket["requests"] += event.requests
        bucket["input_tokens"] += event.input_tokens
        bucket["output_tokens"] += event.output_tokens
        bucket["total_tokens"] += event.total_tokens
        bucket["llm_seconds"] += event.duration_seconds
    return summary
