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
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    duration_seconds: float = 0.0
    started_at: str = ""


class UsageSummary(BaseModel):
    """Aggregated usage across one run, totalled and broken down per stage."""
    requests: int = 0
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    llm_seconds: float = 0.0
    cache_hit_rate: float = 0.0
    per_stage: dict[str, dict[str, float]] = Field(default_factory=dict)


class BudgetStatus(BaseModel):
    """Run-level budget configuration and how much of it was consumed."""
    token_budget: int | None = None
    tokens_used: int = 0
    token_budget_exhausted: bool = False
    retry_budget: int | None = None
    retries_used: int = 0
    retry_budget_exhausted: bool = False
    skipped_stages: list[str] = Field(default_factory=list)


class UsageLog:
    """Mutable per-run collector shared across stage adapters via ContextVar.

    Besides collecting usage events, the log acts as the run's budget ledger
    (Codex-style rollout budget): every stage consults it before spending, and
    model retries are drawn from a shared retry allowance instead of being
    unbounded per call.
    """

    def __init__(self, *, token_budget: int | None = None, retry_budget: int | None = None) -> None:
        self.events: list[StageUsageEvent] = []
        self.token_budget = token_budget
        self.retry_budget = retry_budget
        self.retries_used = 0
        self.skipped_stages: list[str] = []

    @property
    def tokens_used(self) -> int:
        return sum(event.total_tokens for event in self.events)

    def tokens_remaining(self) -> int | None:
        if self.token_budget is None:
            return None
        return max(0, self.token_budget - self.tokens_used)

    def token_budget_exhausted(self) -> bool:
        return self.token_budget is not None and self.tokens_used >= self.token_budget

    def try_consume_retry(self) -> bool:
        """Reserve one retry from the shared allowance; False when none is left."""
        if self.retry_budget is not None and self.retries_used >= self.retry_budget:
            return False
        self.retries_used += 1
        return True

    def retry_budget_exhausted(self) -> bool:
        return self.retry_budget is not None and self.retries_used >= self.retry_budget

    def record_skipped_stage(self, stage: str) -> None:
        self.skipped_stages.append(stage)

    def budget_status(self) -> BudgetStatus:
        return BudgetStatus(
            token_budget=self.token_budget,
            tokens_used=self.tokens_used,
            token_budget_exhausted=self.token_budget_exhausted(),
            retry_budget=self.retry_budget,
            retries_used=self.retries_used,
            retry_budget_exhausted=self.retry_budget_exhausted(),
            skipped_stages=list(self.skipped_stages),
        )


_active_usage_log: ContextVar[UsageLog | None] = ContextVar("facticli_usage_log", default=None)


def activate_usage_log(
    *,
    token_budget: int | None = None,
    retry_budget: int | None = None,
) -> tuple[UsageLog, Token]:
    """Install a fresh usage log (and budget ledger) for the current async context."""
    log = UsageLog(token_budget=token_budget, retry_budget=retry_budget)
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
            cached_input_tokens=_detail(usage, "input_tokens_details", "cached_tokens"),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            reasoning_tokens=_detail(usage, "output_tokens_details", "reasoning_tokens"),
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
        summary.cached_input_tokens += event.cached_input_tokens
        summary.output_tokens += event.output_tokens
        summary.reasoning_tokens += event.reasoning_tokens
        summary.total_tokens += event.total_tokens
        summary.llm_seconds += event.duration_seconds
        bucket = summary.per_stage.setdefault(
            event.stage,
            {
                "calls": 0,
                "requests": 0,
                "input_tokens": 0,
                "cached_input_tokens": 0,
                "output_tokens": 0,
                "reasoning_tokens": 0,
                "total_tokens": 0,
                "llm_seconds": 0.0,
            },
        )
        bucket["calls"] += 1
        bucket["requests"] += event.requests
        bucket["input_tokens"] += event.input_tokens
        bucket["cached_input_tokens"] += event.cached_input_tokens
        bucket["output_tokens"] += event.output_tokens
        bucket["reasoning_tokens"] += event.reasoning_tokens
        bucket["total_tokens"] += event.total_tokens
        bucket["llm_seconds"] += event.duration_seconds
    if summary.input_tokens:
        summary.cache_hit_rate = round(summary.cached_input_tokens / summary.input_tokens, 4)
    return summary
