from __future__ import annotations

from agents import ModelRetryBackoffSettings, ModelRetrySettings, RetryDecision, RetryPolicyContext

from facticli.core.errors import ErrorKind, classify_exception, classify_status_code, is_retryable
from facticli.core.usage import get_usage_log

# Backoff mirrors the consensus in 2026 harnesses (Codex request_max_retries,
# OpenClaw failover, pi retry.baseDelayMs): exponential with full jitter,
# capped, and honouring provider Retry-After hints when present.
DEFAULT_BACKOFF = ModelRetryBackoffSettings(initial_delay=1.0, max_delay=20.0, multiplier=2.0, jitter=True)


def classify_retry_context(context: RetryPolicyContext) -> ErrorKind:
    """Derive an ErrorKind from the SDK's normalized error facts, then the exception."""
    normalized = context.normalized
    if normalized.is_timeout:
        return ErrorKind.TIMEOUT
    if normalized.is_network_error:
        return ErrorKind.NETWORK
    if normalized.status_code is not None:
        return classify_status_code(normalized.status_code, normalized.message or str(context.error))
    return classify_exception(context.error)


def facticli_retry_policy(context: RetryPolicyContext) -> RetryDecision:
    """Retry transient failures only, drawing from the run-level retry budget.

    Non-retryable kinds (auth, bad request, context overflow, schema) fail
    fast so the stage can convert them into an explicit failed finding
    instead of burning the budget on requests that cannot succeed.
    """
    kind = classify_retry_context(context)
    if not is_retryable(kind):
        return RetryDecision(retry=False, reason=f"non-retryable error kind: {kind.value}")

    log = get_usage_log()
    if log is not None and not log.try_consume_retry():
        return RetryDecision(retry=False, reason="run retry budget exhausted")

    delay = context.normalized.retry_after
    if context.provider_advice is not None and context.provider_advice.retry_after is not None:
        delay = context.provider_advice.retry_after
    return RetryDecision(retry=True, delay=delay, reason=f"retrying {kind.value}")


def build_retry_settings(max_retries: int) -> ModelRetrySettings | None:
    """Build SDK retry settings for ``ModelSettings.retry``; None disables retries."""
    if max_retries <= 0:
        return None
    return ModelRetrySettings(max_retries=max_retries, backoff=DEFAULT_BACKOFF, policy=facticli_retry_policy)
