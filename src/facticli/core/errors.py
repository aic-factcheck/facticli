from __future__ import annotations

import asyncio
from enum import Enum


class SearchAuthError(Exception):
    """The search provider rejected its API key; no query can succeed until the key is fixed."""


class ErrorKind(str, Enum):
    """Provider-agnostic failure taxonomy for artifacts and the research retry decision.

    Transient transport and capacity problems (rate limit, timeout, network,
    server) are retried by the OpenAI client. Everything else needs either a different request (schema, context overflow, bad
    request, refusal, forbidden) or a human (auth, quota). ``quota`` is split
    from ``rate_limit`` because both arrive as HTTP 429: a throttled request succeeds after backoff, an
    exhausted account never does. ``forbidden`` (HTTP 403) is split from
    ``auth`` (HTTP 401) because a 403 can be about one request (a moderation
    block, a model the key may not use) while a 401 is about the key itself.
    """

    RATE_LIMIT = "rate_limit"
    QUOTA = "quota"
    TIMEOUT = "timeout"
    NETWORK = "network"
    SERVER = "server"
    AUTH = "auth"
    FORBIDDEN = "forbidden"
    NOT_FOUND = "not_found"
    BAD_REQUEST = "bad_request"
    CONTEXT_OVERFLOW = "context_overflow"
    SCHEMA = "schema"
    MAX_TURNS = "max_turns"
    REFUSAL = "refusal"
    UNKNOWN = "unknown"


RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

# Kinds that no retry, rephrasing, or other claim can fix: every later request
# with the same credentials and configuration fails the same way (a rejected
# key, an account out of credit, a model name the endpoint does not know). A
# run stops on these instead of judging a claim on no evidence, and batch
# runners stop instead of writing fallback predictions claim after claim.
FATAL_KINDS: frozenset[ErrorKind] = frozenset({ErrorKind.AUTH, ErrorKind.QUOTA, ErrorKind.NOT_FOUND})

# Out-of-credit is recognized by the provider's error code, as the structured
# field or inside the serialized error body, never by the prose around it:
# ordinary per-minute throttling messages also say "you exceeded your current
# quota", and a fatal misclassification stops a whole batch.
_QUOTA_CODE = "insufficient_quota"

_CONTEXT_OVERFLOW_MARKERS: tuple[str, ...] = (
    "context_length_exceeded",
    "context length",
    "maximum context",
    "too many tokens",
    "prompt is too long",
    "input is too long",
    "exceeds the model's maximum",
    "reduce the length",
    "request too large",
)


def _looks_like_context_overflow(message: str) -> bool:
    lowered = message.lower()
    return any(marker in lowered for marker in _CONTEXT_OVERFLOW_MARKERS)


def _is_out_of_credit(message: str, code: object = None) -> bool:
    return code == _QUOTA_CODE or _QUOTA_CODE in message


def classify_status_code(status_code: int | None, message: str = "") -> ErrorKind:
    """Map an HTTP status (and optional message) to an error kind."""
    if status_code is None:
        return ErrorKind.UNKNOWN
    if status_code == 429:
        return ErrorKind.QUOTA if _is_out_of_credit(message) else ErrorKind.RATE_LIMIT
    if status_code == 402:
        return ErrorKind.QUOTA
    if status_code == 401:
        return ErrorKind.AUTH
    if status_code == 403:
        return ErrorKind.FORBIDDEN
    if status_code == 404:
        return ErrorKind.NOT_FOUND
    if status_code in (408, 504):
        return ErrorKind.TIMEOUT
    if status_code in RETRYABLE_STATUS_CODES or status_code >= 500:
        return ErrorKind.SERVER
    if status_code in (400, 413, 422):
        return ErrorKind.CONTEXT_OVERFLOW if _looks_like_context_overflow(message) else ErrorKind.BAD_REQUEST
    if 400 <= status_code < 500:
        return ErrorKind.BAD_REQUEST
    return ErrorKind.UNKNOWN


def classify_exception(exc: BaseException) -> ErrorKind:
    """Classify any exception raised around a model or tool call.

    Works without importing provider SDKs at module import time; the OpenAI
    and Agents SDK classes are matched by name and by ``status_code`` so the
    taxonomy also applies to OpenAI-compatible third-party endpoints.
    """
    if isinstance(exc, SearchAuthError):
        return ErrorKind.AUTH
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return ErrorKind.TIMEOUT

    class_names = {cls.__name__ for cls in type(exc).__mro__}
    message = str(exc)

    if "UserError" in class_names and exc.__cause__ is not None:
        # The Agents SDK wraps whatever a tool raised; classify the tool's own error.
        return classify_exception(exc.__cause__)

    if "MaxTurnsExceeded" in class_names:
        return ErrorKind.MAX_TURNS
    if "ModelRefusalError" in class_names:
        return ErrorKind.REFUSAL
    if "ModelBehaviorError" in class_names or "ValidationError" in class_names:
        return ErrorKind.SCHEMA
    if "ToolTimeoutError" in class_names or "ModelTimeoutError" in class_names:
        return ErrorKind.TIMEOUT
    if "RateLimitError" in class_names:
        return ErrorKind.QUOTA if _is_out_of_credit(message, getattr(exc, "code", None)) else ErrorKind.RATE_LIMIT
    if "APITimeoutError" in class_names or "ReadTimeout" in class_names or "ConnectTimeout" in class_names:
        return ErrorKind.TIMEOUT
    if "AuthenticationError" in class_names:
        return ErrorKind.AUTH
    if "PermissionDeniedError" in class_names:
        return ErrorKind.FORBIDDEN
    if "NotFoundError" in class_names:
        return ErrorKind.NOT_FOUND
    if "InternalServerError" in class_names:
        return ErrorKind.SERVER
    if "BadRequestError" in class_names or "UnprocessableEntityError" in class_names:
        return ErrorKind.CONTEXT_OVERFLOW if _looks_like_context_overflow(message) else ErrorKind.BAD_REQUEST

    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return classify_status_code(status_code, message)

    if "APIConnectionError" in class_names or any(
        name in class_names for name in ("ConnectError", "NetworkError", "RemoteProtocolError", "ConnectionError", "OSError")
    ):
        return ErrorKind.NETWORK
    if _looks_like_context_overflow(message):
        return ErrorKind.CONTEXT_OVERFLOW
    return ErrorKind.UNKNOWN


def is_fatal(kind: ErrorKind) -> bool:
    """True when no later request with the same credentials and configuration can succeed."""
    return kind in FATAL_KINDS


def describe_exception(exc: BaseException) -> str:
    """Compact ``kind: Type: message`` string for artifacts and progress events."""
    return f"{classify_exception(exc).value}: {type(exc).__name__}: {exc}"
