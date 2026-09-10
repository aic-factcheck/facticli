from __future__ import annotations

import asyncio
from enum import Enum


class FacticliError(Exception):
    """Base class for all facticli errors."""


class ConfigError(FacticliError):
    """Invalid or missing configuration (API keys, provider names, etc.)."""


class SchemaError(FacticliError):
    """Model output did not match the expected Pydantic schema."""


class TransientError(FacticliError):
    """Retriable failure: network timeout, rate limit, temporary API error."""


class BudgetExhaustedError(FacticliError):
    """The run-level token or retry budget was used up before the stage could run."""


class ResearchError(FacticliError):
    """A research check failed after all retry attempts."""

    def __init__(self, aspect_id: str, message: str) -> None:
        self.aspect_id = aspect_id
        super().__init__(f"[{aspect_id}] {message}")


class ErrorKind(str, Enum):
    """Provider-agnostic failure taxonomy used for retry policy and artifacts.

    Retryable kinds are transient transport or capacity problems. Everything
    else either needs a different request (schema, context overflow, bad
    request), a human (auth, config), or is a deliberate stop (budget).
    """

    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    NETWORK = "network"
    SERVER = "server"
    AUTH = "auth"
    NOT_FOUND = "not_found"
    BAD_REQUEST = "bad_request"
    CONTEXT_OVERFLOW = "context_overflow"
    SCHEMA = "schema"
    MAX_TURNS = "max_turns"
    TOOL = "tool"
    BUDGET = "budget"
    UNKNOWN = "unknown"


RETRYABLE_KINDS: frozenset[ErrorKind] = frozenset(
    {ErrorKind.RATE_LIMIT, ErrorKind.TIMEOUT, ErrorKind.NETWORK, ErrorKind.SERVER}
)

RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

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


def classify_status_code(status_code: int | None, message: str = "") -> ErrorKind:
    """Map an HTTP status (and optional message) to an error kind."""
    if status_code is None:
        return ErrorKind.UNKNOWN
    if status_code == 429:
        return ErrorKind.RATE_LIMIT
    if status_code in (401, 403):
        return ErrorKind.AUTH
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
    if isinstance(exc, BudgetExhaustedError):
        return ErrorKind.BUDGET
    if isinstance(exc, SchemaError):
        return ErrorKind.SCHEMA
    if isinstance(exc, TransientError):
        return ErrorKind.NETWORK
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return ErrorKind.TIMEOUT

    class_names = {cls.__name__ for cls in type(exc).__mro__}
    message = str(exc)

    if "MaxTurnsExceeded" in class_names:
        return ErrorKind.MAX_TURNS
    if "ModelBehaviorError" in class_names or "ValidationError" in class_names:
        return ErrorKind.SCHEMA
    if "ToolTimeoutError" in class_names:
        return ErrorKind.TIMEOUT
    if "RateLimitError" in class_names:
        return ErrorKind.RATE_LIMIT
    if "APITimeoutError" in class_names or "ReadTimeout" in class_names or "ConnectTimeout" in class_names:
        return ErrorKind.TIMEOUT
    if "AuthenticationError" in class_names or "PermissionDeniedError" in class_names:
        return ErrorKind.AUTH
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


def is_retryable(kind: ErrorKind) -> bool:
    """True for transient kinds worth another attempt with backoff."""
    return kind in RETRYABLE_KINDS


def describe_exception(exc: BaseException) -> str:
    """Compact ``kind: Type: message`` string for artifacts and progress events."""
    return f"{classify_exception(exc).value}: {type(exc).__name__}: {exc}"
