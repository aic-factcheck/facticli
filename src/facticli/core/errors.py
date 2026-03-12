from __future__ import annotations


class FacticliError(Exception):
    """Base class for all facticli errors."""


class ConfigError(FacticliError):
    """Invalid or missing configuration (API keys, provider names, etc.)."""


class SchemaError(FacticliError):
    """Model output did not match the expected Pydantic schema."""


class TransientError(FacticliError):
    """Retriable failure: network timeout, rate limit, temporary API error."""


class ResearchError(FacticliError):
    """A research check failed after all retry attempts."""

    def __init__(self, aspect_id: str, message: str) -> None:
        self.aspect_id = aspect_id
        super().__init__(f"[{aspect_id}] {message}")
