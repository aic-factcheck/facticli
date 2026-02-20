from __future__ import annotations

from typing import Any


class FakeRunResult:
    """Minimal stand-in for the openai-agents Runner.run() result object."""

    def __init__(self, output: Any) -> None:
        self.output = output

    def final_output_as(self, _cls: type, raise_if_incorrect_type: bool = False) -> Any:
        return self.output
