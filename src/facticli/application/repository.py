from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from facticli.core.artifacts import RunArtifacts


class RunArtifactRepository(Protocol):
    """Persistence interface for storing run artifacts after execution."""
    def save(self, artifacts: RunArtifacts) -> None:
        """Persist artifacts produced by a completed run."""
        ...


@dataclass
class InMemoryRunArtifactRepository(RunArtifactRepository):
    """Simple repository used by tests and local sessions."""
    runs: list[RunArtifacts] = field(default_factory=list)

    def save(self, artifacts: RunArtifacts) -> None:
        self.runs.append(artifacts)
