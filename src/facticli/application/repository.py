from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from facticli.core.artifacts import RunArtifacts


class RunArtifactRepository(Protocol):
    def save(self, artifacts: RunArtifacts) -> None:
        ...


@dataclass
class InMemoryRunArtifactRepository(RunArtifactRepository):
    runs: list[RunArtifacts] = field(default_factory=list)

    def save(self, artifacts: RunArtifacts) -> None:
        self.runs.append(artifacts)
