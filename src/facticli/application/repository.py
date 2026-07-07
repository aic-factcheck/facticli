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


@dataclass
class FileRunArtifactRepository(RunArtifactRepository):
    """Persists each run as one JSON file for replayable offline evaluation."""
    output_dir: str

    def save(self, artifacts: RunArtifacts) -> None:
        import hashlib
        import uuid
        from datetime import datetime, timezone
        from pathlib import Path

        directory = Path(self.output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        claim_key = (
            str(artifacts.claim_id)
            if artifacts.claim_id is not None
            else hashlib.sha1(artifacts.normalized_claim.encode("utf-8")).hexdigest()[:10]
        )
        filename = f"run_{stamp}_{claim_key}_{uuid.uuid4().hex[:6]}.json"
        path = directory / filename
        path.write_text(artifacts.model_dump_json(indent=2), encoding="utf-8")
