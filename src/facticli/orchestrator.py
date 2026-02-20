from __future__ import annotations

from dataclasses import dataclass

from facticli.application.config import FactCheckRuntimeConfig
from facticli.application.factory import build_fact_check_service
from facticli.application.repository import InMemoryRunArtifactRepository
from facticli.application.services import FactCheckRun
from facticli.core.artifacts import RunArtifacts


@dataclass(frozen=True)
class OrchestratorConfig(FactCheckRuntimeConfig):
    pass


class FactCheckOrchestrator:
    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.artifact_repository = InMemoryRunArtifactRepository()
        self._service = build_fact_check_service(
            config=config,
            artifact_repository=self.artifact_repository,
        )

    async def check_claim(self, claim: str) -> FactCheckRun:
        return await self._service.check_claim(claim)

    def latest_artifacts(self) -> RunArtifacts | None:
        if not self.artifact_repository.runs:
            return None
        return self.artifact_repository.runs[-1]
