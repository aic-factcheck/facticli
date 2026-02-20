from __future__ import annotations

from dataclasses import dataclass

from facticli.application.config import ClaimExtractionRuntimeConfig
from facticli.application.factory import build_claim_extraction_service
from facticli.core.contracts import ClaimExtractionResult


@dataclass(frozen=True)
class ClaimExtractorConfig(ClaimExtractionRuntimeConfig):
    pass


class ClaimExtractor:
    def __init__(self, config: ClaimExtractorConfig):
        self.config = config
        self._service = build_claim_extraction_service(config)

    async def extract(self, input_text: str) -> ClaimExtractionResult:
        return await self._service.extract_claims(input_text)
