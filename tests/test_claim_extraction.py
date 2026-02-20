from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from facticli.claim_extraction import ClaimExtractor, ClaimExtractorConfig
from facticli.types import CheckworthyClaim, ClaimExtractionResult


class ClaimExtractionWrapperTests(unittest.IsolatedAsyncioTestCase):
    async def test_extractor_delegates_to_built_service(self):
        fake_result = ClaimExtractionResult(
            input_text="x",
            claims=[
                CheckworthyClaim(
                    claim_id="c1",
                    claim_text="The first iPhone was released in 2007.",
                    source_fragment="first iPhone was released in 2007",
                    checkworthy_reason="Checkable release date claim.",
                )
            ],
        )
        fake_service = AsyncMock()
        fake_service.extract_claims = AsyncMock(return_value=fake_result)

        with patch("facticli.claim_extraction.build_claim_extraction_service", return_value=fake_service):
            extractor = ClaimExtractor(ClaimExtractorConfig(inference_provider="openai-agents"))
            result = await extractor.extract("The first iPhone was released in 2007.")

        self.assertEqual(result.claims[0].claim_id, "c1")
        fake_service.extract_claims.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
