from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from facticli.claim_extraction import ClaimExtractor, ClaimExtractorConfig
from facticli.types import CheckworthyClaim, ClaimExtractionResult


class _FakeRunResult:
    def __init__(self, output: ClaimExtractionResult):
        self.output = output

    def final_output_as(self, _cls, raise_if_incorrect_type: bool = False):
        return self.output


class ClaimExtractionTests(unittest.IsolatedAsyncioTestCase):
    async def test_openai_extractor_caps_claims_and_fills_missing_ids(self):
        extraction = ClaimExtractionResult(
            input_text="ignored-by-normalizer",
            claims=[
                CheckworthyClaim(
                    claim_id="",
                    claim_text="Inflation fell below 3 percent in 2025.",
                    source_fragment="inflation fell below 3% in 2025",
                    checkworthy_reason="Specific numeric macroeconomic claim.",
                ),
                CheckworthyClaim(
                    claim_id="claim_b",
                    claim_text="Wages rose by 10 percent.",
                    source_fragment="wages rose 10%",
                    checkworthy_reason="Specific numeric socioeconomic claim.",
                ),
                CheckworthyClaim(
                    claim_id="claim_c",
                    claim_text="The government built 200,000 homes.",
                    source_fragment="built 200,000 new homes",
                    checkworthy_reason="Specific policy output claim.",
                ),
            ],
        )

        with patch(
            "facticli.claim_extraction.Runner.run",
            new=AsyncMock(return_value=_FakeRunResult(extraction)),
        ):
            extractor = ClaimExtractor(
                ClaimExtractorConfig(
                    inference_provider="openai-agents",
                    max_claims=2,
                )
            )
            result = await extractor.extract("  Debate transcript content  ")

        self.assertEqual(result.input_text, "Debate transcript content")
        self.assertEqual(len(result.claims), 2)
        self.assertEqual(result.claims[0].claim_id, "claim_1")
        self.assertEqual(result.claims[1].claim_id, "claim_b")

    async def test_gemini_extractor_calls_structured_generation(self):
        fake_result = ClaimExtractionResult(
            input_text="any",
            claims=[
                CheckworthyClaim(
                    claim_id="c1",
                    claim_text="The first iPhone was released in 2007.",
                    source_fragment="first iPhone was released in 2007",
                    checkworthy_reason="Checkable historical release date claim.",
                )
            ],
        )

        fake_client = AsyncMock()
        fake_client.generate_structured = AsyncMock(return_value=fake_result)

        with patch("facticli.claim_extraction.GeminiStructuredClient", return_value=fake_client):
            extractor = ClaimExtractor(
                ClaimExtractorConfig(
                    inference_provider="gemini",
                    gemini_model="gemini-3-pro",
                )
            )
            result = await extractor.extract("The first iPhone was released in 2007.")

        self.assertEqual(len(result.claims), 1)
        self.assertEqual(result.claims[0].claim_id, "c1")
        fake_client.generate_structured.assert_awaited_once()

    async def test_extract_rejects_empty_input(self):
        extractor = ClaimExtractor(ClaimExtractorConfig(inference_provider="openai-agents"))
        with self.assertRaises(ValueError):
            await extractor.extract("   ")


if __name__ == "__main__":
    unittest.main()

