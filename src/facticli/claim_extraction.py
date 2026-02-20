from __future__ import annotations

import json
from dataclasses import dataclass

from agents import Agent, ModelSettings, Runner

from .gemini_inference import GeminiStructuredClient
from .skills import load_skill_prompt
from .types import ClaimExtractionResult


@dataclass(frozen=True)
class ClaimExtractorConfig:
    inference_provider: str = "openai-agents"
    model: str = "gpt-4.1-mini"
    gemini_model: str = "gemini-3-pro"
    max_claims: int = 12
    max_turns: int = 8


class ClaimExtractor:
    def __init__(self, config: ClaimExtractorConfig):
        self.config = config
        self.instructions = load_skill_prompt("extract_claims")

        self._openai_agent: Agent[None] | None = None
        self._gemini_client: GeminiStructuredClient | None = None

        if config.inference_provider == "openai-agents":
            self._openai_agent = Agent(
                name="checkworthy_claim_extractor",
                instructions=self.instructions,
                output_type=ClaimExtractionResult,
                model=config.model,
                model_settings=ModelSettings(
                    temperature=0.1,
                    parallel_tool_calls=False,
                ),
            )
        elif config.inference_provider == "gemini":
            self._gemini_client = GeminiStructuredClient(model=config.gemini_model)
        else:
            raise ValueError(f"Unsupported inference provider: {config.inference_provider}")

    async def extract(self, input_text: str) -> ClaimExtractionResult:
        normalized_text = input_text.strip()
        if not normalized_text:
            raise ValueError("Input text is empty.")

        payload = {
            "input_text": normalized_text,
            "requirements": {
                "max_claims": self.config.max_claims,
                "decontextualized": True,
                "atomic_claims": True,
                "maximize_checkworthy_coverage": True,
                "only_directly_mentioned_facts": True,
            },
        }

        if self.config.inference_provider == "openai-agents":
            if self._openai_agent is None:
                raise RuntimeError("OpenAI extraction agent is not initialized.")
            result = await Runner.run(
                self._openai_agent,
                json.dumps(payload, indent=2),
                max_turns=self.config.max_turns,
            )
            extraction = result.final_output_as(ClaimExtractionResult, raise_if_incorrect_type=True)
        else:
            if self._gemini_client is None:
                raise RuntimeError("Gemini extraction client is not initialized.")
            extraction = await self._gemini_client.generate_structured(
                instructions=self.instructions,
                payload=payload,
                output_model=ClaimExtractionResult,
            )

        extraction.input_text = normalized_text
        extraction.claims = extraction.claims[: self.config.max_claims]
        for idx, claim in enumerate(extraction.claims, start=1):
            if not claim.claim_id.strip():
                claim.claim_id = f"claim_{idx}"

        return extraction

