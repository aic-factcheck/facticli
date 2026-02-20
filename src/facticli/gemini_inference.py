from __future__ import annotations

import asyncio
import json
import os
from typing import Any, TypeVar

from pydantic import BaseModel

TModel = TypeVar("TModel", bound=BaseModel)


class GeminiStructuredClient:
    def __init__(self, model: str, api_key: str | None = None):
        resolved_key = api_key or os.getenv("GEMINI_API_KEY")
        if not resolved_key:
            raise RuntimeError("GEMINI_API_KEY is not set.")

        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError(
                "google-genai is not installed. Install dependencies with `pip install -e .`."
            ) from exc

        self._genai = genai
        self.model = model
        self.client = genai.Client(api_key=resolved_key)

    async def generate_structured(
        self,
        instructions: str,
        payload: dict[str, Any],
        output_model: type[TModel],
    ) -> TModel:
        schema_json = json.dumps(output_model.model_json_schema(), indent=2)
        prompt = (
            f"{instructions}\n\n"
            "Input payload (JSON):\n"
            f"{json.dumps(payload, indent=2)}\n\n"
            "Output requirements:\n"
            "- Return only valid JSON.\n"
            "- Do not include markdown fences.\n"
            "- Match this JSON schema exactly:\n"
            f"{schema_json}\n"
        )

        raw_text = await asyncio.to_thread(self._generate_text, prompt)
        parsed = self._parse_json(raw_text)
        return output_model.model_validate(parsed)

    def _generate_text(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )

        direct_text = getattr(response, "text", None)
        if isinstance(direct_text, str) and direct_text.strip():
            return direct_text

        candidates = getattr(response, "candidates", None) or []
        text_parts: list[str] = []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) or []
            for part in parts:
                text = getattr(part, "text", None)
                if isinstance(text, str) and text:
                    text_parts.append(text)

        if text_parts:
            return "\n".join(text_parts)

        raise RuntimeError("Gemini returned no text content.")

    def _parse_json(self, response_text: str) -> dict[str, Any]:
        text = response_text.strip()

        if "```json" in text:
            text = text.split("```json", maxsplit=1)[1]
        if "```" in text:
            text = text.split("```", maxsplit=1)[0]
        text = text.strip()

        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
            raise ValueError("Expected top-level JSON object.")
        except Exception:
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise RuntimeError(f"Could not parse JSON from Gemini response: {response_text}")

            candidate = text[start : end + 1]
            data = json.loads(candidate)
            if not isinstance(data, dict):
                raise RuntimeError("Gemini response JSON is not an object.")
            return data

