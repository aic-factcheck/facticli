from __future__ import annotations

import asyncio
import json
import os
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

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
        # Schema is enforced by constrained decoding (response_schema), so the
        # prompt only needs to carry the task instructions and input payload.
        prompt = (
            f"{instructions}\n\n"
            "Input payload (JSON):\n"
            f"{json.dumps(payload, indent=2)}"
        )

        raw_text = await asyncio.to_thread(self._generate_text, prompt, output_model)

        try:
            parsed = json.loads(raw_text.strip())
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Gemini returned invalid JSON for {output_model.__name__}: "
                f"{raw_text[:300]!r}"
            ) from exc

        try:
            return output_model.model_validate(parsed)
        except ValidationError as exc:
            raise RuntimeError(
                f"Gemini output did not match {output_model.__name__} schema: {exc}"
            ) from exc

    def _generate_text(self, prompt: str, output_model: type[BaseModel]) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": output_model,
            },
        )

        direct_text = getattr(response, "text", None)
        if isinstance(direct_text, str) and direct_text.strip():
            return direct_text

        # Fallback: reconstruct from candidates (older SDK versions).
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
