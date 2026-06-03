from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from facticli.application.config import ClaimExtractionRuntimeConfig
from facticli.application.factory import build_claim_extraction_service

# Convenience: load a local .env so the server picks up OPENAI_API_* the same
# way the CLI does via scripts/test_routine.sh. Optional dependency; ignore if
# python-dotenv is not installed or no .env exists.
try:  # pragma: no cover - trivial best-effort bootstrap
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass

STATIC_DIR = Path(__file__).parent / "static"


class ExtractRequest(BaseModel):
    """Request body for the claim-extraction endpoint."""

    text: str = Field(description="Raw input text to extract claims from.")
    max_claims: int = Field(default=12, ge=1, le=50)
    model: str | None = Field(
        default=None,
        description="Optional model override. Falls back to OPENAI_API_MODEL.",
    )
    base_url: str | None = Field(
        default=None,
        description="Optional OpenAI-compatible base URL override.",
    )


def _resolve_model(requested: str | None) -> str | None:
    candidate = (requested or os.getenv("OPENAI_API_MODEL") or "").strip()
    return candidate or None


def _has_api_key() -> bool:
    return bool((os.getenv("OPENAI_API_KEY") or "").strip())


def create_app() -> FastAPI:
    """Build the FastAPI app serving the claim-extraction GUI and JSON API."""
    app = FastAPI(
        title="CEDMO Claim Extractor",
        description="Extract decontextualized, atomic, check-worthy claims from text.",
        version="1.0.0",
    )

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/static/{filename}", include_in_schema=False)
    async def static_file(filename: str) -> FileResponse:
        candidate = (STATIC_DIR / filename).resolve()
        if not candidate.is_relative_to(STATIC_DIR.resolve()) or not candidate.is_file():
            raise HTTPException(status_code=404, detail="Not found.")
        return FileResponse(candidate)

    @app.get("/api/health")
    async def health() -> dict[str, object]:
        return {
            "status": "ok",
            "has_api_key": _has_api_key(),
            "default_model": _resolve_model(None),
            "base_url": (os.getenv("OPENAI_API_BASE_URL") or "").strip() or None,
        }

    @app.post("/api/extract")
    async def extract(request: ExtractRequest) -> JSONResponse:
        text = request.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Input text is empty.")
        if not _has_api_key():
            raise HTTPException(
                status_code=503,
                detail="OPENAI_API_KEY is not configured on the server.",
            )
        model = _resolve_model(request.model)
        if not model:
            raise HTTPException(
                status_code=503,
                detail="No model configured. Set OPENAI_API_MODEL or pass a model.",
            )

        config = ClaimExtractionRuntimeConfig(
            model=model,
            base_url=(request.base_url or "").strip() or None,
            max_claims=request.max_claims,
        )
        service = build_claim_extraction_service(config)
        try:
            result = await service.extract_claims(text)
        except Exception as exc:  # surface upstream/model errors to the client
            raise HTTPException(status_code=502, detail=f"Claim extraction failed: {exc}") from exc

        return JSONResponse(result.model_dump())

    return app


app = create_app()
