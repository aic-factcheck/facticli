from __future__ import annotations

import os
import secrets
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
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

MAX_INPUT_CHARS = 20_000
DEFAULT_CORS_ORIGINS = "https://aic-factcheck.github.io"
DEFAULT_RATE_LIMIT_REQUESTS = 30
DEFAULT_RATE_LIMIT_WINDOW = 600


class ExtractRequest(BaseModel):
    """Request body for the claim-extraction endpoint.

    Model and endpoint are deliberately NOT client-controllable: this service
    holds the provider credential, so letting a caller redirect the request
    would hand that credential to an arbitrary host.
    """

    text: str = Field(description="Raw input text to extract claims from.", max_length=MAX_INPUT_CHARS)
    max_claims: int = Field(default=12, ge=1, le=50)


def _env(name: str) -> str:
    return (os.getenv(name) or "").strip()


def _resolve_model() -> str | None:
    return _env("OPENAI_API_MODEL") or None


def _has_api_key() -> bool:
    return bool(_env("OPENAI_API_KEY"))


def _auth_enabled() -> bool:
    """Gatekeeping is on unless explicitly disabled for local development."""
    return _env("FACTICLI_API_AUTH").lower() not in {"off", "0", "false", "none"}


def _presented_key(request: Request) -> str:
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        return header[7:].strip()
    return (request.headers.get("x-api-key") or "").strip()


def _rate_limit_settings() -> tuple[int, int]:
    try:
        limit = int(_env("FACTICLI_RATE_LIMIT_REQUESTS") or DEFAULT_RATE_LIMIT_REQUESTS)
    except ValueError:
        limit = DEFAULT_RATE_LIMIT_REQUESTS
    try:
        window = int(_env("FACTICLI_RATE_LIMIT_WINDOW") or DEFAULT_RATE_LIMIT_WINDOW)
    except ValueError:
        window = DEFAULT_RATE_LIMIT_WINDOW
    return max(limit, 1), max(window, 1)


def create_app() -> FastAPI:
    """Build the FastAPI app serving the claim-extraction GUI and JSON API."""
    app = FastAPI(
        title="CEDMO Claim Extractor",
        description="Extract decontextualized, atomic, check-worthy claims from text.",
        version="1.1.0",
    )

    origins = [o.strip() for o in (_env("FACTICLI_CORS_ORIGINS") or DEFAULT_CORS_ORIGINS).split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-API-Key"],
    )

    hits: dict[str, deque[float]] = defaultdict(deque)

    def require_api_key(request: Request) -> str:
        """Gate every credit-spending endpoint behind a shared API key."""
        if not _auth_enabled():
            return "anonymous"
        expected = _env("FACTICLI_API_KEY")
        if not expected:
            # Fail closed: an unconfigured gate must never mean an open gate.
            raise HTTPException(
                status_code=503,
                detail="FACTICLI_API_KEY is not configured on the server; the API is closed.",
            )
        presented = _presented_key(request)
        if not presented or not secrets.compare_digest(presented, expected):
            raise HTTPException(
                status_code=401,
                detail="Missing or invalid API key. Send 'Authorization: Bearer <key>'.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return presented

    def enforce_rate_limit(request: Request, api_key: str = Depends(require_api_key)) -> str:
        limit, window = _rate_limit_settings()
        client = request.client.host if request.client else "unknown"
        bucket = hits[f"{api_key}:{client}"]
        now = time.monotonic()
        while bucket and now - bucket[0] > window:
            bucket.popleft()
        if len(bucket) >= limit:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded ({limit} requests per {window}s). Try again later.",
                headers={"Retry-After": str(window)},
            )
        bucket.append(now)
        return api_key

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
        """Unauthenticated liveness probe; reveals no endpoint or model detail."""
        return {
            "status": "ok",
            "has_api_key": _has_api_key(),
            "model_configured": _resolve_model() is not None,
            "auth_required": _auth_enabled(),
        }

    @app.post("/api/extract")
    async def extract(request: ExtractRequest, _: str = Depends(enforce_rate_limit)) -> JSONResponse:
        text = request.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Input text is empty.")
        if not _has_api_key():
            raise HTTPException(
                status_code=503,
                detail="OPENAI_API_KEY is not configured on the server.",
            )
        model = _resolve_model()
        if not model:
            raise HTTPException(
                status_code=503,
                detail="No model configured. Set OPENAI_API_MODEL on the server.",
            )

        config = ClaimExtractionRuntimeConfig(
            model=model,
            base_url=_env("OPENAI_API_BASE_URL") or None,
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
