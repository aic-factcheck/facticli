"""Web GUI for the facticli claim-extraction workflow.

Exposes a small FastAPI application that serves a branded single-page frontend
and a JSON ``POST /api/extract`` endpoint backed by the same
``ClaimExtractionService`` used by the CLI.
"""
from __future__ import annotations

from .app import create_app

__all__ = ["create_app"]
