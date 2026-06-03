from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

fastapi = None
try:  # the web GUI is an optional extra; skip these tests if it is absent.
    import fastapi  # noqa: F401
    from fastapi.testclient import TestClient
except Exception:  # pragma: no cover - exercised only without the web extra
    fastapi = None

from facticli.core.contracts import CheckworthyClaim, ClaimExtractionResult


@unittest.skipIf(fastapi is None, "fastapi (web extra) is not installed")
class WebAppTests(unittest.TestCase):
    def _client(self) -> "TestClient":
        from facticli.web.app import create_app

        return TestClient(create_app())

    def test_health_reports_config(self):
        with patch.dict(
            "os.environ",
            {"OPENAI_API_KEY": "dummy", "OPENAI_API_MODEL": "gpt-5.4"},
            clear=False,
        ):
            response = self._client().get("/api/health")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["has_api_key"])
        self.assertEqual(body["default_model"], "gpt-5.4")

    def test_extract_rejects_empty_text(self):
        with patch.dict(
            "os.environ",
            {"OPENAI_API_KEY": "dummy", "OPENAI_API_MODEL": "gpt-5.4"},
            clear=False,
        ):
            response = self._client().post("/api/extract", json={"text": "   "})
        self.assertEqual(response.status_code, 400)

    def test_extract_returns_structured_result(self):
        fake_result = ClaimExtractionResult(
            input_text="Inflace klesla pod 3 %.",
            detected_language="cs",
            claims=[
                CheckworthyClaim(
                    claim_id="claim_1",
                    claim_text="Inflace klesla pod 3 %.",
                    source_fragment="Inflace klesla pod 3 %",
                    checkworthy_reason="Cislo lze overit.",
                )
            ],
        )
        fake_service = AsyncMock()
        fake_service.extract_claims = AsyncMock(return_value=fake_result)

        with (
            patch.dict(
                "os.environ",
                {"OPENAI_API_KEY": "dummy", "OPENAI_API_MODEL": "gpt-5.4"},
                clear=False,
            ),
            patch("facticli.web.app.build_claim_extraction_service", return_value=fake_service),
        ):
            response = self._client().post(
                "/api/extract", json={"text": "Inflace klesla pod 3 %.", "max_claims": 5}
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["detected_language"], "cs")
        self.assertEqual(body["claims"][0]["claim_id"], "claim_1")

    def test_index_is_served(self):
        response = self._client().get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])


if __name__ == "__main__":
    unittest.main()
