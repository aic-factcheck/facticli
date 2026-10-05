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

    @staticmethod
    def _env(**extra: str) -> dict[str, str]:
        base = {
            "OPENAI_API_KEY": "dummy",
            "OPENAI_API_MODEL": "gpt-5.4",
            "FACTICLI_API_KEY": "cedmo_2026",
        }
        base.update(extra)
        return base

    @staticmethod
    def _auth(key: str = "cedmo_2026") -> dict[str, str]:
        return {"Authorization": f"Bearer {key}"}

    def test_health_is_public_and_leaks_no_endpoint_detail(self):
        with patch.dict("os.environ", self._env(), clear=False):
            response = self._client().get("/api/health")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["has_api_key"])
        self.assertTrue(body["model_configured"])
        self.assertTrue(body["auth_required"])
        self.assertNotIn("base_url", body)
        self.assertNotIn("default_model", body)

    def test_extract_requires_api_key(self):
        with patch.dict("os.environ", self._env(), clear=False):
            response = self._client().post("/api/extract", json={"text": "Inflace klesla."})
        self.assertEqual(response.status_code, 401)

    def test_extract_rejects_wrong_api_key(self):
        with patch.dict("os.environ", self._env(), clear=False):
            response = self._client().post(
                "/api/extract", json={"text": "Inflace klesla."}, headers=self._auth("wrong")
            )
        self.assertEqual(response.status_code, 401)

    def test_extract_accepts_x_api_key_header(self):
        fake_service = AsyncMock()
        fake_service.extract_claims = AsyncMock(
            return_value=ClaimExtractionResult(input_text="x", detected_language="cs")
        )
        with (
            patch.dict("os.environ", self._env(), clear=False),
            patch("facticli.web.app.build_claim_extraction_service", return_value=fake_service),
        ):
            response = self._client().post(
                "/api/extract", json={"text": "Inflace klesla."}, headers={"X-API-Key": "cedmo_2026"}
            )
        self.assertEqual(response.status_code, 200)

    def test_api_is_closed_when_gate_key_unset(self):
        """An unconfigured gate must fail closed, never open."""
        with patch.dict(
            "os.environ",
            {"OPENAI_API_KEY": "dummy", "OPENAI_API_MODEL": "gpt-5.4", "FACTICLI_API_KEY": ""},
            clear=False,
        ):
            response = self._client().post("/api/extract", json={"text": "Inflace klesla."})
        self.assertEqual(response.status_code, 503)

    def test_client_cannot_override_model_or_base_url(self):
        """Redirecting the provider endpoint would leak the server credential."""
        captured = {}

        def fake_builder(config):
            captured["model"] = config.model
            captured["base_url"] = config.base_url
            service = AsyncMock()
            service.extract_claims = AsyncMock(
                return_value=ClaimExtractionResult(input_text="x", detected_language="cs")
            )
            return service

        with (
            patch.dict(
                "os.environ",
                self._env(OPENAI_API_BASE_URL="https://api.openai.com/v1"),
                clear=False,
            ),
            patch("facticli.web.app.build_claim_extraction_service", fake_builder),
        ):
            response = self._client().post(
                "/api/extract",
                json={"text": "Inflace klesla.", "model": "evil", "base_url": "https://attacker.example"},
                headers=self._auth(),
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(captured["model"], "gpt-5.4")
        self.assertEqual(captured["base_url"], "https://api.openai.com/v1")

    def test_extract_rejects_oversized_input(self):
        with patch.dict("os.environ", self._env(), clear=False):
            response = self._client().post(
                "/api/extract", json={"text": "x" * 20_001}, headers=self._auth()
            )
        self.assertEqual(response.status_code, 422)

    def test_rate_limit_returns_429(self):
        fake_service = AsyncMock()
        fake_service.extract_claims = AsyncMock(
            return_value=ClaimExtractionResult(input_text="x", detected_language="cs")
        )
        with (
            patch.dict("os.environ", self._env(FACTICLI_RATE_LIMIT_REQUESTS="2"), clear=False),
            patch("facticli.web.app.build_claim_extraction_service", return_value=fake_service),
        ):
            client = self._client()
            codes = [
                client.post("/api/extract", json={"text": "a"}, headers=self._auth()).status_code
                for _ in range(3)
            ]
        self.assertEqual(codes, [200, 200, 429])

    def test_extract_rejects_empty_text(self):
        with patch.dict("os.environ", self._env(), clear=False):
            response = self._client().post(
                "/api/extract", json={"text": "   "}, headers=self._auth()
            )
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
            patch.dict("os.environ", self._env(), clear=False),
            patch("facticli.web.app.build_claim_extraction_service", return_value=fake_service),
        ):
            response = self._client().post(
                "/api/extract",
                json={"text": "Inflace klesla pod 3 %.", "max_claims": 5},
                headers=self._auth(),
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
