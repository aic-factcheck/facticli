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
        # Worded for the person typing it into the form, not for the header name.
        self.assertIn("access password", response.json()["detail"])

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

    def test_client_cannot_override_base_url(self):
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
                json={"text": "Inflace klesla.", "base_url": "https://attacker.example"},
                headers=self._auth(),
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(captured["model"], "gpt-5.4")
        # base_url is not part of the request schema, so it is ignored outright
        # and the server's own endpoint is used.
        self.assertEqual(captured["base_url"], "https://api.openai.com/v1")

    def test_models_endpoint_is_public_and_lists_allowlist(self):
        with patch.dict("os.environ", self._env(), clear=False):
            response = self._client().get("/api/models")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["default"], "gpt-5.4")
        self.assertIn("gpt-6.1-sol", body["models"])
        self.assertIn(body["default"], body["models"])

    def test_extract_accepts_an_allowlisted_model(self):
        captured = {}

        def fake_builder(config):
            captured["model"] = config.model
            service = AsyncMock()
            service.extract_claims = AsyncMock(
                return_value=ClaimExtractionResult(input_text="x", detected_language="cs")
            )
            return service

        with (
            patch.dict("os.environ", self._env(), clear=False),
            patch("facticli.web.app.build_claim_extraction_service", fake_builder),
        ):
            response = self._client().post(
                "/api/extract",
                json={"text": "Inflace klesla.", "model": "gpt-6-luna"},
                headers=self._auth(),
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(captured["model"], "gpt-6-luna")

    def test_extract_rejects_a_model_outside_the_allowlist(self):
        """Free-text models would let any caller spend credits on anything."""
        with patch.dict("os.environ", self._env(), clear=False):
            response = self._client().post(
                "/api/extract",
                json={"text": "Inflace klesla.", "model": "o3-pro-expensive"},
                headers=self._auth(),
            )
        self.assertEqual(response.status_code, 400)
        self.assertIn("not allowed", response.json()["detail"])

    def test_api_index_negotiates_html_for_browsers(self):
        with patch.dict("os.environ", self._env(), clear=False):
            client = self._client()
            browser = client.get("/api", headers={"Accept": "text/html,application/xhtml+xml"})
            api = client.get("/api", headers={"Accept": "application/json"})
            bare = client.get("/api")
        self.assertIn("text/html", browser.headers["content-type"])
        self.assertIn("Claim Extractor API", browser.text)
        self.assertIn("application/json", api.headers["content-type"])
        self.assertEqual(api.json()["ui"], "/extract")
        # curl and SDKs send */* and must not get a web page back
        self.assertIn("application/json", bare.headers["content-type"])

    def test_get_on_extract_documents_instead_of_405(self):
        with patch.dict("os.environ", self._env(), clear=False):
            client = self._client()
            browser = client.get("/api/extract", headers={"Accept": "text/html"})
            api = client.get("/api/extract", headers={"Accept": "application/json"})
        self.assertEqual(browser.status_code, 200)
        self.assertIn("text/html", browser.headers["content-type"])
        self.assertEqual(api.status_code, 200)
        self.assertEqual(api.json()["method"], "POST")
        self.assertIn("401", api.json()["errors"])

    def test_docs_routes_need_no_password(self):
        """Documentation spends no credits, so it must not be gated."""
        with patch.dict("os.environ", self._env(), clear=False):
            client = self._client()
            for path in ("/api", "/api/extract", "/api/health", "/api/models"):
                self.assertEqual(client.get(path).status_code, 200, path)

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

    def test_extractor_ui_is_served(self):
        response = self._client().get("/extract")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])

    def test_root_redirects_to_extract(self):
        response = self._client().get("/", follow_redirects=False)
        self.assertEqual(response.status_code, 308)
        self.assertEqual(response.headers["location"], "/extract")


if __name__ == "__main__":
    unittest.main()
