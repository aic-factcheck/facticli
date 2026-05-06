from __future__ import annotations

import unittest
from unittest.mock import patch

from facticli.adapters.provider_profile import (
    InferenceConfig,
    configure_inference_client,
    infer_api_mode,
    load_inference_config,
)


class InferenceConfigTests(unittest.TestCase):
    def test_load_inference_config_uses_openai_compatible_env(self):
        with patch.dict(
            "os.environ",
            {
                "OPENAI_API_KEY": "test-key",
                "OPENAI_API_MODEL": "gpt-5.4",
                "OPENAI_API_BASE_URL": "https://api.openai.com/v1",
            },
            clear=True,
        ):
            config = load_inference_config(
                requested_model=None,
                base_url=None,
            )
        self.assertEqual(config.api_key, "test-key")
        self.assertEqual(config.model, "gpt-5.4")
        self.assertEqual(config.base_url, "https://api.openai.com/v1")
        self.assertEqual(config.api_mode, "responses")

    def test_requested_model_overrides_env_model(self):
        with patch.dict(
            "os.environ",
            {"OPENAI_API_KEY": "key", "OPENAI_API_MODEL": "env-model"},
            clear=True,
        ):
            config = load_inference_config(
                requested_model="custom-model",
                base_url=None,
            )
        self.assertEqual(config.model, "custom-model")

    def test_base_url_override_via_cli_arg(self):
        with patch.dict(
            "os.environ",
            {
                "OPENAI_API_KEY": "key",
                "OPENAI_API_MODEL": "model",
                "OPENAI_API_BASE_URL": "https://env.url/v1",
            },
            clear=True,
        ):
            config = load_inference_config(
                requested_model=None,
                base_url="https://cli.override/v1",
            )
        self.assertEqual(config.base_url, "https://cli.override/v1")

    def test_missing_api_key_raises_runtime_error(self):
        with (
            patch.dict("os.environ", {"OPENAI_API_MODEL": "model"}, clear=True),
            self.assertRaises(RuntimeError),
        ):
            load_inference_config(
                requested_model=None,
                base_url=None,
            )

    def test_missing_model_raises_runtime_error(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "key"}, clear=True):
            with self.assertRaises(RuntimeError):
                load_inference_config(
                    requested_model=None,
                    base_url=None,
                )

    def test_infer_api_mode_prefers_responses_for_openai_endpoint(self):
        self.assertEqual(infer_api_mode(None), "responses")
        self.assertEqual(infer_api_mode("https://api.openai.com/v1"), "responses")

    def test_infer_api_mode_uses_chat_completions_for_compatible_endpoints(self):
        self.assertEqual(
            infer_api_mode("https://generativelanguage.googleapis.com/v1beta/openai/"),
            "chat_completions",
        )
        self.assertEqual(infer_api_mode("https://llm.ai.e-infra.cz/v1"), "chat_completions")

    def test_configure_inference_client_creates_async_openai_client(self):
        config = InferenceConfig(
            api_key="test-key",
            model="test-model",
            base_url=None,
            api_mode="responses",
        )
        with (
            patch("facticli.adapters.provider_profile.AsyncOpenAI") as async_openai,
            patch("facticli.adapters.provider_profile.set_default_openai_client") as set_client,
            patch("facticli.adapters.provider_profile.set_default_openai_api") as set_api_mode,
            patch("facticli.adapters.provider_profile.set_tracing_disabled"),
        ):
            configure_inference_client(config)

        async_openai.assert_called_once_with(api_key="test-key", base_url=None)
        set_client.assert_called_once()
        set_api_mode.assert_called_once_with("responses")


if __name__ == "__main__":
    unittest.main()
