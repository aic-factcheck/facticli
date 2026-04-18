from __future__ import annotations

import unittest
from unittest.mock import patch

from facticli.adapters.provider_profile import (
    configure_openai_compatible_client,
    resolve_model_name,
    resolve_provider_profile,
)


class ProviderProfileTests(unittest.TestCase):
    def test_resolve_provider_profile_supports_ollama(self):
        profile = resolve_provider_profile("ollama")
        self.assertEqual(profile.name, "ollama")
        self.assertEqual(profile.api_key_env, "OLLAMA_API_KEY")
        self.assertEqual(profile.model_env, "OLLAMA_MODEL")
        self.assertEqual(profile.base_url_env, "OLLAMA_BASE_URL")

    def test_resolve_model_name_uses_ollama_env_default(self):
        with patch.dict("os.environ", {"OLLAMA_MODEL": "kimi-k2.5"}, clear=True):
            self.assertEqual(resolve_model_name("ollama", None), "kimi-k2.5")

    def test_requested_model_overrides_provider_env_default(self):
        with patch.dict("os.environ", {"OLLAMA_MODEL": "ignored"}, clear=True):
            self.assertEqual(resolve_model_name("ollama", "qwen2.5:14b"), "qwen2.5:14b")

    def test_configure_client_uses_provider_specific_base_url_env(self):
        with (
            patch.dict(
                "os.environ",
                {
                    "OLLAMA_API_KEY": "a",
                    "OLLAMA_BASE_URL": "https://llm.ai.e-infra.cz/v1",
                },
                clear=True,
            ),
            patch("facticli.adapters.provider_profile.AsyncOpenAI") as async_openai,
            patch("facticli.adapters.provider_profile.set_default_openai_client"),
            patch("facticli.adapters.provider_profile.set_default_openai_api") as set_api_mode,
            patch("facticli.adapters.provider_profile.set_tracing_disabled"),
        ):
            configure_openai_compatible_client(inference_provider="ollama")

        async_openai.assert_called_once_with(
            api_key="a",
            base_url="https://llm.ai.e-infra.cz/v1",
        )
        set_api_mode.assert_called_once_with("chat_completions")


if __name__ == "__main__":
    unittest.main()
