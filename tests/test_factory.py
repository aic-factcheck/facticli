from __future__ import annotations

import unittest
from unittest.mock import patch

from facticli.application.config import ClaimExtractionRuntimeConfig, FactCheckRuntimeConfig
from facticli.application.factory import build_claim_extraction_service, build_fact_check_service


class FactoryTests(unittest.TestCase):
    def test_build_fact_check_service_uses_compatible_adapters(self):
        planner = object()
        researcher = object()
        judge = object()
        with (
            patch("facticli.application.factory.configure_openai_compatible_client") as configure,
            patch("facticli.application.factory.CompatiblePlannerAdapter", return_value=planner),
            patch("facticli.application.factory.CompatibleResearchAdapter", return_value=researcher),
            patch("facticli.application.factory.CompatibleJudgeAdapter", return_value=judge),
        ):
            service = build_fact_check_service(
                FactCheckRuntimeConfig(
                    inference_provider="openai",
                    model="gpt-4.1-mini",
                )
            )

        configure.assert_called_once_with(inference_provider="openai", base_url=None)
        self.assertIs(service.plan_stage.planner, planner)
        self.assertIs(service.research_stage.researcher, researcher)
        self.assertIs(service.judge_stage.judge, judge)

    def test_build_fact_check_service_bootstraps_gemini_profile_with_same_codepath(self):
        with (
            patch("facticli.application.factory.configure_openai_compatible_client") as configure,
            patch("facticli.application.factory.CompatiblePlannerAdapter"),
            patch("facticli.application.factory.CompatibleResearchAdapter"),
            patch("facticli.application.factory.CompatibleJudgeAdapter"),
        ):
            build_fact_check_service(
                FactCheckRuntimeConfig(
                    inference_provider="gemini",
                    model="gemini-2.0-flash",
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                )
            )

        configure.assert_called_once_with(
            inference_provider="gemini",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )

    def test_build_claim_extraction_service_uses_same_compatible_backend(self):
        backend = object()
        with (
            patch("facticli.application.factory.configure_openai_compatible_client") as configure,
            patch("facticli.application.factory.CompatibleClaimExtractionAdapter", return_value=backend),
        ):
            service = build_claim_extraction_service(
                ClaimExtractionRuntimeConfig(inference_provider="openai", model="gpt-4.1-mini")
            )

        configure.assert_called_once_with(inference_provider="openai", base_url=None)
        self.assertIs(service.extraction_stage.backend, backend)


if __name__ == "__main__":
    unittest.main()
