from __future__ import annotations

import unittest
from unittest.mock import patch

from facticli.application.config import ClaimExtractionRuntimeConfig, FactCheckRuntimeConfig
from facticli.application.factory import build_claim_extraction_service, build_fact_check_service


class FactoryTests(unittest.TestCase):
    def test_build_fact_check_service_openai_uses_openai_strategies(self):
        planner = object()
        researcher = object()
        judge = object()
        with (
            patch("facticli.application.factory.OpenAIPlannerAdapter", return_value=planner),
            patch("facticli.application.factory.OpenAIResearchAdapter", return_value=researcher),
            patch("facticli.application.factory.OpenAIJudgeAdapter", return_value=judge),
        ):
            service = build_fact_check_service(
                FactCheckRuntimeConfig(
                    inference_provider="openai-agents",
                    model="gpt-4.1-mini",
                )
            )

        self.assertIs(service.plan_stage.planner, planner)
        self.assertIs(service.research_stage.researcher, researcher)
        self.assertIs(service.judge_stage.judge, judge)

    def test_build_fact_check_service_gemini_requires_brave(self):
        with self.assertRaises(ValueError):
            build_fact_check_service(
                FactCheckRuntimeConfig(
                    inference_provider="gemini",
                    search_provider="openai",
                )
            )

    def test_build_fact_check_service_gemini_uses_retriever_strategy(self):
        planner = object()
        researcher = object()
        judge = object()
        retriever = object()
        with (
            patch("facticli.application.factory.BraveSearchRetriever", return_value=retriever),
            patch("facticli.application.factory.GeminiPlannerAdapter", return_value=planner),
            patch("facticli.application.factory.GeminiResearchAdapter", return_value=researcher),
            patch("facticli.application.factory.GeminiJudgeAdapter", return_value=judge),
        ):
            service = build_fact_check_service(
                FactCheckRuntimeConfig(
                    inference_provider="gemini",
                    search_provider="brave",
                    gemini_model="gemini-2.0-flash",
                )
            )

        self.assertIs(service.plan_stage.planner, planner)
        self.assertIs(service.research_stage.researcher, researcher)
        self.assertIs(service.judge_stage.judge, judge)

    def test_build_claim_extraction_service_selects_provider_backend(self):
        openai_backend = object()
        gemini_backend = object()
        with patch("facticli.application.factory.OpenAIClaimExtractionAdapter", return_value=openai_backend):
            openai_service = build_claim_extraction_service(
                ClaimExtractionRuntimeConfig(inference_provider="openai-agents")
            )
        with patch("facticli.application.factory.GeminiClaimExtractionAdapter", return_value=gemini_backend):
            gemini_service = build_claim_extraction_service(
                ClaimExtractionRuntimeConfig(inference_provider="gemini")
            )

        self.assertIs(openai_service.extraction_stage.backend, openai_backend)
        self.assertIs(gemini_service.extraction_stage.backend, gemini_backend)


if __name__ == "__main__":
    unittest.main()
