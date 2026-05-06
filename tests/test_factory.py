from __future__ import annotations

import unittest
from unittest.mock import patch

from facticli.adapters.provider_profile import InferenceConfig
from facticli.application.config import ClaimExtractionRuntimeConfig, FactCheckRuntimeConfig
from facticli.application.factory import build_claim_extraction_service, build_fact_check_service


class FactoryTests(unittest.TestCase):
    def test_build_fact_check_service_uses_compatible_adapters(self):
        planner = object()
        researcher = object()
        judge = object()
        review = object()
        inference_config = InferenceConfig(
            api_key="key",
            model="gpt-5.4",
            base_url=None,
            api_mode="responses",
        )
        with (
            patch("facticli.application.factory.configure_inference_client") as configure,
            patch(
                "facticli.application.factory.load_inference_config",
                return_value=inference_config,
            ) as load,
            patch(
                "facticli.application.factory.CompatiblePlannerAdapter",
                return_value=planner,
            ) as planner_adapter,
            patch("facticli.application.factory.CompatibleResearchAdapter", return_value=researcher),
            patch("facticli.application.factory.CompatibleJudgeAdapter", return_value=judge),
            patch("facticli.application.factory.CompatibleReviewAdapter", return_value=review),
        ):
            service = build_fact_check_service(
                FactCheckRuntimeConfig(
                    model="gpt-5.4",
                )
            )

        load.assert_called_once_with(requested_model="gpt-5.4", base_url=None)
        configure.assert_called_once_with(inference_config)
        planner_adapter.assert_called_once()
        self.assertEqual(planner_adapter.call_args.kwargs["model"], "gpt-5.4")
        self.assertIs(service.plan_stage.planner, planner)
        self.assertIs(service.research_stage.researcher, researcher)
        self.assertIs(service.judge_stage.judge, judge)
        self.assertIs(service.review_stage.reviewer, review)

    def test_build_fact_check_service_passes_openai_compatible_base_url(self):
        inference_config = InferenceConfig(
            api_key="key",
            model="compatible-model",
            base_url="https://compatible.example/v1",
            api_mode="chat_completions",
        )
        with (
            patch("facticli.application.factory.configure_inference_client"),
            patch(
                "facticli.application.factory.load_inference_config",
                return_value=inference_config,
            ) as load,
            patch("facticli.application.factory.CompatiblePlannerAdapter") as planner_adapter,
            patch("facticli.application.factory.CompatibleResearchAdapter"),
            patch("facticli.application.factory.CompatibleJudgeAdapter"),
            patch("facticli.application.factory.CompatibleReviewAdapter"),
        ):
            build_fact_check_service(
                FactCheckRuntimeConfig(
                    model=None,
                    base_url="https://compatible.example/v1",
                )
            )

        load.assert_called_once_with(requested_model=None, base_url="https://compatible.example/v1")
        self.assertEqual(planner_adapter.call_args.kwargs["model"], "compatible-model")

    def test_build_claim_extraction_service_uses_same_compatible_backend(self):
        backend = object()
        inference_config = InferenceConfig(
            api_key="key",
            model="gpt-5.4",
            base_url=None,
            api_mode="responses",
        )
        with (
            patch("facticli.application.factory.configure_inference_client") as configure,
            patch(
                "facticli.application.factory.load_inference_config",
                return_value=inference_config,
            ) as load,
            patch(
                "facticli.application.factory.CompatibleClaimExtractionAdapter",
                return_value=backend,
            ) as backend_adapter,
        ):
            service = build_claim_extraction_service(
                ClaimExtractionRuntimeConfig(model="gpt-5.4")
            )

        load.assert_called_once_with(requested_model="gpt-5.4", base_url=None)
        configure.assert_called_once_with(inference_config)
        backend_adapter.assert_called_once()
        self.assertEqual(backend_adapter.call_args.kwargs["model"], "gpt-5.4")
        self.assertIs(service.extraction_stage.backend, backend)


if __name__ == "__main__":
    unittest.main()
