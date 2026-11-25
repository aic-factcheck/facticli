from __future__ import annotations

import unittest
from unittest.mock import patch

from facticli.config import REASONING_EFFORTS, parse_stage_assignments
from facticli.core.constraints import ResearchConstraints
from facticli.core.contracts import (
    AspectFinding,
    EvidenceSignal,
    FindingStatus,
    InvestigationPlan,
    SourceEvidence,
    SourceGrounding,
    SourceTier,
    StopReason,
    VerificationCheck,
)
from facticli.llm import Endpoint, build_model_settings, configure_client, infer_api_mode, resolve_endpoint
from facticli.payloads import (
    FAILED_CHECK_NOTICE,
    dated,
    extraction_payload,
    index_sources,
    judge_payload,
    plan_payload,
    research_payload,
)


def _check(aspect_id: str, criteria: list[str] | None = None) -> VerificationCheck:
    return VerificationCheck(
        aspect_id=aspect_id,
        question=f"Question {aspect_id}?",
        rationale="r",
        search_queries=["q"],
        acceptance_criteria=criteria or [],
    )


def _source(url: str, tier: SourceTier | None = None) -> SourceEvidence:
    return SourceEvidence(title=url, url=url, snippet=f"snippet from {url}", published_at="2020-01-01", tier=tier)


class PayloadTests(unittest.TestCase):
    def test_judge_payload_groups_by_signal_and_indexes_sources(self):
        plan = InvestigationPlan(claim="c", checks=[_check("a", ["done when x"]), _check("b"), _check("c")])
        shared = _source("https://www.bbc.com/news/1?utm_source=x", SourceTier.NEWS)
        findings = [
            AspectFinding(
                aspect_id="a", question="Qa", signal=EvidenceSignal.SUPPORTS, summary="sa", confidence=0.9,
                sources=[shared, _source("https://data.gov/x", SourceTier.PRIMARY)],
            ),
            AspectFinding(
                aspect_id="b", question="Qb", signal=EvidenceSignal.REFUTES, summary="sb", confidence=0.6,
                sources=[SourceEvidence(title="dup", url="https://www.bbc.com/news/1/", snippet="same page", tier=SourceTier.NEWS)],
            ),
            AspectFinding(
                aspect_id="c", question="Qc", signal=EvidenceSignal.INSUFFICIENT, summary="failed", confidence=0.0,
                status=FindingStatus.FAILED, failure_reason="timeout: TimeoutError",
            ),
        ]
        payload = judge_payload(claim="c", plan=plan, findings=findings)

        self.assertEqual(payload["evidence_overview"]["checks_completed"], 2)
        self.assertEqual(payload["evidence_overview"]["checks_failed"], 1)
        self.assertEqual(payload["evidence_overview"]["signal_counts"]["supports"], 1)
        self.assertEqual(payload["evidence_overview"]["signal_counts"]["refutes"], 1)
        self.assertEqual(payload["evidence_overview"]["signal_counts"]["insufficient"], 0)
        self.assertEqual(payload["evidence_overview"]["source_tier_counts"], {"news": 1, "primary": 1})
        # Shared source deduplicated across findings with a stable id and cited_by both.
        self.assertEqual([row["id"] for row in payload["sources"]], ["S1", "S2"])
        self.assertEqual(payload["sources"][0]["cited_by"], ["a", "b"])
        self.assertEqual(payload["findings_by_signal"]["supports"][0]["source_ids"], ["S1", "S2"])
        self.assertEqual(payload["findings_by_signal"]["supports"][0]["acceptance_criteria"], ["done when x"])
        self.assertEqual(payload["findings_by_signal"]["refutes"][0]["source_ids"], ["S1"])
        self.assertEqual(payload["failed_checks"][0]["aspect_id"], "c")
        self.assertEqual(payload["failed_checks_notice"], FAILED_CHECK_NOTICE)
        self.assertIn("[S2]", payload["citation_rule"])

    def test_judge_payload_without_failures_omits_notice(self):
        plan = InvestigationPlan(claim="c", checks=[_check("a")])
        findings = [AspectFinding(aspect_id="a", question="Q", signal=EvidenceSignal.MIXED, summary="s", confidence=0.5)]
        payload = judge_payload(claim="c", plan=plan, findings=findings)
        self.assertNotIn("failed_checks", payload)
        self.assertEqual(index_sources(findings), ([], {}))

    def test_research_payload_carries_constraints_and_the_search_budget(self):
        constraints = ResearchConstraints(claim_date="2020-10-31", blocked_domains=["politifact.com"])
        payload = research_payload(claim="c", check=_check("a"), search_provider="brave", constraints=constraints, tool_call_budget=6)
        self.assertEqual(list(payload), ["claim", "check", "requirements", "constraints"])
        self.assertEqual(payload["constraints"]["evidence_cutoff_date"], "2020-10-31")
        self.assertEqual(payload["requirements"]["tool_call_budget"], 6)
        self.assertIn("acceptance_criteria", payload["check"])

        bare = research_payload(claim="c", check=_check("a"), search_provider="openai", constraints=None, tool_call_budget=8)
        self.assertNotIn("constraints", bare)

    def test_plan_and_extraction_payloads(self):
        self.assertEqual(plan_payload(claim="c", max_checks=3)["requirements"]["max_checks"], 3)
        extraction = extraction_payload(input_text="t", max_claims=7)
        self.assertEqual(list(extraction), ["input_text", "requirements"])
        self.assertTrue(extraction["requirements"]["preserve_original_diacritics"])

    def test_models_are_told_only_about_untraceable_sources_and_cut_short_research(self):
        traced = SourceEvidence(title="a", url="https://data.gov/a", snippet="q", grounding=SourceGrounding.QUOTE_MATCHED)
        paraphrased = SourceEvidence(title="b", url="https://data.gov/b", snippet="p", grounding=SourceGrounding.URL_MATCHED)
        invented = SourceEvidence(title="c", url="https://x.com/u/1", snippet="i", grounding=SourceGrounding.UNMATCHED)
        findings = [
            AspectFinding(aspect_id="a", question="Qa", signal=EvidenceSignal.SUPPORTS, summary="s", confidence=0.9, sources=[traced, paraphrased]),
            AspectFinding(
                aspect_id="b", question="Qb", signal=EvidenceSignal.REFUTES, summary="s", confidence=0.5, sources=[invented],
                stop_reason=StopReason.TURN_LIMIT,
            ),
        ]
        plan = InvestigationPlan(claim="c", checks=[_check("a"), _check("b")])

        judge = judge_payload(claim="c", plan=plan, findings=findings)
        self.assertEqual([row.get("grounding") for row in judge["sources"]], [None, None, "unmatched"])
        self.assertEqual(judge["findings_by_signal"]["refutes"][0]["stop_reason"], "turn_limit")
        self.assertNotIn("stop_reason", judge["findings_by_signal"]["supports"][0])

    def test_reference_date_leads_the_payload_when_the_run_has_one(self):
        payload = {"claim": "c"}
        self.assertIs(dated(payload, None), payload)
        self.assertIs(dated(payload, ResearchConstraints()), payload)
        self.assertEqual(
            list(dated(payload, ResearchConstraints(reference_date="2020-10-31")).items()),
            [("reference_date", "2020-10-31"), ("claim", "c")],
        )


class ModelSettingsTests(unittest.TestCase):
    def test_effort_vocabulary_covers_current_reasoning_models(self):
        parsed = parse_stage_assignments(["judge=XHIGH", "research=none"], allowed_values=REASONING_EFFORTS, option_name="--stage-effort")
        self.assertEqual(parsed, {"judge": "xhigh", "research": "none"})
        with self.assertRaises(ValueError):
            parse_stage_assignments(["judge=ultra"], allowed_values=REASONING_EFFORTS, option_name="--stage-effort")

    def test_build_model_settings(self):
        settings = build_model_settings(parallel_tool_calls=True, reasoning_effort="low", service_tier="flex")
        self.assertTrue(settings.parallel_tool_calls)
        self.assertEqual(settings.reasoning.effort, "low")
        self.assertEqual(settings.extra_args, {"service_tier": "flex"})
        self.assertIsNone(settings.temperature)
        # No SDK retry policy: the OpenAI client retries transient errors itself.
        self.assertIsNone(settings.retry)

        plain = build_model_settings(parallel_tool_calls=False)
        self.assertIsNone(plain.reasoning)
        self.assertIsNone(plain.extra_args)


class EndpointTests(unittest.TestCase):
    def test_resolve_endpoint_uses_openai_compatible_env(self):
        with patch.dict(
            "os.environ",
            {
                "OPENAI_API_KEY": "test-key",
                "OPENAI_API_MODEL": "gpt-5.4",
                "OPENAI_API_BASE_URL": "https://api.openai.com/v1",
            },
            clear=True,
        ):
            config = resolve_endpoint(
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
            config = resolve_endpoint(
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
            config = resolve_endpoint(
                requested_model=None,
                base_url="https://cli.override/v1",
            )
        self.assertEqual(config.base_url, "https://cli.override/v1")

    def test_missing_api_key_raises_runtime_error(self):
        with (
            patch.dict("os.environ", {"OPENAI_API_MODEL": "model"}, clear=True),
            self.assertRaises(RuntimeError),
        ):
            resolve_endpoint(
                requested_model=None,
                base_url=None,
            )

    def test_missing_model_raises_runtime_error(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "key"}, clear=True):
            with self.assertRaises(RuntimeError):
                resolve_endpoint(
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

    def test_configure_client_creates_async_openai_client(self):
        endpoint = Endpoint(api_key="test-key", model="test-model", base_url=None, api_mode="responses")
        with (
            patch("facticli.llm.AsyncOpenAI") as async_openai,
            patch("facticli.llm.set_default_openai_client") as set_client,
            patch("facticli.llm.set_default_openai_api") as set_api_mode,
            patch("facticli.llm.set_tracing_disabled"),
        ):
            configure_client(endpoint, max_retries=5)

        # Transient errors are retried by the client itself.
        async_openai.assert_called_once_with(api_key="test-key", base_url=None, max_retries=5)
        set_client.assert_called_once()
        set_api_mode.assert_called_once_with("responses")



if __name__ == "__main__":
    unittest.main()
