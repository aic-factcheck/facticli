"""Golden test: everything the model sees in one full run equals what the measured code sent.

``fixtures/model_inputs.json`` was captured from the code that produced the
measurements in TODO.md (before the 2026-11 simplification). Every stage's
user message (byte for byte), instructions (by hash), output schema, tool
schemas, model settings, and turn limit must stay the same, so a refactor
cannot move verdicts. If a change to a prompt or payload is intended,
regenerate the fixture and say so in the commit.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import agents
from agents.agent_output import AgentOutputSchema

from facticli.config import ClaimExtractionRuntimeConfig, FactCheckRuntimeConfig
from facticli.core.contracts import AspectFinding, ClaimExtractionResult, FactCheckReport, InvestigationPlan
from facticli.pipeline import build_claim_extractor, build_fact_checker

FIXTURE = Path(__file__).parent / "fixtures" / "model_inputs.json"

PLAN = {
    "claim": "x",
    "checks": [
        {"aspect_id": "a1", "question": "Q1?", "rationale": "r1", "search_queries": ["s1", "s2"], "acceptance_criteria": ["c1"]},
        {"aspect_id": "a2", "question": "Q2?", "rationale": "r2", "search_queries": ["s3"], "acceptance_criteria": ["c2"]},
    ],
    "assumptions": ["as1"],
}
REPORT = {
    "claim": "x", "verdict": "Refuted", "verdict_confidence": 0.7, "justification": "j [S1] [S9]", "key_points": ["k [S2]"],
    "counter_argument": "ca", "evidence_gaps": ["g"], "findings": [],
    "sources": [{"title": "Extra", "url": "https://judge-only.example.com/z", "snippet": "z", "publisher": None, "published_at": None}],
}
EXTRACT = {
    "input_text": "t", "detected_language": "EN ", "coverage_notes": [], "excluded_nonfactual": [],
    "claims": [{"claim_id": "", "claim_text": "c", "source_fragment": "f", "checkworthy_reason": "r"}],
}


def _finding(aspect_id: str, signal: str) -> dict:
    return {
        "aspect_id": aspect_id, "question": "Q?", "signal": signal, "summary": f"sum {aspect_id}", "confidence": 0.6, "caveats": ["cv"],
        "sources": [
            {"title": "Gov", "url": "https://www.census.gov/x", "snippet": "snip one", "publisher": None, "published_at": "2019-01-01"},
            {"title": "News", "url": f"https://www.reuters.com/{aspect_id}", "snippet": "snip two", "publisher": "Reuters", "published_at": "2019-02-01"},
            {"title": "FC", "url": "https://www.politifact.com/y", "snippet": "blocked", "publisher": None, "published_at": None},
        ],
    }


class ModelInputTests(unittest.TestCase):
    def test_model_inputs_match_the_measured_code(self):
        captures: list[dict] = []

        async def fake_run(agent, input, **kwargs):
            message = json.loads(input)
            settings = dataclasses.asdict(agent.model_settings)
            settings.pop("retry", None)
            captures.append(
                {
                    "agent": agent.name,
                    "raw": input,
                    "instructions_sha256": hashlib.sha256(agent.instructions.encode()).hexdigest(),
                    "output_schema": AgentOutputSchema(agent.output_type).json_schema(),
                    "tools": [
                        {
                            "name": getattr(tool, "name", type(tool).__name__),
                            "schema": getattr(tool, "params_json_schema", None),
                            "desc": getattr(tool, "description", None),
                        }
                        for tool in agent.tools
                    ],
                    "settings": settings,
                    "max_turns": kwargs.get("max_turns"),
                    "model": agent.model,
                }
            )
            if agent.output_type is InvestigationPlan:
                output = InvestigationPlan.model_validate(PLAN)
            elif agent.output_type is AspectFinding:
                aspect_id = message["check"]["aspect_id"]
                output = AspectFinding.model_validate(_finding(aspect_id, "refutes" if aspect_id == "a1" else "supports"))
            elif agent.output_type is FactCheckReport:
                output = FactCheckReport.model_validate(REPORT)
            else:
                output = ClaimExtractionResult.model_validate(EXTRACT)
            return SimpleNamespace(final_output_as=lambda cls, raise_if_incorrect_type=False: output)

        async def run_both():
            checker = build_fact_checker(
                FactCheckRuntimeConfig(
                    search_provider="brave", block_fact_checkers=True, max_checks=4, service_tier="flex", stage_efforts={"judge": "high"}
                )
            )
            run = await checker.check_claim("  The claim text.  ", claim_id=7, claim_date="30-10-2020")
            extraction = await build_claim_extractor(ClaimExtractionRuntimeConfig(max_claims=5)).extract_claims("  some text  ")
            return run, extraction

        env = {"OPENAI_API_KEY": "sk-dummy", "OPENAI_API_MODEL": "gold-model", "BRAVE_SEARCH_API_KEY": "b", "OPENAI_API_BASE_URL": ""}
        with patch.dict("os.environ", env), patch.object(agents.Runner, "run", fake_run):
            run, extraction = asyncio.run(run_both())

        # Same serialization the fixture went through (objects such as Reasoning become strings).
        captures = json.loads(json.dumps(captures, default=str))
        captures.sort(key=lambda capture: (capture["agent"], capture["raw"]))
        expected = sorted(json.loads(FIXTURE.read_text(encoding="utf-8")), key=lambda capture: (capture["agent"], capture["raw"]))
        self.assertEqual(len(captures), len(expected))
        for got, want in zip(captures, expected):
            for key in want:
                self.assertEqual(got[key], want[key], f"{want['agent']}: {key} differs from the measured code")

        # And the harness around the calls did its part.
        self.assertEqual(run.artifacts.unresolved_citations, [9])
        self.assertEqual([source.url for source in run.report.sources][-1], "https://judge-only.example.com/z")
        self.assertTrue(all("politifact" not in source.url for source in run.report.sources))
        self.assertEqual((extraction.detected_language, extraction.claims[0].claim_id), ("en", "claim_1"))


if __name__ == "__main__":
    unittest.main()
