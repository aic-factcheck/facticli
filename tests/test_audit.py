from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from facticli.config import FactCheckRuntimeConfig, parse_stage_assignments
from facticli.artifacts_audit import aggregate, audit_run, load_artifacts, main as audit_main
from facticli.cli import build_parser, run_skills_command
from facticli.core.artifacts import ResearchCheckArtifact, RunArtifacts
from facticli.core.constraints import ResearchConstraints
from facticli.core.contracts import (
    AspectFinding,
    EvidenceSignal,
    FactCheckReport,
    FindingStatus,
    InvestigationPlan,
    SourceEvidence,
    SourceGrounding,
    StopReason,
    VeracityVerdict,
    VerificationCheck,
)
from facticli.core.retrieval import RetrievalSummary
from facticli.core.usage import UsageSummary
from facticli.eval.label_metrics import compute_consistency_metrics


class StageAssignmentTests(unittest.TestCase):
    def test_parse_stage_assignments(self):
        models = parse_stage_assignments(["research=gpt-5-mini", "JUDGE=gpt-5.4"], option_name="--stage-model")
        self.assertEqual(models, {"research": "gpt-5-mini", "judge": "gpt-5.4"})
        efforts = parse_stage_assignments(["plan=HIGH"], allowed_values=("low", "medium", "high"), option_name="--stage-effort")
        self.assertEqual(efforts, {"plan": "high"})
        self.assertEqual(parse_stage_assignments(None, option_name="x"), {})
        for bad in (["nope=x"], ["research"], ["research="]):
            with self.assertRaises(ValueError):
                parse_stage_assignments(bad, option_name="--stage-model")
        with self.assertRaises(ValueError):
            parse_stage_assignments(["plan=extreme"], allowed_values=("low",), option_name="--stage-effort")

    def test_config_routing_helpers(self):
        config = FactCheckRuntimeConfig(stage_models={"judge": "big"}, stage_efforts={"research": "low"})
        self.assertEqual(config.model_for("judge", "default"), "big")
        self.assertEqual(config.model_for("plan", "default"), "default")
        self.assertEqual(config.effort_for("research"), "low")
        self.assertIsNone(config.effort_for("judge"))


class SkillsCommandTests(unittest.TestCase):
    def test_show_prints_prompt_and_rejects_unknown(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = run_skills_command(show="judge")
        self.assertEqual(code, 0)
        self.assertIn("# judge:", stdout.getvalue())
        self.assertIn("# Output contract", stdout.getvalue())
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            self.assertEqual(run_skills_command(show="nope"), 2)

    def test_parser_accepts_new_flags(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "check", "claim", "--stage-model", "judge=x", "--stage-effort", "research=low", "--block-fact-checkers",
                "--blocked-domain", "a.b", "--show-usage", "--model-retries", "1", "--artifacts-dir", "/tmp/x",
            ]
        )
        self.assertEqual(args.stage_model, ["judge=x"])
        self.assertEqual(args.blocked_domains, ["a.b"])
        self.assertEqual(args.model_retries, 1)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["check", "claim", "--max-checks", "0"])


def _artifact(claim_id: str, verdict: VeracityVerdict, urls: list[str], *, claim_date: str | None = "2020-10-31", removed: int = 0, failed: bool = False) -> RunArtifacts:
    sources = [SourceEvidence(title=u, url=u, snippet="s", published_at="2020-01-01") for u in urls]
    finding = AspectFinding(
        aspect_id="a", question="q", signal=EvidenceSignal.SUPPORTS, summary="s", confidence=0.8, sources=sources,
        status=FindingStatus.FAILED if failed else FindingStatus.COMPLETED,
    )
    check = VerificationCheck(aspect_id="a", question="q", rationale="r", search_queries=["q"])
    report = FactCheckReport(claim="c", verdict=verdict, verdict_confidence=0.8, justification="j", findings=[finding], sources=sources)
    artifacts = RunArtifacts(
        claim="c", normalized_claim="c", claim_id=claim_id,
        constraints=ResearchConstraints(claim_id=claim_id, claim_date=claim_date),
        plan_normalized=InvestigationPlan(claim="c", checks=[check]),
        research_checks=[
            ResearchCheckArtifact(
                check=check, attempts=1, finding=finding,
                removed_sources=[SourceEvidence(title="fc", url="https://www.politifact.com/x", snippet="s")] * removed,
            )
        ],
        report_final=report,
        usage_summary=UsageSummary(total_tokens=1000, input_tokens=800, cached_input_tokens=400, cache_hit_rate=0.5),
        duration_seconds=3.0,
    )
    return artifacts


class ArtifactsAuditTests(unittest.TestCase):
    def test_audit_run_and_aggregate_with_gold(self):
        clean = _artifact("0", VeracityVerdict.SUPPORTED, ["https://www.reuters.com/world/x", "https://www.czso.cz/x"])
        leaked = _artifact("1", VeracityVerdict.REFUTED, ["https://www.snopes.com/fact-check/x"], removed=1)
        late = _artifact("2", VeracityVerdict.SUPPORTED, ["https://example.org/x"], failed=True)
        late.report_final.sources[0].published_at = "2021-05-05"

        rows = [audit_run(a) for a in (clean, leaked, late)]
        self.assertFalse(rows[0]["leaked"])
        self.assertTrue(rows[1]["leaked"])
        self.assertEqual(rows[1]["fact_checker_sources_in_final"], 1)
        self.assertEqual(rows[1]["removed_sources"], 1)
        self.assertEqual(rows[2]["post_claim_date_sources_in_final"], 1)
        self.assertEqual(rows[2]["failed_checks"], 1)
        self.assertEqual(rows[0]["tier_counts"], {"news": 1, "primary": 1})

        gold = [{"label": "Supported"}, {"label": "Supported"}, {"label": "Supported"}]
        summary = aggregate(rows, gold)
        self.assertEqual(summary["runs"], 3)
        self.assertEqual(summary["runs_leaked"], 2)
        self.assertEqual(summary["accuracy_split"]["clean"], {"n": 1, "accuracy": 1.0})
        self.assertEqual(summary["accuracy_split"]["leaked"]["n"], 2)
        self.assertEqual(summary["accuracy_split"]["leaked"]["accuracy"], 0.5)
        self.assertEqual(summary["mean_cache_hit_rate"], 0.5)
        # Artifacts written before the retrieval journal existed carry no provenance.
        self.assertEqual(summary["grounding_counts"], {})

    def test_audit_reports_provenance_limits_and_wasted_tokens(self):
        run = _artifact("0", VeracityVerdict.SUPPORTED, ["https://www.reuters.com/world/x", "https://www.czso.cz/x", "https://example.org/y"])
        levels = [SourceGrounding.QUOTE_MATCHED, SourceGrounding.URL_MATCHED, SourceGrounding.UNMATCHED]
        for source, level in zip(run.report_final.sources, levels):
            source.grounding = level
        run.research_checks[0].finding.stop_reason = StopReason.TURN_LIMIT
        run.usage_summary = UsageSummary(total_tokens=1000, failed_attempt_tokens=250)
        run.retrieval_summary = RetrievalSummary(searches=5, cache_hits=2)
        run.unresolved_citations = [4]

        summary = aggregate([audit_run(run), audit_run(_artifact("1", VeracityVerdict.REFUTED, ["https://www.bbc.com/x"]))], None)
        self.assertEqual(summary["grounding_counts"], {"quote_matched": 1, "url_matched": 1, "unmatched": 1})
        self.assertEqual((summary["runs_with_unmatched_source"], summary["runs_with_unresolved_citation"]), (1, 1))
        self.assertEqual(summary["checks_cut_short"], {"turn_limit": 1})
        self.assertEqual((summary["mean_failed_attempt_tokens"], summary["mean_searches"]), (125.0, 5.0))

    def test_cli_end_to_end_over_artifact_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            for index in range(2):
                artifacts = _artifact(str(index), VeracityVerdict.SUPPORTED, ["https://www.reuters.com/world/x"])
                (Path(tmp) / f"run_2026_{index}_x.json").write_text(artifacts.model_dump_json(), encoding="utf-8")
            (Path(tmp) / "run_bad.json").write_text("{not json", encoding="utf-8")
            output = Path(tmp) / "audit.json"
            stdout, stderr = io.StringIO(), io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = audit_main(["--artifacts-dir", tmp, "--output", str(output)])
            self.assertEqual(code, 0)
            self.assertIn("runs: 2", stdout.getvalue())
            self.assertIn("provenance: not assessed", stdout.getvalue())
            self.assertIn("skipping run_bad.json", stderr.getvalue())
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["summary"]["runs_leaked"], 0)
            self.assertEqual(len(load_artifacts(Path(tmp), limit=1)), 1)


class ConsistencyMetricTests(unittest.TestCase):
    def test_pass_hat_k_and_agreement(self):
        gold = ["Supported", "Refuted", "Refuted", "Not Enough Evidence"]
        runs = [
            ["Supported", "Refuted", "Supported", "Refuted"],
            ["Supported", "Refuted", "Refuted", "Refuted"],
            ["Supported", "Supported", "Refuted", "Not Enough Evidence"],
        ]
        metrics = compute_consistency_metrics(gold, runs)
        self.assertEqual(metrics["k"], 3)
        self.assertEqual(metrics["pass_hat_k"], 0.25)  # only claim 0 right everywhere
        self.assertEqual(metrics["pass_at_k"], 1.0)
        self.assertEqual(metrics["agreement_rate"], 0.25)
        self.assertEqual(metrics["majority_vote_accuracy"], 0.75)
        self.assertEqual(metrics["per_run_accuracy"], [0.5, 0.75, 0.75])
        with self.assertRaises(ValueError):
            compute_consistency_metrics(gold, [["Supported"]])
        with self.assertRaises(ValueError):
            compute_consistency_metrics(gold, [])


if __name__ == "__main__":
    unittest.main()
