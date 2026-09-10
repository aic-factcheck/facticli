from __future__ import annotations

import argparse
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from facticli.application.config import FactCheckRuntimeConfig, parse_stage_assignments
from facticli.application.settings_file import apply_settings, find_config_path, load_settings
from facticli.artifacts_audit import aggregate, audit_run, load_artifacts, main as audit_main
from facticli.cli import build_parser, main as cli_main, run_skills_command
from facticli.core.artifacts import ResearchCheckArtifact, RunArtifacts
from facticli.core.constraints import ResearchConstraints
from facticli.core.contracts import (
    AspectFinding,
    EvidenceSignal,
    FactCheckReport,
    FindingStatus,
    InvestigationPlan,
    SourceEvidence,
    VeracityVerdict,
    VerificationCheck,
)
from facticli.core.usage import BudgetStatus, UsageSummary
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


class SettingsFileTests(unittest.TestCase):
    def _write(self, tmp: str, body: str) -> Path:
        path = Path(tmp) / "facticli.toml"
        path.write_text(body, encoding="utf-8")
        return path

    def test_load_settings_merges_defaults_and_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(
                tmp,
                """
                search_provider = "brave"
                [defaults]
                max_checks = 3
                stage_models = { research = "small", judge = "big" }
                blocked_domains = ["a.example", "b.example"]
                [profiles.benchmark]
                max_checks = 6
                block_fact_checkers = true
                stage_efforts = { judge = "high" }
                """,
            )
            base = load_settings(path)
            self.assertEqual(base["search_provider"], "brave")
            self.assertEqual(base["max_checks"], 3)
            self.assertEqual(base["stage_model"], ["research=small", "judge=big"])
            self.assertEqual(base["blocked_domain"], ["a.example", "b.example"])
            self.assertNotIn("block_fact_checkers", base)

            bench = load_settings(path, "benchmark")
            self.assertEqual(bench["max_checks"], 6)
            self.assertTrue(bench["block_fact_checkers"])
            self.assertEqual(bench["stage_effort"], ["judge=high"])
            with self.assertRaises(ValueError):
                load_settings(path, "missing")

    def test_apply_settings_respects_explicit_flags(self):
        parser = build_parser()
        args = parser.parse_args(["check", "claim", "--max-checks", "7"])
        applied = apply_settings(args, {"max_checks": 3, "parallel": 2, "unknown_key": 1, "stage_model": ["judge=x"]}, parser)
        self.assertEqual(args.max_checks, 7)
        self.assertEqual(args.parallel, 2)
        self.assertEqual(args.stage_model, ["judge=x"])
        self.assertEqual(sorted(applied), ["parallel", "stage_model"])

    def test_find_config_path_precedence_and_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            explicit = self._write(tmp, "max_checks = 1\n")
            self.assertEqual(find_config_path(str(explicit)), explicit)
            with self.assertRaises(FileNotFoundError):
                find_config_path(str(Path(tmp) / "nope.toml"))
            with patch.dict("os.environ", {"FACTICLI_CONFIG": str(explicit)}, clear=False):
                self.assertEqual(find_config_path(None), explicit)
            with patch.dict("os.environ", {"FACTICLI_CONFIG": str(Path(tmp) / "gone.toml")}, clear=False):
                with self.assertRaises(FileNotFoundError):
                    find_config_path(None)

    def test_cli_main_reports_config_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, "[profiles.a]\nmax_checks = 2\n")
            stderr = io.StringIO()
            with redirect_stderr(stderr), self.assertRaises(SystemExit) as ctx:
                cli_main(["--config", str(path), "--profile", "zzz", "skills"])
            self.assertEqual(ctx.exception.code, 2)
            self.assertIn("Profile 'zzz' not found", stderr.getvalue())

            stdout = io.StringIO()
            with redirect_stdout(stdout), self.assertRaises(SystemExit) as ctx:
                cli_main(["--config", str(path), "--profile", "a", "skills"])
            self.assertEqual(ctx.exception.code, 0)
            self.assertIn("- judge:", stdout.getvalue())


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
                "check", "claim", "--strategy", "single_agent", "--token-budget", "5000", "--stage-model", "judge=x",
                "--stage-effort", "research=low", "--verify-sources", "--block-fact-checkers", "--blocked-domain", "a.b",
                "--show-usage", "--retry-budget", "0", "--model-retries", "1", "--artifacts-dir", "/tmp/x",
            ]
        )
        self.assertEqual(args.strategy, "single_agent")
        self.assertEqual(args.token_budget, 5000)
        self.assertEqual(args.stage_model, ["judge=x"])
        self.assertTrue(args.verify_sources)
        self.assertEqual(args.blocked_domains, ["a.b"])
        self.assertEqual(args.retry_budget, 0)
        with self.assertRaises(SystemExit):
            parser.parse_args(["check", "claim", "--token-budget", "0"])


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
        budget=BudgetStatus(),
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
