from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from facticli.application.repository import FileRunArtifactRepository
from facticli.application.stages import _apply_source_constraints
from facticli.core.artifacts import ResearchCheckArtifact, RunArtifacts
from facticli.core.constraints import (
    ResearchConstraints,
    activate_constraints,
    deactivate_constraints,
    is_blocked_url,
    normalize_claim_date,
    parse_date_loose,
    unwrap_archive_url,
    url_domain,
    violates_date_cutoff,
)
from facticli.core.contracts import AspectFinding, EvidenceSignal, SourceEvidence, VerificationCheck
from facticli.core.usage import activate_usage_log, apply_usage, deactivate_usage_log, summarize_usage, track_usage


class ConstraintTests(unittest.TestCase):
    def test_parse_date_loose_supports_iso_and_averitec_formats(self):
        self.assertEqual(parse_date_loose("2020-10-31").isoformat(), "2020-10-31")
        self.assertEqual(parse_date_loose("31-10-2020").isoformat(), "2020-10-31")
        self.assertIsNone(parse_date_loose("October 2020"))
        self.assertIsNone(parse_date_loose(None))
        self.assertIsNone(parse_date_loose("25-8-20"))  # not the year 20

    def test_normalize_claim_date(self):
        self.assertEqual(normalize_claim_date("25-8-2020"), "2020-08-25")
        self.assertIsNone(normalize_claim_date("unknown"))

    def test_is_blocked_url_matches_domains_subdomains_and_paths(self):
        blocked = ["politifact.com", "reuters.com/fact-check"]
        self.assertTrue(is_blocked_url("https://www.politifact.com/article/1", blocked))
        self.assertTrue(is_blocked_url("https://sub.politifact.com/x", blocked))
        self.assertTrue(is_blocked_url("https://www.reuters.com/fact-check/some-claim", blocked))
        self.assertFalse(is_blocked_url("https://www.reuters.com/world/some-news", blocked))
        self.assertFalse(is_blocked_url("https://notpolitifact.com/x", blocked))

    def test_blocklist_sees_through_ports_dots_and_archive_wrappers(self):
        blocked = ["snopes.com", "reuters.com/fact-check"]
        for url in (
            "https://www.snopes.com:443/fact-check/x",
            "https://snopes.com./fact-check/x",
            "https://web.archive.org/web/20200101000000/https://www.snopes.com/fact-check/x",
            "https://web.archive.org/web/20200101id_/http://reuters.com/fact-check/abc",
            "https://web.archive.org/web/https://snopes.com/x",  # no timestamp
            "https://web.archive.org:443/web/2/https%3A%2F%2Fsnopes.com%2Fx",  # port, short timestamp, encoded
            "https://web.archive.org/web/2020/https://web.archive.org/web/2019/https://snopes.com/x",  # wrapped twice
        ):
            self.assertTrue(is_blocked_url(url, blocked), url)
        self.assertFalse(is_blocked_url("https://web.archive.org/web/20200101000000/https://www.reuters.com/world/x", blocked))
        self.assertEqual(url_domain("https://User@WWW.Example.org:8443/x"), "example.org")
        self.assertEqual(unwrap_archive_url("https://web.archive.org/web/2020/https:/example.org/a"), "https://example.org/a")
        self.assertEqual(unwrap_archive_url("https://example.org/a"), "https://example.org/a")

    def test_violates_date_cutoff(self):
        self.assertTrue(violates_date_cutoff("2021-01-05", "2020-10-31"))
        self.assertFalse(violates_date_cutoff("2020-10-30", "2020-10-31"))
        self.assertFalse(violates_date_cutoff(None, "2020-10-31"))
        self.assertFalse(violates_date_cutoff("2021-01-05", None))


class SourceFilterTests(unittest.TestCase):
    def _finding(self, urls_and_dates: list[tuple[str, str | None]]) -> AspectFinding:
        return AspectFinding(
            aspect_id="check_1",
            question="Q?",
            signal=EvidenceSignal.SUPPORTS,
            summary="S.",
            confidence=0.8,
            sources=[
                SourceEvidence(title=url, url=url, snippet="snippet", published_at=published_at)
                for url, published_at in urls_and_dates
            ],
            caveats=[],
        )

    def _artifact(self) -> ResearchCheckArtifact:
        return ResearchCheckArtifact(
            check=VerificationCheck(
                aspect_id="check_1", question="Q?", rationale="R", search_queries=["q"]
            )
        )

    def test_filter_removes_blocked_and_late_sources_and_records_them(self):
        constraints = ResearchConstraints(
            claim_date="2020-10-31",
            blocked_domains=["politifact.com"],
        )
        token = activate_constraints(constraints)
        try:
            artifact = self._artifact()
            finding = self._finding(
                [
                    ("https://example.com/a", "2020-01-01"),
                    ("https://politifact.com/b", "2020-01-01"),
                    ("https://example.com/c", "2021-06-01"),
                ]
            )
            filtered = _apply_source_constraints(finding, artifact)
        finally:
            deactivate_constraints(token)

        self.assertEqual([source.url for source in filtered.sources], ["https://example.com/a"])
        self.assertEqual(len(artifact.removed_sources), 2)
        self.assertTrue(any("removed by retrieval constraints" in c for c in filtered.caveats))
        # Raw finding object is untouched for auditability.
        self.assertEqual(len(finding.sources), 3)

    def test_filter_is_noop_without_active_constraints(self):
        artifact = self._artifact()
        finding = self._finding([("https://politifact.com/b", None)])
        filtered = _apply_source_constraints(finding, artifact)
        self.assertEqual(len(filtered.sources), 1)
        self.assertEqual(artifact.removed_sources, [])


class UsageTrackingTests(unittest.TestCase):
    def test_record_and_summarize_usage(self):
        log, token = activate_usage_log()
        try:
            with track_usage(stage="plan", model="m") as event:
                apply_usage(event, SimpleNamespace(requests=1, input_tokens=100, output_tokens=20, total_tokens=120))
            with track_usage(stage="research", model="m") as event:
                apply_usage(event, SimpleNamespace(requests=2, input_tokens=300, output_tokens=50, total_tokens=350))
        finally:
            deactivate_usage_log(token)

        summary = summarize_usage(log.events)
        self.assertEqual(summary.requests, 3)
        self.assertEqual(summary.total_tokens, 470)
        self.assertEqual(summary.per_stage["plan"]["calls"], 1)
        self.assertEqual(summary.per_stage["research"]["input_tokens"], 300)

    def test_tracking_outside_an_active_log_is_a_noop(self):
        with track_usage(stage="plan", model="m") as event:
            self.assertIsNone(event)


class FileRepositoryTests(unittest.TestCase):
    def test_save_writes_one_json_file_per_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            repository = FileRunArtifactRepository(output_dir=tmp)
            artifacts = RunArtifacts(claim="C", normalized_claim="C", claim_id="7")
            repository.save(artifacts)
            files = list(Path(tmp).glob("run_*_7_*.json"))
            self.assertEqual(len(files), 1)
            payload = json.loads(files[0].read_text(encoding="utf-8"))
            self.assertEqual(payload["claim_id"], "7")


if __name__ == "__main__":
    unittest.main()
