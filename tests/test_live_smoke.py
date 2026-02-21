from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path


RUN_LIVE_SMOKE = os.getenv("FACTICLI_RUN_LIVE_SMOKE") == "1"


@unittest.skipUnless(RUN_LIVE_SMOKE, "Set FACTICLI_RUN_LIVE_SMOKE=1 to run live smoke tests.")
class LiveSmokeTests(unittest.TestCase):
    def test_openai_profile_cli_smoke_check(self) -> None:
        if not os.getenv("OPENAI_API_KEY"):
            self.skipTest("OPENAI_API_KEY is not set.")

        repo_root = Path(__file__).resolve().parents[1]
        cmd = [
            sys.executable,
            "-m",
            "facticli",
            "check",
            "--inference-provider",
            "openai",
            "--max-checks",
            "2",
            "--parallel",
            "2",
            "--json",
            "The Eiffel Tower was built in 1889.",
        ]
        completed = subprocess.run(
            cmd,
            cwd=repo_root,
            text=True,
            capture_output=True,
            timeout=240,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"CLI smoke run failed:\nSTDERR:\n{completed.stderr}\nSTDOUT:\n{completed.stdout}",
        )

        payload = json.loads(completed.stdout)
        report = payload["report"]
        self.assertIn(
            report["verdict"],
            {
                "Supported",
                "Refuted",
                "Not Enough Evidence",
                "Conflicting Evidence/Cherrypicking",
            },
        )
        sources = report.get("sources", [])
        self.assertGreaterEqual(len(sources), 1)
        for source in sources:
            self.assertTrue(str(source.get("url", "")).startswith(("http://", "https://")))


if __name__ == "__main__":
    unittest.main()
