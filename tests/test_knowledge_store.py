from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from facticli.core.constraints import ResearchConstraints, activate_constraints, deactivate_constraints
from facticli.knowledge_store import run_knowledge_store_search


class KnowledgeStoreSearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_returns_relevant_chunk_for_configured_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            store_path = Path(tmp) / "42.json"
            records = [
                {
                    "url": "https://example.com/eiffel",
                    "url2text": [
                        "The Eiffel Tower was completed in 1889 for the World's Fair.",
                        "It is located on the Champ de Mars in Paris, France.",
                    ],
                },
                {
                    "url": "https://example.com/unrelated",
                    "url2text": ["Bananas are rich in potassium and grow in tropical regions."],
                },
            ]
            store_path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

            token = activate_constraints(
                ResearchConstraints(claim_id="42", knowledge_store_dir=tmp)
            )
            try:
                result = await run_knowledge_store_search("Eiffel Tower completed 1889", count=2)
            finally:
                deactivate_constraints(token)

        self.assertEqual(result["provider"], "knowledge_store")
        self.assertGreaterEqual(result["result_count"], 1)
        self.assertEqual(result["results"][0]["url"], "https://example.com/eiffel")
        self.assertIn("1889", result["results"][0]["description"])

    async def test_search_without_constraints_reports_error(self):
        result = await run_knowledge_store_search("anything")
        self.assertEqual(result["result_count"], 0)
        self.assertIn("error", result)

    async def test_search_with_missing_store_file_reports_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            token = activate_constraints(
                ResearchConstraints(claim_id="404", knowledge_store_dir=tmp)
            )
            try:
                result = await run_knowledge_store_search("anything")
            finally:
                deactivate_constraints(token)
        self.assertIn("not found", result["error"])


if __name__ == "__main__":
    unittest.main()
