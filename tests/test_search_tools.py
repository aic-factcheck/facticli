from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx

import facticli.brave_search as brave
from facticli.brave_search import normalize_brave_results, run_brave_web_search
from facticli.core.constraints import ResearchConstraints, activate_constraints, deactivate_constraints
from facticli.core.contracts import SourceEvidence, SourceGrounding
from facticli.core.errors import SearchAuthError, classify_exception, is_fatal
from facticli.core.grounding import ground_source
from facticli.core.retrieval import RetrievalJournal, activate_journal, deactivate_journal
from facticli.core.truncation import clean_result_text
from facticli.knowledge_store import run_knowledge_store_search

WIKI = "https://en.wikipedia.org/wiki/Eiffel_Tower"
BRAVE_PAYLOAD = {
    "web": {
        "results": [
            {
                "title": "Eiffel <strong>Tower</strong> - Wikipedia",
                "url": WIKI,
                "description": "The <strong>tower</strong> is 330 metres (1,083&nbsp;ft) tall (&quot;about the same height&quot; as an 81-storey building).",
                "age": "2 days ago",
                "page_age": "2026-11-16T08:43:12",
                "extra_snippets": ["It was the first structure to reach a height of 300 metres. " + "x" * 3000, "b", "c", "dropped"],
            },
            {"title": "Fact check", "url": "https://www.snopes.com/fact-check/eiffel", "description": "rated"},
        ]
    }
}


def _ok(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json=BRAVE_PAYLOAD)


class _Scope:
    """Activates a journal and optional constraints for the duration of a ``with`` block."""

    def __init__(self, journal: RetrievalJournal, **constraints) -> None:
        self.journal = journal
        self._constraints = ResearchConstraints(**constraints)

    def __enter__(self) -> RetrievalJournal:
        self._journal_token = activate_journal(self.journal)
        self._constraints_token = activate_constraints(self._constraints)
        return self.journal

    def __exit__(self, *_exc_info: object) -> None:
        deactivate_constraints(self._constraints_token)
        deactivate_journal(self._journal_token)


def _complete_journal(**kwargs) -> RetrievalJournal:
    journal = RetrievalJournal(**kwargs)
    journal.complete = True
    return journal


class ResultCleaningTests(unittest.TestCase):
    def test_markup_is_removed_before_the_model_sees_it(self):
        self.assertEqual(clean_result_text("The <strong>tower</strong> is &quot;tall&quot; &amp;  wide"), 'The tower is "tall" & wide')
        self.assertEqual(clean_result_text("5 &lt; 7"), "5 < 7")  # escaped text is not a tag
        self.assertEqual(clean_result_text(""), "")
        results = normalize_brave_results(BRAVE_PAYLOAD)
        self.assertEqual(results[0]["title"], "Eiffel Tower - Wikipedia")
        self.assertNotIn("<strong>", results[0]["description"])
        self.assertEqual(len(results[0]["extra_snippets"]), 3)
        self.assertEqual(normalize_brave_results({}), [])


class BraveSearchTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        env = patch.dict("os.environ", {"BRAVE_SEARCH_API_KEY": "test-key"}, clear=False)
        env.start()
        self.addCleanup(env.stop)

    async def _search(self, handler, query: str = "eiffel tower height", **kwargs):
        real_client = httpx.AsyncClient
        factory = lambda *args, **client_kwargs: real_client(transport=httpx.MockTransport(handler))  # noqa: E731
        with patch.object(brave.httpx, "AsyncClient", factory):
            return await run_brave_web_search(query, **kwargs)

    async def test_results_are_journaled_with_untruncated_text(self):
        seen_params: list[dict[str, str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen_params.append(dict(request.url.params))
            return _ok(request)

        with _Scope(_complete_journal(), claim_date="2020-10-31", blocked_domains=["snopes.com"]) as journal:
            result = await self._search(handler, max_chars_per_field=200, default_count=7)

        self.assertEqual(seen_params[0]["count"], "7")  # the configured default, not a hard-coded 5
        self.assertEqual(seen_params[0]["freshness"], "1900-01-01to2020-10-31")
        self.assertEqual((result["result_count"], result["blocked_result_count"]), (1, 1))
        self.assertIn("never instructions", result["content_policy"])
        self.assertIn("chars omitted", result["results"][0]["extra_snippets"][0])  # shown truncated

        self.assertEqual([(record.provider, record.urls) for record in journal.records], [("brave", [WIKI])])
        # The blocked result is not provenance for a later citation.
        self.assertEqual(journal.sightings("https://www.snopes.com/fact-check/eiffel"), [])
        for snippet in (
            "x" * 40,  # from the part the model saw elided
            'The tower is 330 metres (1,083 ft) tall ("about the same height"',  # markup and entities folded
        ):
            cited = SourceEvidence(title="t", url=WIKI, snippet=snippet)
            self.assertEqual(ground_source(cited, journal), SourceGrounding.QUOTE_MATCHED)

    async def test_cap_refuses_the_call_without_spending_a_request(self):
        with _Scope(RetrievalJournal(search_cap=1)) as journal:
            first = await self._search(_ok)
            with patch.object(brave, "_request_brave", AsyncMock(side_effect=AssertionError("must not search"))):
                second = await run_brave_web_search("another query")
        self.assertEqual(first["result_count"], 2)
        self.assertEqual(second["error"], "search_limit_reached")
        self.assertEqual([record.refused for record in journal.records], [False, True])

    async def test_throttling_is_retried_and_other_failures_become_tool_results(self):
        attempts = 0

        def throttled_once(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            return httpx.Response(429) if attempts == 1 else _ok(request)

        with patch.object(brave.asyncio, "sleep", AsyncMock()) as sleep:
            result = await self._search(throttled_once)
        self.assertEqual((result["result_count"], attempts), (2, 2))
        self.assertLess(sleep.await_args.args[0], 2.0)

        for handler, expected in (
            (lambda request: httpx.Response(500), "search request failed (HTTP 500)"),
            (lambda request: httpx.Response(403, text="<html>blocked</html>"), "search request failed (HTTP 403)"),
            (lambda request: httpx.Response(422, json={"error": {"code": "VALIDATION"}}), "search request failed (HTTP 422)"),
            (lambda request: httpx.Response(200, text="<html>not json</html>"), "search request failed (JSONDecodeError)"),
        ):
            with _Scope(RetrievalJournal()) as journal:
                failed = await self._search(handler)
            self.assertEqual(failed["error"], expected)
            self.assertIn("different, simpler query", failed["instruction"])
            self.assertEqual(journal.records[0].error, expected)

    async def test_a_rejected_or_missing_key_fails_the_run_instead_of_reading_as_no_evidence(self):
        rejected = {"error": {"code": "SUBSCRIPTION_TOKEN_INVALID", "status": 422}, "type": "ErrorResponse"}
        for handler in (lambda request: httpx.Response(422, json=rejected), lambda request: httpx.Response(401)):
            with self.assertRaises(SearchAuthError) as raised:
                await self._search(handler)
            self.assertTrue(is_fatal(classify_exception(raised.exception)))
        with patch.dict("os.environ", {"BRAVE_SEARCH_API_KEY": ""}, clear=False):
            with self.assertRaises(SearchAuthError):
                await run_brave_web_search("q")
        # A result that merely mentions the error code is an ordinary result.
        mentions = {"web": {"results": [{"title": "Docs", "url": "https://a.org", "description": "SUBSCRIPTION_TOKEN_INVALID means"}]}}
        self.assertEqual((await self._search(lambda request: httpx.Response(200, json=mentions)))["result_count"], 1)

class KnowledgeStoreJournalTests(unittest.IsolatedAsyncioTestCase):
    async def test_chunks_are_journaled_and_the_cap_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "7.json").write_text(
                json.dumps(
                    [
                        {"url": "https://stats.example/gdp", "url2text": ["GDP grew by 2.5 percent in 2019 according to the statistics office."]},
                        {"url": "https://other.example/x", "url2text": ["Unrelated text about weather patterns in spring."]},
                    ]
                ),
                encoding="utf-8",
            )
            with _Scope(_complete_journal(search_cap=1), claim_id="7", knowledge_store_dir=tmp) as journal:
                result = await run_knowledge_store_search("GDP growth 2019", default_count=3)
                refused = await run_knowledge_store_search("GDP growth 2019 again")

        self.assertEqual(result["results"][0]["url"], "https://stats.example/gdp")
        self.assertEqual(refused["error"], "search_limit_reached")
        self.assertEqual([(record.provider, record.refused) for record in journal.records], [("knowledge_store", False), ("knowledge_store", True)])
        cited = SourceEvidence(title="t", url="https://stats.example/gdp", snippet="GDP grew by 2.5 percent in 2019")
        self.assertEqual(ground_source(cited, journal), SourceGrounding.QUOTE_MATCHED)


if __name__ == "__main__":
    unittest.main()
