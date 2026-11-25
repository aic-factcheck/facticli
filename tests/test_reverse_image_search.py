"""Reverse image search: leakage controls, journal receipts, cap, cache, and failure handling (no network)."""

from __future__ import annotations

import json
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

import httpx

import facticli.reverse_image_search as ris
from facticli.core.constraints import ResearchConstraints, activate_constraints, deactivate_constraints
from facticli.core.contracts import SourceEvidence, SourceGrounding
from facticli.core.errors import SearchAuthError
from facticli.core.grounding import ground_source
from facticli.core.images import ClaimImage, activate_images, deactivate_images
from facticli.core.retrieval import RetrievalJournal, activate_journal, deactivate_journal
from facticli.core.search_cache import SearchCache
from facticli.llm import build_search_tools

IMAGE = ClaimImage(
    image_id="image_1", source="a.jpg", mime_type="image/jpeg", byte_size=3, sha256="0" * 64,
    public_url="https://host.example/a.jpg", data=b"\xff\xd8\xff",
)
LOCAL_ONLY = IMAGE.model_copy(update={"image_id": "image_2", "public_url": None})

LENS = {
    "organic": [
        {"title": "Old <b>photo</b> story", "link": "https://news.example/2015/story", "source": "News"},
        {"title": "Fact check", "link": "https://www.snopes.com/fact-check/photo", "source": "Snopes"},
        {"title": "Later repost", "link": "https://blog.example/2024/repost", "source": "Blog"},
        {"title": "Social post", "link": "https://www.facebook.com/p/1", "source": "Facebook"},
        {"title": "no link"},
    ]
}
PAGES = {
    "results": [
        {"id": "https://news.example/2015/story", "url": "https://news.example/2015/story", "publishedDate": "2015-04-01T00:00:00.000Z",
         "author": "A", "highlights": ["Rosa Camfield, 101, holds her great-granddaughter."]},
        {"id": "https://blog.example/2024/repost", "url": "https://blog.example/2024/repost", "publishedDate": "2024-05-01T00:00:00.000Z",
         "highlights": ["Woman, 101, gives birth to 17th child."]},
    ]
}
ENV = {"SERPER_API_KEY": "s", "EXA_API_KEY": "e"}


class _Scope:
    def __init__(self, journal: RetrievalJournal, **constraints) -> None:
        self.journal, self.constraints = journal, ResearchConstraints(**constraints)

    def __enter__(self) -> RetrievalJournal:
        self.tokens = (activate_journal(self.journal), activate_constraints(self.constraints), activate_images([IMAGE, LOCAL_ONLY]))
        return self.journal

    def __exit__(self, *_exc: object) -> None:
        deactivate_images(self.tokens[2])
        deactivate_constraints(self.tokens[1])
        deactivate_journal(self.tokens[0])


class ReverseImageSearchTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        patcher = patch.dict("os.environ", ENV, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _patch(self, lens=None, pages=None):
        lens_mock = AsyncMock(return_value=LENS) if lens is None else lens
        pages_mock = AsyncMock(return_value=PAGES) if pages is None else pages
        return lens_mock, pages_mock, patch.object(ris, "_request_lens", lens_mock), patch.object(ris, "_request_exa", pages_mock)

    async def test_blocked_and_post_cutoff_and_undated_pages_never_reach_the_model(self):
        lens, pages, p1, p2 = self._patch()
        with p1, p2, _Scope(RetrievalJournal(), claim_date="2023-07-04", blocked_domains=["snopes.com"]) as journal:
            result = await ris.run_reverse_image_search("image_1", focus="who is the woman")

        lens.assert_awaited_once_with("https://host.example/a.jpg", "s")
        body = pages.await_args.args[0]
        self.assertNotIn("https://www.snopes.com/fact-check/photo", body["urls"])  # blocked before reading
        self.assertEqual(body["highlights"]["query"], "who is the woman")
        self.assertEqual([item["url"] for item in result["results"]], ["https://news.example/2015/story"])
        self.assertEqual(result["results"][0]["published"], "2015-04-01T00:00:00.000Z")
        self.assertEqual(result["results"][0]["title"], "Old photo story")
        self.assertEqual(
            (result["blocked_result_count"], result["after_cutoff_result_count_removed"], result["undated_result_count_removed"]),
            (1, 1, 1),
        )
        self.assertEqual(result["matches_found"], 4)
        [record] = journal.records
        self.assertEqual((record.provider, record.kind, record.query), ("reverse_image_search", "reverse_image", "image_1 | who is the woman"))
        self.assertEqual(record.urls, ["https://news.example/2015/story"])
        # Snippets the model copies from the result are traceable to the journal.
        cited = SourceEvidence(title="t", url="https://news.example/2015/story", snippet="Rosa Camfield, 101, holds her great-granddaughter.")
        journal.complete = True
        self.assertEqual(ground_source(cited, journal), SourceGrounding.QUOTE_MATCHED)

    async def test_without_a_cutoff_undated_matches_are_kept_with_their_titles(self):
        _, _, p1, p2 = self._patch()
        with p1, p2, _Scope(RetrievalJournal()):
            result = await ris.run_reverse_image_search("image_1", count=3)
        self.assertEqual(
            [item["url"] for item in result["results"]],
            ["https://news.example/2015/story", "https://www.snopes.com/fact-check/photo", "https://blog.example/2024/repost"],
        )
        self.assertNotIn("published_before", result)

    async def test_images_it_cannot_search_get_a_notice_without_spending_a_call(self):
        lens, _, p1, p2 = self._patch()
        with p1, p2, _Scope(RetrievalJournal(search_cap=1)) as journal:
            no_url = await ris.run_reverse_image_search("image_2")
            unknown = await ris.run_reverse_image_search("image_9")
        self.assertEqual((no_url["error"], unknown["error"]), ("no_public_url", "unknown_image_id"))
        lens.assert_not_awaited()
        self.assertEqual(journal.records, [])

    async def test_the_search_cap_refuses_further_calls(self):
        lens, _, p1, p2 = self._patch()
        with p1, p2, _Scope(RetrievalJournal(search_cap=1)) as journal:
            await ris.run_reverse_image_search("image_1")
            refused = await ris.run_reverse_image_search("image_1")
        self.assertEqual(refused["error"], "search_limit_reached")
        self.assertEqual(lens.await_count, 1)
        self.assertTrue(journal.records[-1].refused)

    async def test_the_cache_replays_lens_and_page_reads(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = SearchCache(tmp)
            _, _, p1, p2 = self._patch()
            with p1, p2, _Scope(RetrievalJournal(), claim_date="2023-07-04"):
                first = await ris.run_reverse_image_search("image_1", cache=cache)
            lens, pages, p1, p2 = self._patch(lens=AsyncMock(side_effect=AssertionError), pages=AsyncMock(side_effect=AssertionError))
            with p1, p2, _Scope(RetrievalJournal(), claim_date="2023-07-04") as journal:
                second = await ris.run_reverse_image_search("image_1", cache=cache)
        self.assertEqual(first["results"], second["results"])
        self.assertTrue(journal.records[0].cached)

    async def test_a_second_call_on_the_same_image_reuses_the_lens_matches(self):
        lens, pages, p1, p2 = self._patch()
        memo: dict = {}
        with p1, p2, _Scope(RetrievalJournal()) as journal:
            await ris.run_reverse_image_search("image_1", focus="who", lens_memo=memo)
            await ris.run_reverse_image_search("image_1", focus="where", lens_memo=memo)
        self.assertEqual(lens.await_count, 1)
        self.assertEqual(pages.await_count, 2)  # highlights depend on the focus
        self.assertEqual([record.cached for record in journal.records], [False, True])

    async def test_failures_become_tool_results_and_undatable_pages_are_dropped_under_a_cutoff(self):
        failing = AsyncMock(side_effect=httpx.ConnectError("down"))
        _, _, p1, p2 = self._patch(lens=failing)
        with p1, p2, _Scope(RetrievalJournal()) as journal:
            result = await ris.run_reverse_image_search("image_1")
        self.assertIn("reverse image search failed", result["error"])
        self.assertEqual(journal.records[0].error, result["error"])

        _, _, p1, p2 = self._patch(pages=AsyncMock(side_effect=httpx.ReadTimeout("slow")))
        with p1, p2, _Scope(RetrievalJournal(), claim_date="2023-07-04"):
            result = await ris.run_reverse_image_search("image_1")
        self.assertEqual(result["results"], [])
        self.assertIn("page dating failed", result["page_error"])

    async def test_a_rejected_or_missing_key_fails_the_run(self):
        _, _, p1, p2 = self._patch(lens=AsyncMock(side_effect=SearchAuthError("Serper rejected the API key (HTTP 403).")))
        with p1, p2, _Scope(RetrievalJournal()), self.assertRaises(SearchAuthError):
            await ris.run_reverse_image_search("image_1")
        with patch.dict("os.environ", {"SERPER_API_KEY": ""}), _Scope(RetrievalJournal()), self.assertRaises(SearchAuthError):
            await ris.run_reverse_image_search("image_1")

    def test_the_tool_is_added_next_to_the_web_search_tool_only_when_enabled(self):
        shared = dict(search_context_size="high", max_chars_per_field=100, max_total_chars=1000, search_results_per_query=5)
        names = [tool.name for tool in build_search_tools(search_provider="exa", reverse_image_search=True, **shared)]
        self.assertEqual(names, ["exa_web_search", "reverse_image_search"])
        self.assertEqual(len(build_search_tools(search_provider="exa", **shared)), 1)
        [tool] = build_search_tools(search_provider="exa", reverse_image_search=True, **shared)[1:]
        self.assertEqual(sorted(json.loads(json.dumps(tool.params_json_schema))["properties"]), ["count", "focus", "image_id"])


if __name__ == "__main__":
    unittest.main()
