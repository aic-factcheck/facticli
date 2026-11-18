from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from facticli.core.contracts import AspectFinding, EvidenceSignal, FindingStatus, SourceEvidence, SourceGrounding
from facticli.core.grounding import (
    ground_finding,
    ground_source,
    normalize_evidence_text,
    quote_segments,
    snippet_in_text,
    summarize_grounding,
    unresolved_citation_ids,
)
from facticli.core.retrieval import (
    OPEN_PAGE,
    RetrievalJournal,
    RetrievalRecord,
    activate_journal,
    deactivate_journal,
    refuse_search_over_cap,
    summarize_retrieval,
    url_key,
)
from facticli.core.search_cache import SearchCache

RETRIEVED = (
    "The Eiffel Tower is 330 metres (1,083 ft) tall, about the same height as an 81-storey building. "
    'It was named part of a UNESCO site ("Paris, Banks of the Seine") in 1991. '
    "Officials said inflation stayed < 2% in March and > 3% in June. GDP grew 2.1% in 2020."
)
WIKI = "https://en.wikipedia.org/wiki/Eiffel_Tower"


def _source(url: str, snippet: str = "a long enough verbatim snippet here", **kwargs) -> SourceEvidence:
    return SourceEvidence(title="t", url=url, snippet=snippet, **kwargs)


def _finding(sources: list[SourceEvidence]) -> AspectFinding:
    return AspectFinding(
        aspect_id="a", question="q", signal=EvidenceSignal.SUPPORTS, summary="s", confidence=0.7, sources=sources
    )


def _journal(*records: tuple[RetrievalRecord, list[str]], complete: bool = True) -> RetrievalJournal:
    journal = RetrievalJournal()
    journal.complete = complete
    for record, texts in records:
        journal.add(record, texts)
    return journal


class UrlKeyTests(unittest.TestCase):
    def test_equivalent_spellings_share_a_key(self):
        base = url_key("https://www.Example.org/News/Story/")
        self.assertEqual(base, ("example.org/News/Story", frozenset()))
        for spelling in (
            "http://example.org/News/Story#section",
            "https://example.org/News/Story?utm_source=openai&fbclid=abc",
            "example.org/News/Story",
            "https://example.org./News/Story",
        ):
            self.assertEqual(url_key(spelling), base, spelling)
        self.assertEqual(url_key("https://example.org/caf%C3%A9")[0], "example.org/café")
        self.assertEqual(url_key(""), ("", frozenset()))

    def test_identity_carrying_query_parameters_are_kept(self):
        self.assertEqual(url_key("https://example.org/article?id=7&utm_medium=x"), ("example.org/article", frozenset({("id", "7")})))


class JournalTests(unittest.TestCase):
    def test_cap_admits_then_refuses_with_an_instruction_to_answer(self):
        journal = RetrievalJournal(search_cap=2)
        token = activate_journal(journal)
        try:
            self.assertIsNone(refuse_search_over_cap("brave", "q1"))
            self.assertIsNone(refuse_search_over_cap("brave", "q2"))
            refusal = refuse_search_over_cap("brave", "q3")
        finally:
            deactivate_journal(token)
        self.assertEqual(refusal["error"], "search_limit_reached")
        self.assertIn("All 2 searches", refusal["instruction"])
        self.assertIn("Do not call the search tool again", refusal["instruction"])
        self.assertEqual([(record.query, record.refused) for record in journal.records], [("q3", True)])
        self.assertEqual(journal.refusals, 1)

        # No journal, or no cap: nothing is refused.
        self.assertIsNone(refuse_search_over_cap("brave", "q"))
        uncapped = RetrievalJournal()
        self.assertTrue(all(uncapped.admit_search() for _ in range(50)))

    def test_a_citation_may_drop_query_parameters_but_not_add_them(self):
        journal = _journal(
            (RetrievalRecord(provider="brave", urls=["https://www.example.org/a?id=1", "https://example.org/list"]), ["alpha", "index"])
        )
        self.assertEqual(journal.sightings("http://example.org/a/?id=1&utm_source=openai"), ["alpha"])
        self.assertEqual(journal.sightings("https://example.org/a"), ["alpha"])
        self.assertEqual(journal.sightings("https://example.org/a?id=2"), [])  # a different article
        self.assertEqual(journal.sightings("https://example.org/list?id=12345"), [])  # an id the tools never returned
        self.assertEqual(journal.sightings("https://example.org/b"), [])
        # A URL reported without text (hosted search) is still a sighting.
        journal.add(RetrievalRecord(provider="openai", urls=["https://example.org/b"]))
        self.assertEqual(journal.sightings("https://example.org/b"), [""])

    def test_summary_counts_by_outcome(self):
        summary = summarize_retrieval(
            [
                RetrievalRecord(provider="brave", urls=["https://a.org/x?id=1", "https://a.org/x?id=2"], cached=True),
                RetrievalRecord(provider="openai", kind=OPEN_PAGE, urls=["https://a.org/x?id=1"]),
                RetrievalRecord(provider="brave", error="search request failed (HTTP 500)"),
                RetrievalRecord(provider="brave", refused=True),
            ]
        )
        self.assertEqual(
            summary.model_dump(),
            {"searches": 2, "pages_opened": 1, "cache_hits": 1, "refused": 1, "errors": 1, "distinct_urls": 2},
        )


class GroundingTests(unittest.TestCase):
    def test_normalization_survives_markup_typography_and_spacing(self):
        self.assertEqual(normalize_evidence_text("<strong>Eiff</strong>el  Tower"), "eiffeltower")
        self.assertEqual(normalize_evidence_text("“Paris” – it’s"), normalize_evidence_text('"Paris" - it\'s'))
        self.assertEqual(normalize_evidence_text("1,083&nbsp;ft"), normalize_evidence_text("1,083 ft"))
        self.assertEqual(normalize_evidence_text(""), "")

    def test_verbatim_copy_matches_and_paraphrase_does_not(self):
        for quote in (
            "The Eiffel Tower is 330 metres (1,083 ft) tall",
            'named part of a UNESCO site ("Paris, Banks of the Seine") in 1991',
            "The Eiffel Tower is 330 metres ... about the same height as an 81-storey building",  # parts checked separately
            "inflation stayed < 2% in March and > 3% in June",  # comparison operators are not markup
            "“Officials said inflation stayed < 2% in March”",  # wrapped in typographic quotation marks
            "GDP grew 2.1% in 2020",  # a short factual sentence is checkable
        ):
            self.assertTrue(snippet_in_text(quote, RETRIEVED), quote)
        for not_a_copy in (
            "The Eiffel Tower is 330 metres ... it is the tallest structure in France",
            "The tower stands three hundred and thirty metres high",
            "330 metres",  # too short to prove a copy
        ):
            self.assertFalse(snippet_in_text(not_a_copy, RETRIEVED), not_a_copy)
        self.assertEqual(quote_segments("2.1% in 2020"), [])
        # The elision marker written by tool-output truncation splits like an ellipsis.
        self.assertEqual(len(quote_segments("The Eiffel Tower is 330 metres [...] (120 chars omitted) about the same height")), 2)

    def test_levels(self):
        journal = _journal((RetrievalRecord(provider="brave", urls=[WIKI]), [RETRIEVED]))
        quoted = _source(WIKI + "/", "The Eiffel Tower is 330 metres (1,083 ft) tall")
        paraphrased = _source(WIKI, "The tower reaches a height of 330 m in total today")
        invented = _source("https://example.com/made-up", "The Eiffel Tower is 330 metres (1,083 ft) tall")
        grounded = ground_finding(_finding([quoted, paraphrased, invented]), journal)
        self.assertEqual(
            [source.grounding for source in grounded.sources],
            # A real quote under a URL that was never retrieved is still unmatched.
            [SourceGrounding.QUOTE_MATCHED, SourceGrounding.URL_MATCHED, SourceGrounding.UNMATCHED],
        )
        self.assertEqual(summarize_grounding(grounded.sources), {"quote_matched": 1, "url_matched": 1, "unmatched": 1})
        self.assertEqual(summarize_grounding([quoted]), {})

        # Hosted search reports URLs but no text: URL-level provenance only.
        hosted = _journal((RetrievalRecord(provider="openai", urls=[WIKI]), []))
        self.assertEqual(ground_source(quoted, hosted), SourceGrounding.URL_MATCHED)

    def test_only_a_complete_journal_can_call_a_source_unmatched(self):
        source = _source("https://example.org/a")
        self.assertIsNone(ground_source(source, None))
        self.assertIsNone(ground_source(source, RetrievalJournal()))  # nobody vouched for it
        # A complete journal with no search at all: the citation did not come from a tool.
        self.assertEqual(ground_source(source, _journal()), SourceGrounding.UNMATCHED)
        # The URLs of one call are unknown: a missing URL proves nothing.
        blind = _journal((RetrievalRecord(provider="openai", query="q"), []), complete=False)
        self.assertIsNone(ground_finding(_finding([source]), blind).sources[0].grounding)

    def test_a_label_emitted_by_the_model_never_survives(self):
        # Providers that do not enforce the output schema strictly accept hidden fields.
        finding = AspectFinding.model_validate(
            {
                "aspect_id": "a",
                "question": "q",
                "signal": "supports",
                "summary": "s",
                "confidence": 0.5,
                "status": "failed",
                "failure_reason": "made up",
                "stop_reason": "turn_limit",
                "sources": [
                    {
                        "title": "t",
                        "url": "https://example.org/a",
                        "snippet": "s",
                        "grounding": "quote_matched",
                        "tier": "primary",
                        "url_status": "live",
                        "archived_url": "https://web.archive.org/web/2020/https://example.org/a",
                    }
                ],
            }
        )
        clean = finding.without_harness_fields()
        self.assertEqual((clean.status, clean.failure_reason, clean.stop_reason), (FindingStatus.COMPLETED, None, None))
        source = clean.sources[0]
        self.assertEqual((source.grounding, source.tier, source.url_status, source.archived_url), (None, None, None, None))
        # Grounding overwrites the field even when it cannot assess the source.
        self.assertIsNone(ground_finding(finding, None).sources[0].grounding)

    def test_cited_ids_must_exist_in_the_source_table(self):
        self.assertEqual(unresolved_citation_ids(["Supported [S1][S4].", "Also [S0] and [S4]."], source_count=3), [4, 0])
        self.assertEqual(unresolved_citation_ids(["A [S2][S1].", "no citations", ""], source_count=2), [])
        # Models do not always keep to one id per bracket.
        self.assertEqual(unresolved_citation_ids(["A [S1, S9]; B [S2-S8]; C [S7, p. 3]; [sic] US5"], source_count=3), [9, 8, 7])


class SearchCacheTests(unittest.TestCase):
    def test_records_and_replays_across_instances(self):
        with tempfile.TemporaryDirectory() as tmp:
            request = {"q": "eiffel", "count": 5, "freshness": None}
            first = SearchCache(Path(tmp) / "cache")
            self.assertIsNone(first.get(request))
            first.put(request, [{"url": "https://a.org", "title": "ž"}])

            second = SearchCache(Path(tmp) / "cache")
            self.assertEqual(second.get({"count": 5, "freshness": None, "q": "eiffel"}), [{"url": "https://a.org", "title": "ž"}])
            # Anything that changes what the provider is asked is a different request.
            self.assertIsNone(second.get({**request, "freshness": "1900-01-01to2020-10-31"}))

            # A corrupt entry is a miss, not a crash, and the next put replaces it.
            (entry,) = (Path(tmp) / "cache").rglob("*.json")
            entry.write_text("{not json", encoding="utf-8")
            self.assertIsNone(second.get(request))
            # An empty result list is a recorded answer, not a miss.
            second.put(request, [])
            self.assertEqual(second.get(request), [])

    def test_an_unusable_directory_fails_at_startup(self):
        with tempfile.NamedTemporaryFile() as occupied:
            with self.assertRaisesRegex(ValueError, "cannot be created"):
                SearchCache(Path(occupied.name) / "cache")


if __name__ == "__main__":
    unittest.main()
