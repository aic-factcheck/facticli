from __future__ import annotations

import unittest
from unittest.mock import patch

from facticli.eval.ev2r import (
    Ev2rJudgeConfig,
    Ev2rScorer,
    gold_record_evidence,
    join_qa_pairs,
    submission_row_evidence,
)
from facticli.eval.label_metrics import compute_label_metrics


class LabelMetricTests(unittest.TestCase):
    def test_metrics_on_small_example(self):
        gold = ["Supported", "Refuted", "Refuted", "Not Enough Evidence"]
        pred = ["Supported", "Refuted", "Supported", "Refuted"]
        metrics = compute_label_metrics(gold, pred)

        self.assertEqual(metrics["examples"], 4)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["per_label"]["Supported"]["support"], 1)
        self.assertEqual(metrics["per_label"]["Supported"]["predicted"], 2)
        self.assertEqual(metrics["per_label"]["Supported"]["false_positives"], 1)
        self.assertEqual(metrics["confusion"]["Refuted"]["Supported"], 1)
        # FP rate of Supported: 1 wrong Supported among 3 non-Supported gold rows.
        self.assertAlmostEqual(metrics["per_label"]["Supported"]["fp_rate"], 1 / 3, places=3)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            compute_label_metrics(["Supported"], [])


class EvidenceSerializationTests(unittest.TestCase):
    def test_join_qa_pairs_matches_shared_task_encoding(self):
        joined = join_qa_pairs([("Q1", "A1"), ("Q2", "A2")])
        self.assertEqual(joined, "Q1\t\t\nA1\t\t\n\nQ2\t\t\nA2")

    def test_submission_row_evidence(self):
        row = {"evidence": [{"question": "Q", "answer": "A", "url": "u"}, {"question": "", "answer": ""}]}
        self.assertEqual(submission_row_evidence(row), "Q\t\t\nA")

    def test_gold_record_evidence_handles_missing_answers(self):
        record = {
            "questions": [
                {"question": "Q1", "answers": [{"answer": "A1"}]},
                {"question": "Q2", "answers": []},
            ]
        }
        self.assertEqual(gold_record_evidence(record), "Q1\t\t\nA1\t\t\n\nQ2\t\t\nNo answer could be found.")


class Ev2rScorerTests(unittest.TestCase):
    def test_score_computes_recall_gated_averitec_score(self):
        examples = [
            {
                "claim_id": 0,
                "claim": "C1",
                "reference_evidence": "Q\t\t\nA",
                "predicted_evidence": "Q\t\t\nA",
                "gold_label": "Supported",
                "pred_label": "Supported",
            },
            {
                "claim_id": 1,
                "claim": "C2",
                "reference_evidence": "Q\t\t\nA",
                "predicted_evidence": "Q\t\t\nB",
                "gold_label": "Refuted",
                "pred_label": "Supported",
            },
        ]
        judge_responses = [
            {
                "facts count predicted evidence": 2,
                "support predicted evidence": 2,
                "facts count reference evidence": 2,
                "support reference evidence": 2,
            },
            {
                "facts count predicted evidence": 2,
                "support predicted evidence": 0,
                "facts count reference evidence": 2,
                "support reference evidence": 2,
            },
        ]

        with patch.object(Ev2rScorer, "__init__", lambda self, config=None: None):
            scorer = Ev2rScorer()
            scorer.config = Ev2rJudgeConfig(model="test-judge", workers=1)
            with patch.object(scorer, "_query_judge", side_effect=judge_responses):
                result = scorer.score(examples=examples, thresholds=[0.5])

        self.assertEqual(result["examples"], 2)
        self.assertEqual(result["judge_failures"], 0)
        self.assertEqual(result["mean_recall"], 1.0)
        # Example 0: recall 1.0 > 0.5 and label correct -> 1; example 1: label wrong -> 0.
        self.assertEqual(result["new_averitec_score"]["0.5"], 0.5)

    def test_judge_failure_counts_as_zero_recall(self):
        examples = [
            {
                "claim_id": 0,
                "claim": "C",
                "reference_evidence": "Q\t\t\nA",
                "predicted_evidence": "Q\t\t\nA",
                "gold_label": "Supported",
                "pred_label": "Supported",
            }
        ]
        with patch.object(Ev2rScorer, "__init__", lambda self, config=None: None):
            scorer = Ev2rScorer()
            scorer.config = Ev2rJudgeConfig(model="test-judge", workers=1)
            with patch.object(scorer, "_query_judge", return_value=None):
                result = scorer.score(examples=examples, thresholds=[0.5])

        self.assertEqual(result["judge_failures"], 1)
        self.assertEqual(result["new_averitec_score"]["0.5"], 0.0)


if __name__ == "__main__":
    unittest.main()
