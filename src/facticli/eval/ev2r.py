from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from .ev2r_prompts import (
    ATOMIC_QUESTION_REFERENCE_PROMPT_PREC_RECALL,
    ATOMIC_REFERENCE_PROMPT_PREC_RECALL,
)

# Vendored re-implementation of the FEVER-8 shared task Ev2R scorer
# (Raldir/FEVER-8-Shared-Task averitec_evaluate.py, EV2REvaluator), trimmed of
# the torch/pandas dependencies. Prompt text, evidence serialization, JSON
# parsing, and the recall arithmetic are kept identical for scoring parity.
#
# Threshold note: the FEVER-8 local scorer reports at recall > 0.5; the
# permanent HF leaderboard (fever/AVeriTeC) uses 0.44 with a Llama-3.3-70B
# judge. Both the threshold list and the judge model/endpoint are
# configurable here; state your choice when reporting numbers.

_JSON_BLOCK = re.compile(r"\{(.*?)\}", re.DOTALL)


def join_qa_pairs(qa_pairs: list[tuple[str, str]]) -> str:
    """Serialize QA pairs exactly like the shared-task scorer's evi encoding."""
    return "\t\t\n\n".join(f"{question}\t\t\n{answer}" for question, answer in qa_pairs)


def submission_row_evidence(row: dict[str, Any]) -> str:
    """Build the predicted-evidence string from a submission row."""
    qa_pairs: list[tuple[str, str]] = []
    for item in row.get("evidence", []):
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        if question or answer:
            qa_pairs.append((question, answer))
    return join_qa_pairs(qa_pairs)


def gold_record_evidence(record: dict[str, Any]) -> str:
    """Build the reference-evidence string from an AVeriTeC gold record."""
    qa_pairs: list[tuple[str, str]] = []
    for question_item in record.get("questions", []):
        question = str(question_item.get("question", "")).strip()
        answers = question_item.get("answers", [])
        if not isinstance(answers, list):
            answers = [answers]
        if len(answers) == 0:
            qa_pairs.append((question, "No answer could be found."))
            continue
        for answer_item in answers:
            answer_text = "No answer could be found."
            if isinstance(answer_item, dict):
                answer_text = str(answer_item.get("answer", answer_text)).strip()
            elif answer_item is not None:
                answer_text = str(answer_item).strip()
            qa_pairs.append((question, answer_text))
    return join_qa_pairs(qa_pairs)


def _questions_only(evidence: str) -> str:
    questions = []
    for qa_pair in evidence.split("\t\t\n\n"):
        question = qa_pair.split("\t\t\n")[0]
        if question:
            questions.append(question)
    return " ".join(questions)


@dataclass
class Ev2rJudgeConfig:
    """Judge endpoint configuration for the Ev2R scorer."""
    model: str = field(default_factory=lambda: os.getenv("OPENAI_EVAL_MODEL", "gpt-5.2"))
    base_url: str | None = field(default_factory=lambda: os.getenv("OPENAI_EVAL_BASE_URL") or None)
    api_key: str | None = field(default_factory=lambda: os.getenv("OPENAI_EVAL_API_KEY") or None)
    max_tokens: int = 3000
    max_retries: int = 3
    workers: int = 8


class Ev2rScorer:
    """Computes Ev2R QA-recall (and the recall-gated new AVeriTeC score)."""

    def __init__(self, config: Ev2rJudgeConfig | None = None):
        from openai import OpenAI

        self.config = config or Ev2rJudgeConfig()
        kwargs: dict[str, Any] = {}
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url
        if self.config.api_key:
            kwargs["api_key"] = self.config.api_key
        self._client = OpenAI(**kwargs)

    def _query_judge(self, prompt: str) -> dict[str, Any] | None:
        for attempt in range(self.config.max_retries):
            try:
                request_kwargs: dict[str, Any] = {
                    "model": self.config.model,
                    "messages": [{"role": "user", "content": prompt}],
                }
                try:
                    completion = self._client.chat.completions.create(
                        **request_kwargs, max_completion_tokens=self.config.max_tokens
                    )
                except Exception as exc:
                    if "max_completion_tokens" in str(exc) and "max_tokens" in str(exc):
                        completion = self._client.chat.completions.create(
                            **request_kwargs, max_tokens=self.config.max_tokens
                        )
                    else:
                        raise
                content = completion.choices[0].message.content or ""
                matches = _JSON_BLOCK.findall(content)
                if not matches:
                    raise ValueError("Judge response contained no JSON object.")
                return json.loads("{" + matches[0] + "}")
            except Exception:
                if attempt + 1 >= self.config.max_retries:
                    return None
                time.sleep(min(2 ** (attempt + 1), 30))
        return None

    def _score_example(
        self,
        *,
        claim: str,
        reference_evidence: str,
        predicted_evidence: str,
        questions_only: bool,
    ) -> dict[str, float] | None:
        if questions_only:
            prompt = ATOMIC_QUESTION_REFERENCE_PROMPT_PREC_RECALL.format(
                claim, _questions_only(reference_evidence), _questions_only(predicted_evidence)
            )
            support_key, count_key = "support reference questions", "facts count reference questions"
            pred_support_key, pred_count_key = "support predicted questions", "facts count predicted questions"
        else:
            prompt = ATOMIC_REFERENCE_PROMPT_PREC_RECALL.format(
                claim, reference_evidence, predicted_evidence
            )
            support_key, count_key = "support reference evidence", "facts count reference evidence"
            pred_support_key, pred_count_key = "support predicted evidence", "facts count predicted evidence"

        response = self._query_judge(prompt)
        if response is None:
            return None
        try:
            recall = float(response[support_key]) / float(response[count_key])
            precision = float(response[pred_support_key]) / float(response[pred_count_key])
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            return None
        return {"recall": recall, "precision": precision}

    def score(
        self,
        *,
        examples: list[dict[str, Any]],
        thresholds: list[float] | None = None,
        questions_only: bool = False,
    ) -> dict[str, Any]:
        """Score examples: dicts with claim, reference/predicted evidence, labels.

        Each example needs: claim, reference_evidence, predicted_evidence,
        gold_label, pred_label, and optionally claim_id.
        """
        thresholds = thresholds or [0.5]

        def worker(example: dict[str, Any]) -> dict[str, Any]:
            scores = self._score_example(
                claim=example["claim"],
                reference_evidence=example["reference_evidence"],
                predicted_evidence=example["predicted_evidence"],
                questions_only=questions_only,
            )
            recall = scores["recall"] if scores else 0.0
            precision = scores["precision"] if scores else 0.0
            label_correct = example["gold_label"] == example["pred_label"]
            return {
                "claim_id": example.get("claim_id"),
                "recall": recall,
                "precision": precision,
                "judge_failed": scores is None,
                "label_correct": label_correct,
                "scored": {
                    str(level): (1.0 if (recall > level and label_correct) else 0.0)
                    for level in thresholds
                },
            }

        with ThreadPoolExecutor(max_workers=max(1, self.config.workers)) as pool:
            per_example = list(pool.map(worker, examples))

        count = len(per_example) or 1
        return {
            "judge_model": self.config.model,
            "judge_base_url": self.config.base_url,
            "questions_only": questions_only,
            "examples": len(per_example),
            "judge_failures": sum(1 for item in per_example if item["judge_failed"]),
            "mean_recall": round(sum(item["recall"] for item in per_example) / count, 4),
            "mean_precision": round(sum(item["precision"] for item in per_example) / count, 4),
            "new_averitec_score": {
                str(level): round(
                    sum(item["scored"][str(level)] for item in per_example) / count, 4
                )
                for level in thresholds
            },
            "per_example": per_example,
        }
