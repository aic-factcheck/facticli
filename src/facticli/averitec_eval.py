from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from facticli.eval.label_metrics import AVERITEC_LABELS, compute_consistency_metrics, compute_label_metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m facticli.averitec_eval",
        description=(
            "Score an AVeriTeC submission file against gold data: label accuracy, "
            "macro-F1, per-label metrics, confusion matrix, and optionally the "
            "Ev2R-recall-based new AVeriTeC score (LLM judge)."
        ),
    )
    parser.add_argument(
        "--submission",
        required=True,
        action="append",
        help=(
            "Submission JSON (facticli averitec_submission output). Repeat the flag with runs of the same "
            "claims to also report consistency (pass^k, pass@k, agreement, majority vote); the first file "
            "is scored in full."
        ),
    )
    parser.add_argument("--gold", required=True, help="Gold AVeriTeC JSON (e.g. data/averitec/dev.json).")
    parser.add_argument("--output", default=None, help="Optional path to write the metrics JSON.")
    parser.add_argument(
        "--ev2r",
        action="store_true",
        help="Also compute Ev2R QA-recall and the recall-gated new AVeriTeC score (requires judge LLM access).",
    )
    parser.add_argument(
        "--ev2r-questions",
        action="store_true",
        help="Additionally compute the question-only Ev2R recall.",
    )
    parser.add_argument(
        "--ev2r-model",
        default=None,
        help="Judge model (default: OPENAI_EVAL_MODEL or gpt-5.2). "
        "The official permanent leaderboard uses Llama-3.3-70B.",
    )
    parser.add_argument(
        "--ev2r-base-url",
        default=None,
        help="Judge endpoint base URL (default: OPENAI_EVAL_BASE_URL; use an Ollama/vLLM endpoint for on-prem judging).",
    )
    parser.add_argument(
        "--ev2r-threshold",
        type=float,
        action="append",
        default=None,
        help="Recall threshold(s) for the new AVeriTeC score (repeatable). "
        "Default: 0.5 (FEVER-8 local scorer); the permanent HF leaderboard uses 0.44.",
    )
    parser.add_argument("--ev2r-workers", type=int, default=8, help="Concurrent judge requests (default: 8).")
    return parser


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def match_submission_to_gold(
    submission: list[dict[str, Any]],
    gold: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Pair each submission row with its gold record via claim_id row index."""
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for position, row in enumerate(submission):
        claim_id = row.get("claim_id", position)
        try:
            index = int(claim_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Submission row {position} has non-numeric claim_id {claim_id!r}.") from exc
        if not 0 <= index < len(gold):
            raise ValueError(f"Submission claim_id {index} is out of range for gold size {len(gold)}.")
        gold_record = gold[index]
        claim = str(row.get("claim", "")).strip()
        gold_claim = str(gold_record.get("claim", "")).strip()
        if claim and gold_claim and claim != gold_claim:
            print(
                f"[warn] claim text mismatch at claim_id {index}; scoring by id anyway.",
                file=sys.stderr,
            )
        pairs.append((row, gold_record))
    return pairs


def _run(args: argparse.Namespace) -> int:
    submission_paths = [Path(item) for item in args.submission]
    submission_path = submission_paths[0]
    gold_path = Path(args.gold)
    for path in (*submission_paths, gold_path):
        if not path.is_file():
            print(f"File does not exist: {path}", file=sys.stderr)
            return 2

    submission = _load_json(submission_path)
    gold = _load_json(gold_path)
    if not isinstance(submission, list) or not isinstance(gold, list):
        print("Both submission and gold files must contain JSON lists.", file=sys.stderr)
        return 2

    try:
        pairs = match_submission_to_gold(submission, gold)
    except ValueError as exc:
        print(f"Failed to match submission to gold: {exc}", file=sys.stderr)
        return 2

    gold_labels = [str(gold_record.get("label", "")) for _, gold_record in pairs]
    pred_labels = [str(row.get("pred_label", "")) for row, _ in pairs]
    metrics: dict[str, Any] = {
        "submission": str(submission_path),
        "gold": str(gold_path),
        "label_metrics": compute_label_metrics(gold_labels, pred_labels),
    }

    if len(submission_paths) > 1:
        try:
            metrics["consistency"] = _consistency_across_runs(submission_paths, gold)
            metrics["consistency"]["submissions"] = [str(path) for path in submission_paths]
        except ValueError as exc:
            print(f"Failed to compute consistency: {exc}", file=sys.stderr)
            return 2

    if args.ev2r or args.ev2r_questions:
        if not (os.getenv("OPENAI_EVAL_API_KEY") or os.getenv("OPENAI_API_KEY")):
            print("Ev2R judging requires OPENAI_EVAL_API_KEY or OPENAI_API_KEY.", file=sys.stderr)
            return 2
        from facticli.eval.ev2r import (
            Ev2rJudgeConfig,
            Ev2rScorer,
            gold_record_evidence,
            submission_row_evidence,
        )

        config = Ev2rJudgeConfig(workers=args.ev2r_workers)
        if args.ev2r_model:
            config.model = args.ev2r_model
        if args.ev2r_base_url:
            config.base_url = args.ev2r_base_url
        scorer = Ev2rScorer(config)

        examples = [
            {
                "claim_id": row.get("claim_id"),
                "claim": str(gold_record.get("claim", "")),
                "reference_evidence": gold_record_evidence(gold_record),
                "predicted_evidence": submission_row_evidence(row),
                "gold_label": str(gold_record.get("label", "")),
                "pred_label": str(row.get("pred_label", "")),
            }
            for row, gold_record in pairs
        ]
        thresholds = args.ev2r_threshold or [0.5]
        if args.ev2r:
            metrics["ev2r_qa"] = scorer.score(examples=examples, thresholds=thresholds)
        if args.ev2r_questions:
            metrics["ev2r_questions"] = scorer.score(
                examples=examples, thresholds=thresholds, questions_only=True
            )

    _print_summary(metrics)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[info] Wrote metrics to {output_path}.", file=sys.stderr)
    return 0


def _consistency_across_runs(submission_paths: list[Path], gold: list[dict[str, Any]]) -> dict[str, Any]:
    """Align repeated submissions on shared claim_ids and score their agreement."""
    per_run: list[dict[int, str]] = []
    for path in submission_paths:
        rows = _load_json(path)
        if not isinstance(rows, list):
            raise ValueError(f"{path} does not contain a JSON list.")
        labels: dict[int, str] = {}
        for position, row in enumerate(rows):
            labels[int(row.get("claim_id", position))] = str(row.get("pred_label", ""))
        per_run.append(labels)
    shared_ids = sorted(set.intersection(*(set(labels) for labels in per_run)))
    if not shared_ids:
        raise ValueError("Submissions share no claim_ids.")
    gold_labels = [str(gold[claim_id].get("label", "")) for claim_id in shared_ids if 0 <= claim_id < len(gold)]
    valid_ids = [claim_id for claim_id in shared_ids if 0 <= claim_id < len(gold)]
    runs = [[labels[claim_id] for claim_id in valid_ids] for labels in per_run]
    return compute_consistency_metrics(gold_labels, runs)


def _print_summary(metrics: dict[str, Any]) -> None:
    label_metrics = metrics["label_metrics"]
    print(f"examples:  {label_metrics['examples']}")
    print(f"accuracy:  {label_metrics['accuracy']}")
    print(f"macro F1:  {label_metrics['macro_f1']}")
    print("per label (P / R / F1 / support / FP-rate):")
    for label in AVERITEC_LABELS:
        entry = label_metrics["per_label"][label]
        print(
            f"  {label:<38} {entry['precision']:.3f} / {entry['recall']:.3f} / "
            f"{entry['f1']:.3f} / {entry['support']:>3} / {entry['fp_rate']:.3f}"
        )
    print("confusion (rows = gold, cols = predicted):")
    header = " ".join(f"{label[:12]:>13}" for label in AVERITEC_LABELS)
    print(f"  {'':<38}{header}")
    for gold_label in AVERITEC_LABELS:
        row = metrics["label_metrics"]["confusion"][gold_label]
        cells = " ".join(f"{row[pred]:>13}" for pred in AVERITEC_LABELS)
        print(f"  {gold_label:<38}{cells}")
    if "consistency" in metrics:
        block = metrics["consistency"]
        print(
            f"consistency over k={block['k']} runs ({block['examples']} shared claims): "
            f"pass^k {block['pass_hat_k']}, pass@k {block['pass_at_k']}, agreement {block['agreement_rate']}, "
            f"majority-vote accuracy {block['majority_vote_accuracy']}, per-run {block['per_run_accuracy']}"
        )
    for key in ("ev2r_qa", "ev2r_questions"):
        if key in metrics:
            block = metrics[key]
            print(
                f"{key}: mean recall {block['mean_recall']}, "
                f"new AVeriTeC score {block['new_averitec_score']} "
                f"(judge {block['judge_model']}, failures {block['judge_failures']})"
            )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
