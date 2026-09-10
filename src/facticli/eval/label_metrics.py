from __future__ import annotations

from typing import Any

AVERITEC_LABELS: tuple[str, ...] = (
    "Supported",
    "Refuted",
    "Not Enough Evidence",
    "Conflicting Evidence/Cherrypicking",
)


def compute_label_metrics(
    gold_labels: list[str],
    pred_labels: list[str],
    labels: tuple[str, ...] = AVERITEC_LABELS,
) -> dict[str, Any]:
    """Accuracy, per-label P/R/F1, macro-F1, confusion matrix, and FP rates.

    The per-label false-positive rate is the share of examples with a
    different gold label that were predicted as this label — the failure
    mode highlighted for NEI and Conflicting verdicts in judge analyses.
    """
    if len(gold_labels) != len(pred_labels):
        raise ValueError(
            f"Gold ({len(gold_labels)}) and predicted ({len(pred_labels)}) label counts differ."
        )
    total = len(gold_labels)
    if total == 0:
        raise ValueError("No examples to score.")

    confusion: dict[str, dict[str, int]] = {gold: {pred: 0 for pred in labels} for gold in labels}
    unknown_gold: list[str] = []
    unknown_pred: list[str] = []
    correct = 0
    for gold, pred in zip(gold_labels, pred_labels):
        if gold not in confusion:
            unknown_gold.append(gold)
            continue
        if pred not in confusion[gold]:
            unknown_pred.append(pred)
            continue
        confusion[gold][pred] += 1
        if gold == pred:
            correct += 1

    per_label: dict[str, dict[str, float]] = {}
    f1_values: list[float] = []
    for label in labels:
        tp = confusion[label][label]
        support = sum(confusion[label].values())
        predicted = sum(confusion[gold][label] for gold in labels)
        fp = predicted - tp
        fn = support - tp
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        negatives = total - support
        per_label[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
            "predicted": predicted,
            "false_positives": fp,
            "false_negatives": fn,
            "fp_rate": round(fp / negatives, 4) if negatives else 0.0,
        }
        f1_values.append(f1)

    return {
        "examples": total,
        "accuracy": round(correct / total, 4),
        "macro_f1": round(sum(f1_values) / len(f1_values), 4),
        "per_label": per_label,
        "confusion": confusion,
        "unknown_gold_labels": sorted(set(unknown_gold)),
        "unknown_pred_labels": sorted(set(unknown_pred)),
    }


def compute_consistency_metrics(
    gold_labels: list[str],
    runs: list[list[str]],
) -> dict[str, Any]:
    """Agreement across k repeated runs of the same claims (pass^k, pass@k, majority vote).

    ``runs`` holds one predicted-label list per repetition, aligned with
    ``gold_labels``. pass^k is the share of claims every run got right (the
    reliability number), pass@k the share at least one run got right, and
    ``agreement_rate`` the share of claims where all runs agree regardless of
    gold. Mean accuracy hides verdict flip-flopping; these do not.
    """
    if not runs:
        raise ValueError("At least one run is required.")
    total = len(gold_labels)
    for index, run in enumerate(runs):
        if len(run) != total:
            raise ValueError(f"Run {index} has {len(run)} labels, expected {total}.")
    if total == 0:
        raise ValueError("No examples to score.")

    k = len(runs)
    all_correct = 0
    any_correct = 0
    all_agree = 0
    majority_correct = 0
    per_run_accuracy: list[float] = []
    for run in runs:
        per_run_accuracy.append(round(sum(1 for g, p in zip(gold_labels, run) if g == p) / total, 4))
    for index, gold in enumerate(gold_labels):
        predictions = [run[index] for run in runs]
        correct = [prediction == gold for prediction in predictions]
        all_correct += int(all(correct))
        any_correct += int(any(correct))
        all_agree += int(len(set(predictions)) == 1)
        counts: dict[str, int] = {}
        for prediction in predictions:
            counts[prediction] = counts.get(prediction, 0) + 1
        top = max(counts.values())
        winners = sorted(label for label, count in counts.items() if count == top)
        majority_correct += int(len(winners) == 1 and winners[0] == gold)
    return {
        "k": k,
        "examples": total,
        "per_run_accuracy": per_run_accuracy,
        "mean_accuracy": round(sum(per_run_accuracy) / k, 4),
        "pass_hat_k": round(all_correct / total, 4),
        "pass_at_k": round(any_correct / total, 4),
        "agreement_rate": round(all_agree / total, 4),
        "majority_vote_accuracy": round(majority_correct / total, 4),
    }
