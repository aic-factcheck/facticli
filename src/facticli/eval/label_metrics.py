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
