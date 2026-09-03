from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def add_label_report(
    metrics_csv: str | Path,
    output_dir: str | Path,
    class_names: tuple[str, ...],
) -> dict[str, object] | None:
    """Add class-wise metrics omitted by the base reconstruction evaluator."""

    with Path(metrics_csv).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or all(int(row["inferred_label"]) < 0 for row in rows):
        return None

    truth = np.asarray([int(row["true_label"]) for row in rows], dtype=np.int64)
    predicted = np.asarray(
        [int(row["inferred_label"]) for row in rows], dtype=np.int64
    )
    labels = np.arange(len(class_names), dtype=np.int64)
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=labels, zero_division=0
    )
    _, _, macro_f1, _ = precision_recall_fscore_support(
        truth, predicted, labels=labels, average="macro", zero_division=0
    )
    matrix = confusion_matrix(truth, predicted, labels=labels)
    report: dict[str, object] = {
        "accuracy": float((truth == predicted).mean()),
        "macro_f1": float(macro_f1),
        "classes": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(class_names)
        },
        "confusion_matrix": matrix.tolist(),
        "class_order": list(class_names),
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "label_inference_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    np.savetxt(output / "label_confusion_matrix.csv", matrix, delimiter=",", fmt="%d")
    return report


__all__ = ["add_label_report"]
