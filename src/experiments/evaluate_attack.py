from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ..shared.configuration.workspace_paths import DEFAULT_WORKSPACE_PATHS
from .attacks.gradient_features import load_gradient_dataset
from ..shared.data.class_catalog import ClassCatalog
from ..shared.evaluation.clustering_metrics import evaluate_clusters
from ..shared.evaluation.visualization import plot_confusion, plot_pca


METRIC_LABELS = {
    "purity": "Clustering purity",
    "ARI": "Adjusted Rand Index",
    "NMI": "Normalized Mutual Information",
    "attack_accuracy": "Attack accuracy",
    "precision": "Macro precision",
    "recall": "Macro recall",
    "F1": "Macro F1 score",
}


def load_truth(path: Path, epoch: int | None = None) -> dict[str, int]:
    truth: dict[str, int] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if epoch is None or int(row["epoch"]) == epoch:
                truth[row["sample_id"]] = int(row["true_label"])
    return truth


def load_clusters(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    sample_ids, epochs, clusters = [], [], []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            sample_ids.append(row["sample_id"])
            epochs.append(int(row["epoch"]))
            clusters.append(int(row["cluster_id"]))
    return sample_ids, np.asarray(epochs), np.asarray(clusters)


def confusion_markdown(matrix: np.ndarray, display_names: tuple[str, ...]) -> str:
    header = "| Actual / Inferred | " + " | ".join(display_names) + " |"
    separator = "|---|" + "---:|" * len(display_names)
    rows = [
        f"| {name} | " + " | ".join(str(int(value)) for value in matrix[index]) + " |"
        for index, name in enumerate(display_names)
    ]
    return "\n".join([header, separator, *rows])


def save_evaluation_reports(
    results_dir: Path,
    metrics: dict[str, float],
    matrix: np.ndarray,
    mapping: dict[int, int],
    num_samples: int,
    epoch: int | None,
    class_names: tuple[str, ...],
) -> tuple[Path, Path]:
    display_names = tuple(name.upper() for name in class_names)
    correct = int(np.trace(matrix))
    errors = int(matrix.sum() - correct)
    payload = {
        "epoch": epoch,
        "num_samples": num_samples,
        "cluster_to_label": {
            str(cluster): class_names[label] for cluster, label in sorted(mapping.items())
        },
        "metrics": {name: float(value) for name, value in metrics.items()},
        "confusion_matrix": {
            "labels": list(class_names),
            "rows_are_actual": True,
            "values": matrix.astype(int).tolist(),
        },
        "correct": correct,
        "errors": errors,
        "integrity_note": (
            "Ground truth was joined only after label-free feature extraction and K-means fitting."
        ),
    }
    json_path = results_dir / "evaluation_summary.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    mapping_lines = "\n".join(
        f"- Cluster {cluster} -> **{display_names[label]}**"
        for cluster, label in sorted(mapping.items())
    )
    metric_rows = "\n".join(
        f"| {METRIC_LABELS[name]} | {value:.4f} | {value * 100:.2f}% |"
        for name, value in metrics.items()
    )
    report = f"""# Gradient label-inference evaluation

- Epoch: `{epoch if epoch is not None else 'all'}`
- Samples: `{num_samples}`
- Correct: `{correct}`
- Errors: `{errors}`

## Anchor-to-cluster mapping

{mapping_lines}

## Metrics

| Metric | Score | Percentage |
|---|---:|---:|
{metric_rows}

## Confusion matrix

{confusion_markdown(matrix, display_names)}

Ground truth is used only after feature extraction and clustering are complete.
"""
    markdown_path = results_dir / "evaluation_report.md"
    markdown_path.write_text(report, encoding="utf-8")
    return markdown_path, json_path


def print_evaluation_summary(
    metrics: dict[str, float],
    matrix: np.ndarray,
    mapping: dict[int, int],
    num_samples: int,
    epoch: int | None,
    results_dir: Path,
    class_names: tuple[str, ...],
) -> None:
    display_names = tuple(name.upper() for name in class_names)
    correct = int(np.trace(matrix))
    errors = int(matrix.sum() - correct)
    print("\n" + "=" * 64)
    print("GRADIENT LABEL-INFERENCE EVALUATION")
    print("=" * 64)
    print(f"Epoch: {epoch if epoch is not None else 'all'}")
    print(f"Samples: {num_samples}; correct/errors: {correct}/{errors}")
    for cluster, label in sorted(mapping.items()):
        print(f"Cluster {cluster} -> {display_names[label]}")
    for key, label in METRIC_LABELS.items():
        print(f"{label:30s} {metrics[key]:.4f}")
    print("\nConfusion matrix (rows=actual, columns=inferred)")
    print(" " * 16 + " ".join(f"{name:>10s}" for name in display_names))
    for name, row in zip(display_names, matrix):
        print(f"{name:>14s}  " + " ".join(f"{int(value):10d}" for value in row))
    print(f"Reports: {results_dir / 'evaluation_report.md'}")


def evaluate_files(
    clusters_path: Path,
    truth_path: Path,
    mapping: dict[int, int],
    results_dir: Path,
    transcripts: Path | None = None,
    epoch: int | None = None,
    class_names: tuple[str, ...] = ("cat", "dog"),
):
    sample_ids, _, cluster_ids = load_clusters(clusters_path)
    truth_by_id = load_truth(truth_path, epoch)
    missing = [sample_id for sample_id in sample_ids if sample_id not in truth_by_id]
    if missing:
        raise ValueError(f"Evaluator truth missing for {len(missing)} samples")
    true_labels = np.asarray([truth_by_id[sample_id] for sample_id in sample_ids])
    metrics, matrix, inferred = evaluate_clusters(
        true_labels, cluster_ids, mapping, num_classes=len(class_names)
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    evaluation_path = results_dir / "evaluation_clusters.csv"
    with evaluation_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("sample_id", "cluster_id", "true_label"))
        writer.writeheader()
        for sample_id, cluster, true_label in zip(sample_ids, cluster_ids, true_labels):
            writer.writerow(
                {"sample_id": sample_id, "cluster_id": int(cluster), "true_label": int(true_label)}
            )
    display_names = tuple(name.upper() for name in class_names)
    plot_confusion(matrix, results_dir / "confusion_matrix.png", display_names)
    if transcripts is not None and epoch is not None:
        features = load_gradient_dataset(transcripts, epoch).normalized
        plot_pca(
            features,
            true_labels,
            results_dir / "pca_gradient_ground_truth.png",
            "Evaluator-only gradient PCA by ground truth",
            labels=display_names,
        )
    save_evaluation_reports(
        results_dir, metrics, matrix, mapping, len(true_labels), epoch, class_names
    )
    return metrics, matrix, inferred, true_labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clusters",
        default=str(DEFAULT_WORKSPACE_PATHS.reports / "gradient_clusters.csv"),
    )
    parser.add_argument(
        "--ground-truth",
        default=str(
            DEFAULT_WORKSPACE_PATHS.transcripts
            / "evaluator_ground_truth"
            / "ground_truth.csv"
        ),
    )
    parser.add_argument(
        "--mapping",
        default=str(DEFAULT_WORKSPACE_PATHS.reports / "cluster_mapping.json"),
    )
    parser.add_argument("--results", default=str(DEFAULT_WORKSPACE_PATHS.reports))
    parser.add_argument("--transcripts", default=str(DEFAULT_WORKSPACE_PATHS.transcripts))
    parser.add_argument("--data", default=str(DEFAULT_WORKSPACE_PATHS.dataset))
    parser.add_argument("--epoch", type=int)
    args = parser.parse_args()
    catalog = ClassCatalog.discover(args.data)
    mapping = {int(k): int(v) for k, v in json.loads(Path(args.mapping).read_text()).items()}
    metrics, matrix, _, true_labels = evaluate_files(
        Path(args.clusters),
        Path(args.ground_truth),
        mapping,
        Path(args.results),
        Path(args.transcripts),
        args.epoch,
        catalog.names,
    )
    print_evaluation_summary(
        metrics,
        matrix,
        mapping,
        len(true_labels),
        args.epoch,
        Path(args.results),
        catalog.names,
    )


if __name__ == "__main__":
    main()
