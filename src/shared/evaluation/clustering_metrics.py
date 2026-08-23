from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    confusion_matrix,
    f1_score,
    normalized_mutual_info_score,
    precision_score,
    recall_score,
)


def clustering_purity(true_labels: np.ndarray, cluster_ids: np.ndarray) -> float:
    total = 0
    for cluster in np.unique(cluster_ids):
        labels = true_labels[cluster_ids == cluster]
        total += int(np.bincount(labels).max())
    return total / len(true_labels)


def evaluate_clusters(
    true_labels: np.ndarray,
    cluster_ids: np.ndarray,
    cluster_to_label: dict[int, int],
    num_classes: int | None = None,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    inferred = np.asarray([cluster_to_label[int(c)] for c in cluster_ids])
    if num_classes is None:
        num_classes = max((*cluster_to_label.values(), *true_labels.tolist())) + 1
    metrics = {
        "purity": clustering_purity(true_labels, cluster_ids),
        "ARI": adjusted_rand_score(true_labels, cluster_ids),
        "NMI": normalized_mutual_info_score(true_labels, cluster_ids),
        "attack_accuracy": accuracy_score(true_labels, inferred),
        "precision": precision_score(
            true_labels, inferred, average="macro", zero_division=0
        ),
        "recall": recall_score(true_labels, inferred, average="macro", zero_division=0),
        "F1": f1_score(true_labels, inferred, average="macro", zero_division=0),
    }
    labels = np.arange(num_classes)
    return metrics, confusion_matrix(true_labels, inferred, labels=labels), inferred
