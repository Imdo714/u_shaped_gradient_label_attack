from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans


def cluster_normalized_gradients(
    normalized_features: np.ndarray, k: int, random_seed: int = 42
) -> tuple[np.ndarray, np.ndarray, KMeans]:
    """Unsupervised K-means. The signature intentionally cannot receive labels."""
    if normalized_features.ndim != 2:
        raise ValueError("normalized_features must have shape [samples, features]")
    if len(normalized_features) < k:
        raise ValueError(f"Need at least k={k} samples")
    model = KMeans(n_clusters=k, random_state=random_seed, n_init=20)
    cluster_ids = model.fit_predict(normalized_features)
    centroids = model.cluster_centers_
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12
    return cluster_ids, centroids, model


def save_cluster_results(
    path: str | Path, sample_ids: list[str], epochs: np.ndarray, cluster_ids: np.ndarray
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("sample_id", "epoch", "cluster_id"))
        writer.writeheader()
        for sample_id, epoch, cluster_id in zip(sample_ids, epochs, cluster_ids):
            writer.writerow(
                {"sample_id": sample_id, "epoch": int(epoch), "cluster_id": int(cluster_id)}
            )
