from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from ...shared.data.image_dataset import load_image
from ...split_learning.architecture.split_learning_model import SplitLearningModel
from ...split_learning.gradient_flow.gradient_exchange import (
    observe_frozen_gradient_exchange,
)


@dataclass
class AnchorMappingResult:
    cluster_to_label: dict[int, int]
    distances: np.ndarray
    cosine_similarities: np.ndarray
    assigned_clusters: np.ndarray


def extract_anchor_gradient(
    model: SplitLearningModel,
    image: torch.Tensor,
    known_anchor_label: int,
    device: torch.device,
) -> np.ndarray:
    """Known auxiliary labels stay client-side while the server receives dL/du."""
    label = torch.tensor([known_anchor_label], dtype=torch.long, device=device)
    result = observe_frozen_gradient_exchange(
        model,
        image.to(device),
        label,
        nn.CrossEntropyLoss(),
    )
    gradient = result.grad_h_to_g.cpu().numpy().reshape(-1)
    return gradient / (np.linalg.norm(gradient) + 1e-12)


def extract_anchor_matrix(
    model: SplitLearningModel,
    anchor_paths: Sequence[Path],
    image_size: int,
    device: torch.device,
) -> np.ndarray:
    missing = [path for path in anchor_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing class anchors: {[str(path) for path in missing]}")
    return np.stack(
        [
            extract_anchor_gradient(model, load_image(path, image_size), label, device)
            for label, path in enumerate(anchor_paths)
        ]
    )


def map_anchors_to_clusters(anchor_features: np.ndarray, centroids: np.ndarray) -> AnchorMappingResult:
    """Generic one-anchor-per-class assignment using the Hungarian algorithm.

    Anchor row index is the semantic class ID. Each cluster is assigned exactly
    one semantic class, preventing both binary anchors selecting one cluster.
    """
    if anchor_features.ndim != 2 or centroids.ndim != 2:
        raise ValueError("Anchor features and centroids must be 2-D matrices")
    if anchor_features.shape != centroids.shape:
        raise ValueError(
            "One anchor and one centroid are required per class; "
            f"got {anchor_features.shape} and {centroids.shape}"
        )
    anchors = anchor_features / (np.linalg.norm(anchor_features, axis=1, keepdims=True) + 1e-12)
    centers = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)
    similarities = anchors @ centers.T
    distances = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * similarities))
    class_rows, cluster_cols = linear_sum_assignment(distances)
    cluster_to_label = {int(cluster): int(label) for label, cluster in zip(class_rows, cluster_cols)}
    assigned = np.full(len(anchors), -1, dtype=np.int64)
    assigned[class_rows] = cluster_cols
    return AnchorMappingResult(cluster_to_label, distances, similarities, assigned)


def print_anchor_mapping(result: AnchorMappingResult, class_names: list[str]) -> None:
    print("=" * 50)
    print("ANCHOR-BASED CLUSTER IDENTIFICATION")
    print("=" * 50)
    for label, name in enumerate(class_names):
        print(f"\n{name.title()} anchor:")
        for cluster in range(result.distances.shape[1]):
            print(f"    distance -> Cluster {cluster} : {result.distances[label, cluster]:.6f}")
        print(f"    nearest assigned cluster : Cluster {result.assigned_clusters[label]}")
    print("\nFINAL LABEL MAPPING\n")
    for cluster, label in sorted(result.cluster_to_label.items()):
        print(f"    Cluster {cluster} -> {class_names[label].upper()}")
    print("\n" + "=" * 50)
