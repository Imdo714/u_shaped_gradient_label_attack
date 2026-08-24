from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class LabelPrediction:
    label: int
    probabilities: Tensor
    confidence: float
    cluster_id: int


class GradientLabelPredictor:
    """Assign an observed gradient to a learned cluster and map it to a label."""

    def __init__(
        self,
        centroids: Tensor,
        cluster_to_label: dict[int, int],
        num_classes: int,
        temperature: float = 0.1,
    ) -> None:
        if centroids.ndim != 2:
            raise ValueError("centroids must have shape [clusters, features]")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        expected = set(range(centroids.shape[0]))
        if set(cluster_to_label) != expected:
            raise ValueError(
                f"cluster mapping must cover {sorted(expected)}, got {sorted(cluster_to_label)}"
            )
        labels = set(cluster_to_label.values())
        if not labels.issubset(set(range(num_classes))):
            raise ValueError("cluster mapping contains a label outside num_classes")
        self.centroids = centroids.float()
        self.cluster_to_label = cluster_to_label
        self.num_classes = num_classes
        self.temperature = temperature

    @classmethod
    def from_files(
        cls,
        centroid_path: str | Path,
        mapping_path: str | Path,
        num_classes: int,
        temperature: float = 0.1,
    ) -> "GradientLabelPredictor":
        centroids = torch.from_numpy(np.load(centroid_path)).float()
        with Path(mapping_path).open("r", encoding="utf-8") as handle:
            raw_mapping = json.load(handle)
        mapping = {int(cluster): int(label) for cluster, label in raw_mapping.items()}
        return cls(centroids, mapping, num_classes, temperature)

    def predict(self, gradient: Tensor, soft: bool = False) -> LabelPrediction:
        feature = gradient.detach().float().flatten().cpu()
        if feature.numel() != self.centroids.shape[1]:
            raise ValueError(
                f"gradient has {feature.numel()} features, centroids expect "
                f"{self.centroids.shape[1]}"
            )
        feature = feature / feature.norm().clamp_min(1e-12)
        distances = torch.cdist(feature.unsqueeze(0), self.centroids).squeeze(0)
        cluster_probabilities = torch.softmax(-distances / self.temperature, dim=0)
        label_probabilities = torch.zeros(self.num_classes)
        for cluster_id, probability in enumerate(cluster_probabilities):
            label_probabilities[self.cluster_to_label[cluster_id]] += probability
        cluster_id = int(distances.argmin())
        label = self.cluster_to_label[cluster_id]
        if not soft:
            label_probabilities.zero_()
            label_probabilities[label] = 1.0
        return LabelPrediction(
            label=label,
            probabilities=label_probabilities,
            confidence=float(label_probabilities[label]),
            cluster_id=cluster_id,
        )


__all__ = ["GradientLabelPredictor", "LabelPrediction"]
