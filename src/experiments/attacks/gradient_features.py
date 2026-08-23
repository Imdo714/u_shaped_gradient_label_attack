from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


EPSILON = 1e-12


@dataclass
class GradientDataset:
    sample_ids: list[str]
    epochs: np.ndarray
    raw: np.ndarray
    normalized: np.ndarray


def flatten_and_normalize(gradient: np.ndarray, epsilon: float = EPSILON) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(gradient, dtype=np.float32).reshape(-1)
    normalized = raw / (np.linalg.norm(raw) + epsilon)
    return raw, normalized


def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    x_hat = x.reshape(-1) / (np.linalg.norm(x) + EPSILON)
    y_hat = y.reshape(-1) / (np.linalg.norm(y) + EPSILON)
    return float(np.dot(x_hat, y_hat))


def normalized_euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    x_hat = x.reshape(-1) / (np.linalg.norm(x) + EPSILON)
    y_hat = y.reshape(-1) / (np.linalg.norm(y) + EPSILON)
    return float(np.linalg.norm(x_hat - y_hat))


def cosine_similarity_matrix(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    normalized = features / (norms + EPSILON)
    return normalized @ normalized.T


def load_gradient_dataset(transcript_root: str | Path, epoch: int | None = None) -> GradientDataset:
    """Load attacker records. No evaluator path or labels are accepted by this API."""
    root = Path(transcript_root)
    attacker_root = root if root.name == "attacker_transcript" else root / "attacker_transcript"
    index_path = attacker_root / "index.csv"
    sample_ids: list[str] = []
    epochs: list[int] = []
    raw_features: list[np.ndarray] = []
    normalized_features: list[np.ndarray] = []
    with index_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row_epoch = int(row["epoch"])
            if epoch is not None and row_epoch != epoch:
                continue
            record = np.load(attacker_root / row["record_file"])
            raw, normalized = flatten_and_normalize(record["grad_h_to_g"])
            sample_ids.append(row["sample_id"])
            epochs.append(row_epoch)
            raw_features.append(raw)
            normalized_features.append(normalized)
    if not raw_features:
        raise ValueError(f"No transcript gradients found for epoch={epoch}")
    return GradientDataset(
        sample_ids,
        np.asarray(epochs, dtype=np.int64),
        np.stack(raw_features),
        np.stack(normalized_features),
    )


def load_smashed_features(transcript_root: str | Path, epoch: int | None = None):
    """Label-free feature path for inference_smashed mode."""
    root = Path(transcript_root)
    attacker_root = root if root.name == "attacker_transcript" else root / "attacker_transcript"
    sample_ids, epochs, features = [], [], []
    with (attacker_root / "index.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row_epoch = int(row["epoch"])
            if epoch is not None and row_epoch != epoch:
                continue
            data = np.load(attacker_root / row["record_file"])["smashed_z"].reshape(-1)
            _, normalized = flatten_and_normalize(data)
            sample_ids.append(row["sample_id"])
            epochs.append(row_epoch)
            features.append(normalized)
    return sample_ids, np.asarray(epochs), np.stack(features)
