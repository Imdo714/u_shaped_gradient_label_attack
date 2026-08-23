from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def _ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_pca(features: np.ndarray, colors: np.ndarray, path: str | Path, title: str, labels=None) -> None:
    path = _ensure_parent(path)
    if features.shape[0] < 2:
        return
    points = PCA(n_components=2).fit_transform(features)
    plt.figure(figsize=(7, 6))
    for value in np.unique(colors):
        mask = colors == value
        name = labels[int(value)] if labels is not None else f"Cluster {value}"
        plt.scatter(points[mask, 0], points[mask, 1], s=18, alpha=0.75, label=name)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_similarity_heatmap(matrix: np.ndarray, path: str | Path) -> None:
    path = _ensure_parent(path)
    plt.figure(figsize=(7, 6))
    plt.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(label="Cosine similarity")
    plt.title("Server-observed gradient cosine similarity")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_confusion(
    matrix: np.ndarray, path: str | Path, class_names: tuple[str, ...]
) -> None:
    path = _ensure_parent(path)
    plt.figure(figsize=(5, 4))
    plt.imshow(matrix, cmap="Blues")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, int(matrix[i, j]), ha="center", va="center")
    plt.xticks(range(len(class_names)), class_names)
    plt.yticks(range(len(class_names)), class_names)
    plt.xlabel("Inferred label")
    plt.ylabel("True label")
    plt.title("Anchor-mapped attack confusion matrix")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_epoch_f1(rows: list[dict], path: str | Path) -> None:
    path = _ensure_parent(path)
    plt.figure(figsize=(7, 4))
    plt.plot([r["epoch"] for r in rows], [r["F1"] for r in rows], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Attack F1")
    plt.ylim(0, 1.02)
    plt.title("Gradient label-inference F1 by epoch")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
