from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ...shared.configuration.workspace_paths import DEFAULT_WORKSPACE_PATHS
from ..attacks.gradient_clustering import (
    cluster_normalized_gradients,
    save_cluster_results,
)
from ..attacks.gradient_features import (
    cosine_similarity_matrix,
    load_gradient_dataset,
)
from ...shared.data.class_catalog import ClassCatalog
from ...shared.evaluation.visualization import plot_pca, plot_similarity_heatmap


def cluster_epoch(transcripts: Path, results: Path, epoch: int, k: int, seed: int = 42):
    data = load_gradient_dataset(transcripts, epoch)
    cluster_ids, centroids, _ = cluster_normalized_gradients(data.normalized, k, seed)
    results.mkdir(parents=True, exist_ok=True)
    suffix = f"_epoch_{epoch:03d}"
    cluster_path = results / f"gradient_clusters{suffix}.csv"
    save_cluster_results(cluster_path, data.sample_ids, data.epochs, cluster_ids)
    save_cluster_results(results / "gradient_clusters.csv", data.sample_ids, data.epochs, cluster_ids)
    np.save(results / f"gradient_centroids{suffix}.npy", centroids)
    similarity = cosine_similarity_matrix(data.normalized)
    np.save(results / "gradient_cosine_similarity_matrix.npy", similarity)
    plot_similarity_heatmap(similarity, results / "gradient_cosine_similarity_heatmap.png")
    plot_pca(data.normalized, cluster_ids, results / "pca_gradient_clusters.png", "Gradient PCA by predicted cluster")
    counts = np.bincount(cluster_ids, minlength=k)
    print(f"Epoch {epoch}: clustered {len(cluster_ids)} gradients; counts={counts.tolist()}")
    return data, cluster_ids, centroids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcripts", default=str(DEFAULT_WORKSPACE_PATHS.transcripts))
    parser.add_argument("--results", default=str(DEFAULT_WORKSPACE_PATHS.reports))
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--k", type=int)
    parser.add_argument("--data", default=str(DEFAULT_WORKSPACE_PATHS.dataset))
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()
    k = args.k or ClassCatalog.discover(args.data).num_classes
    cluster_epoch(Path(args.transcripts), Path(args.results), args.epoch, k, args.random_seed)


if __name__ == "__main__":
    main()
