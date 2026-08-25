from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ...shared.configuration.workspace_paths import DEFAULT_WORKSPACE_PATHS
from ..attacks.gradient_clustering import (
    cluster_normalized_gradients,
    save_cluster_results,
)
from ..attacks.gradient_features import load_smashed_features
from ...shared.data.class_catalog import ClassCatalog
from ...shared.evaluation.visualization import plot_pca


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inference-safe clustering of smashed data (no label or loss gradient)"
    )
    parser.add_argument("--transcripts", default=str(DEFAULT_WORKSPACE_PATHS.transcripts))
    parser.add_argument("--results", default=str(DEFAULT_WORKSPACE_PATHS.reports))
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--k", type=int)
    parser.add_argument("--data", default=str(DEFAULT_WORKSPACE_PATHS.dataset))
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()
    sample_ids, epochs, features = load_smashed_features(args.transcripts, args.epoch)
    k = args.k or ClassCatalog.discover(args.data).num_classes
    clusters, centroids, _ = cluster_normalized_gradients(features, k, args.random_seed)
    results = Path(args.results)
    results.mkdir(parents=True, exist_ok=True)
    save_cluster_results(results / "smashed_clusters.csv", sample_ids, epochs, clusters)
    np.save(results / "smashed_centroids.npy", centroids)
    plot_pca(features, clusters, results / "pca_smashed_clusters.png", "Smashed-data PCA by cluster")
    print(f"Clustered {len(features)} smashed-data inference features without labels or gradients.")


if __name__ == "__main__":
    main()
