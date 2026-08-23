import numpy as np

from src.experiments.attacks.gradient_clustering import cluster_normalized_gradients


def test_unsupervised_clustering_separates_synthetic_features_without_labels():
    rng = np.random.default_rng(9)
    features = np.vstack(
        [rng.normal(loc=-3, scale=0.1, size=(10, 4)), rng.normal(loc=3, scale=0.1, size=(10, 4))]
    )
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    clusters, centroids, _ = cluster_normalized_gradients(features, k=2, random_seed=9)
    assert len(np.unique(clusters[:10])) == 1
    assert len(np.unique(clusters[10:])) == 1
    assert clusters[0] != clusters[-1]
    assert centroids.shape == (2, 4)
