import numpy as np

from src.experiments.attacks.gradient_features import cosine_similarity, normalized_euclidean_distance


def test_normalized_cosine_euclidean_identity():
    rng = np.random.default_rng(7)
    for _ in range(20):
        x, y = rng.normal(size=128), rng.normal(size=128)
        cosine = cosine_similarity(x, y)
        distance = normalized_euclidean_distance(x, y)
        assert np.isclose(distance**2, 2.0 - 2.0 * cosine, atol=1e-10)
