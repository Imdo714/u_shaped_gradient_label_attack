import numpy as np

from src.experiments.attacks.anchor_mapping import map_anchors_to_clusters


def test_hungarian_anchor_mapping_recovers_semantics():
    # Cluster IDs are intentionally opposite semantic class row IDs.
    centroids = np.asarray([[0.0, 1.0], [1.0, 0.0]])
    cat_anchor = np.asarray([0.99, 0.01])
    dog_anchor = np.asarray([0.02, 0.98])
    result = map_anchors_to_clusters(np.stack([cat_anchor, dog_anchor]), centroids)
    assert result.cluster_to_label == {1: 0, 0: 1}
    assert result.assigned_clusters.tolist() == [1, 0]
