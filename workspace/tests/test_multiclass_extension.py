from pathlib import Path

import numpy as np
import torch

from src.experiments.attacks.anchor_mapping import map_anchors_to_clusters
from src.shared.data.class_catalog import ClassCatalog, checkpoint_class_catalog
from src.shared.evaluation.clustering_metrics import evaluate_clusters
from src.split_learning.architecture.split_learning_model import (
    build_split_learning_model,
    load_split_learning_model,
)


def test_catalog_discovers_three_sorted_classes(tmp_path: Path):
    for name in ("pug", "cat", "dog"):
        (tmp_path / "train" / name).mkdir(parents=True)

    catalog = ClassCatalog.discover(tmp_path)

    assert catalog.names == ("cat", "dog", "pug")
    assert catalog.class_to_idx == {"cat": 0, "dog": 1, "pug": 2}
    assert catalog.num_classes == 3


def test_three_class_checkpoint_round_trip(tmp_path: Path):
    checkpoint = tmp_path / "model.pt"
    model = build_split_learning_model(num_classes=3)
    model.save(checkpoint, class_names=["cat", "dog", "pug"])

    restored, metadata = load_split_learning_model(checkpoint)
    catalog = checkpoint_class_catalog(metadata, restored.h_model.num_classes)

    assert restored.h_model.num_classes == 3
    assert restored.predict(torch.randn(2, 3, 32, 32)).shape == (2, 3)
    assert catalog.names == ("cat", "dog", "pug")


def test_three_class_anchor_mapping_and_macro_metrics():
    centroids = np.eye(3)
    anchors = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    mapping = map_anchors_to_clusters(anchors, centroids).cluster_to_label
    true_labels = np.asarray([0, 1, 2, 0, 1, 2])
    cluster_ids = np.asarray([2, 0, 1, 2, 0, 1])

    metrics, matrix, inferred = evaluate_clusters(
        true_labels, cluster_ids, mapping, num_classes=3
    )

    assert mapping == {2: 0, 0: 1, 1: 2}
    assert matrix.shape == (3, 3)
    assert np.array_equal(inferred, true_labels)
    assert metrics["F1"] == 1.0
