from __future__ import annotations

import torch

from src.shared_pretrained_encoder_drift_attack.pipeline.run_online_z_attack import (
    ZLabelClassifier,
    ZLabelClassifierConfig,
    ZTranscriptAccumulator,
    _evaluate_z_prototype,
    _temporal_z_bundles,
)
from src.shared.data.class_catalog import ClassCatalog


def _bundle():
    accumulator = ZTranscriptAccumulator()
    for round_index in (1, 5, 10, 20):
        labels = torch.tensor([0, 0, 1, 1])
        z = torch.stack(
            (
                torch.full((2, 4, 4, 4), -1.0),
                torch.full((2, 4, 4, 4), 1.0),
            )
        ).reshape(4, 4, 4, 4)
        accumulator.add(
            z,
            labels,
            ("a", "b", "c", "d"),
            round_index,
            round_index * 10,
        )
    return accumulator.bundle()


def test_z_temporal_bundles_match_existing_window_budgets() -> None:
    outputs = _temporal_z_bundles(
        _bundle(),
        validation_ids={"b", "d"},
        collection_rounds=(1, 5, 10, 20),
        include_matched=True,
        seed=42,
    )
    latest_train, latest_validation, _ = outputs["single_latest"]
    natural_train, natural_validation, _ = outputs["multi_natural"]
    matched_train, matched_validation, _ = outputs["multi_matched"]
    assert len(latest_train) == 2
    assert len(latest_validation) == 2
    assert len(natural_train) == 8
    assert len(natural_validation) == 8
    assert len(matched_train) == 2
    assert len(matched_validation) == 8


def test_learned_cpsi_accepts_z_only() -> None:
    model = ZLabelClassifier(
        ZLabelClassifierConfig(
            z_channels=4,
            num_classes=5,
            signal_spatial_size=4,
            signal_channels=8,
            hidden_channels=16,
        )
    )
    assert model(torch.randn(3, 4, 8, 8)).shape == (3, 5)


def test_z_cosine_prototype_uses_class_centroids() -> None:
    bundle = _bundle().select(torch.tensor([0, 2]))
    catalog = ClassCatalog(("negative", "positive"))
    summary, _, matrix = _evaluate_z_prototype(bundle, bundle, catalog)
    assert summary["accuracy"] == 1.0
    assert matrix.tolist() == [[1, 0], [0, 1]]
