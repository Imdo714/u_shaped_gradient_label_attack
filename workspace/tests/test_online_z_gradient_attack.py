from __future__ import annotations

import torch

from src.shared.data.class_catalog import ClassCatalog
from src.shared_pretrained_encoder_drift_attack.pipeline.run_online_z_gradient_attack import (
    ZGradientLabelClassifier,
    ZGradientLabelClassifierConfig,
    ZGradientTranscriptAccumulator,
    _evaluate_prototype,
    _temporal_bundles,
)


def _bundle():
    accumulator = ZGradientTranscriptAccumulator()
    for round_index in (1, 5, 10, 20):
        labels = torch.tensor([0, 0, 1, 1])
        z = torch.cat(
            (
                torch.full((2, 4, 4, 4), -1.0),
                torch.full((2, 4, 4, 4), 1.0),
            )
        )
        grad_z = z * 0.1
        accumulator.add(
            z,
            grad_z,
            labels,
            ("a", "b", "c", "d"),
            round_index,
            round_index * 10,
        )
    return accumulator.bundle()


def test_z_gradient_temporal_bundles_match_window_budgets() -> None:
    outputs = _temporal_bundles(
        _bundle(),
        validation_ids={"b", "d"},
        collection_rounds=(1, 5, 10, 20),
        include_matched=True,
        seed=42,
    )
    assert len(outputs["single_latest"][0]) == 2
    assert len(outputs["multi_natural"][0]) == 8
    assert len(outputs["multi_matched"][0]) == 2
    assert len(outputs["multi_natural"][1]) == 8


def test_learned_cpsi_accepts_z_and_gradient() -> None:
    model = ZGradientLabelClassifier(
        ZGradientLabelClassifierConfig(
            z_channels=4,
            grad_z_channels=4,
            num_classes=5,
            signal_spatial_size=4,
            signal_channels=8,
            hidden_channels=16,
            norm_channels=4,
        )
    )
    assert model(torch.randn(3, 4, 8, 8), torch.randn(3, 4, 8, 8)).shape == (3, 5)


def test_z_gradient_cosine_prototype_uses_class_centroids() -> None:
    bundle = _bundle().select(torch.tensor([0, 2]))
    catalog = ClassCatalog(("negative", "positive"))
    summary, _, matrix = _evaluate_prototype(bundle, bundle, catalog)
    assert summary["accuracy"] == 1.0
    assert matrix.tolist() == [[1, 0], [0, 1]]
