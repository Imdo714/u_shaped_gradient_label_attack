from __future__ import annotations

import torch

from src.shared_pretrained_encoder_drift_attack.pipeline.run_online_transcript_label_attack import (
    OnlineLabelClassifier,
    OnlineLabelClassifierConfig,
    TranscriptAccumulator,
    _temporal_bundles,
)


def _bundle():
    accumulator = TranscriptAccumulator()
    for round_index in (1, 5, 10, 20):
        labels = torch.tensor([0, 0, 1, 1])
        accumulator.add(
            torch.randn(4, 6, 4, 4),
            torch.randn(4, 3, 8, 8),
            labels,
            ("a", "b", "c", "d"),
            round_index,
            round_index * 10,
        )
    return accumulator.bundle()


def test_temporal_bundles_keep_latest_and_match_budget() -> None:
    bundle = _bundle()
    outputs = _temporal_bundles(
        bundle,
        validation_ids={"b", "d"},
        collection_rounds=(1, 5, 10, 20),
        include_matched=True,
        seed=42,
    )

    assert set(outputs) == {"single_latest", "multi_natural", "multi_matched"}
    latest_train, latest_validation, _ = outputs["single_latest"]
    natural_train, natural_validation, _ = outputs["multi_natural"]
    matched_train, matched_validation, _ = outputs["multi_matched"]
    assert len(latest_train) == 2
    assert len(latest_validation) == 2
    assert len(natural_train) == 8
    assert len(natural_validation) == 8
    assert len(matched_train) == len(latest_train)
    assert len(matched_validation) == len(natural_validation)
    assert set(matched_train.rounds.tolist()).issubset({1, 5, 10, 20})


def test_online_classifier_supports_all_signal_modes() -> None:
    u = torch.randn(3, 6, 4, 4)
    grad_z = torch.randn(3, 3, 8, 8)
    for signal_mode in ("u_only", "gradient_only", "u_gradient"):
        model = OnlineLabelClassifier(
            OnlineLabelClassifierConfig(
                u_channels=6,
                grad_z_channels=3,
                num_classes=5,
                signal_mode=signal_mode,
                signal_spatial_size=4,
                signal_channels=8,
                hidden_channels=16,
            )
        )
        assert model(u, grad_z).shape == (3, 5)
