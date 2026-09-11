from __future__ import annotations

import torch

from src.shared_pretrained_encoder_drift_attack.data_generation.prepare_animal5_dataset import (
    DEFAULT_CLASSES,
    partition_source_paths,
)
from src.shared_pretrained_encoder_drift_attack.models import (
    DriftRobustLabelClassifier,
    DriftRobustLabelClassifierConfig,
    ReconstructionDecoder,
    ReconstructionDecoderConfig,
    label_classifier_for_condition,
)


def test_animal5_partition_is_disjoint_and_balanced() -> None:
    paths = [
        f"{breeds[0]}_{index}.jpg"
        for breeds in DEFAULT_CLASSES.values()
        for index in range(1, 201)
    ]
    splits = (
        ("pretrain", "train", 50),
        ("pretrain", "val", 10),
        ("attacker", "train", 40),
        ("attacker", "val", 10),
        ("victim", "train", 40),
        ("victim", "val", 10),
        ("victim", "new_holdout", 20),
    )
    rows = partition_source_paths(paths, DEFAULT_CLASSES, splits, seed=2026)

    assert len(rows) == 900
    assert len({row["source_file"] for row in rows}) == 900
    for label in DEFAULT_CLASSES:
        holdouts = [
            row
            for row in rows
            if row["label"] == label and row["split"] == "new_holdout"
        ]
        assert len(holdouts) == 20


def test_joint_u_gradient_models_accept_different_signal_shapes() -> None:
    u = torch.randn(2, 7, 8, 8)
    grad_z = torch.randn(2, 5, 16, 16)
    decoder = ReconstructionDecoder(
        ReconstructionDecoderConfig(
            z_channels=7,
            grad_channels=5,
            image_size=32,
            use_z=True,
            use_gradient=True,
            signal_spatial_size=8,
            signal_channels=8,
            base_channels=32,
            min_channels=8,
        )
    )
    classifier = label_classifier_for_condition(
        "u_grad_z",
        u_channels=7,
        grad_z_channels=5,
        num_classes=5,
        signal_spatial_size=4,
        signal_channels=8,
        hidden_channels=16,
    )

    assert decoder(u, grad_z).shape == (2, 3, 32, 32)
    assert classifier(u, grad_z).shape == (2, 5)


def test_drift_robust_classifier_preserves_norm_and_gates_u() -> None:
    u = torch.randn(3, 7, 8, 8)
    grad_z = torch.randn(3, 5, 16, 16)
    classifier = DriftRobustLabelClassifier(
        DriftRobustLabelClassifierConfig(
            u_channels=7,
            grad_z_channels=5,
            num_classes=5,
            use_u=True,
            use_gradient_norm=True,
            signal_spatial_size=4,
            signal_channels=8,
            hidden_channels=16,
            norm_channels=4,
        )
    )

    logits, gate = classifier.forward_with_gate(u, grad_z)
    assert logits.shape == (3, 5)
    assert gate is not None and gate.shape == (3, 1)
    assert torch.all((gate >= 0) & (gate <= 1))
    assert classifier.gradient_log_norm(grad_z).shape == (3, 1)
