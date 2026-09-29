from __future__ import annotations

from pathlib import Path

import torch

from src.shared_pretrained_encoder_drift_attack.paper_label_benchmark import (
    IndependentRandomHorizontalFlip,
)
from src.shared_pretrained_encoder_drift_attack.pipeline.run_paper_label_augmentation_ablation import (
    build_conditions,
    build_parser,
)
from src.shared_pretrained_encoder_drift_attack.pipeline.run_paper_label_condition import (
    build_parser as build_condition_parser,
)


def test_default_ablation_has_four_paired_worlds_and_five_seeds(
    tmp_path: Path,
) -> None:
    args = build_parser().parse_args(["--output", str(tmp_path)])
    conditions = build_conditions(args, tmp_path)
    assert len(conditions) == 4 * 5
    assert {
        (condition.victim_augmentation, condition.auxiliary_augmentation)
        for condition in conditions
    } == {
        ("none", "none"),
        ("none", "dynamic_flip"),
        ("dynamic_flip", "none"),
        ("dynamic_flip", "dynamic_flip"),
    }
    assert {condition.seed for condition in conditions} == {42, 43, 44, 45, 46}


def test_controlled_flip_does_not_advance_global_torch_rng() -> None:
    transform = IndependentRandomHorizontalFlip(0.5, 123)
    image = torch.arange(12).reshape(1, 3, 4)
    torch.manual_seed(987)
    before = torch.random.get_rng_state().clone()
    transform(image)
    after = torch.random.get_rng_state()
    assert torch.equal(before, after)


def test_condition_parser_accepts_independent_augmentation_modes() -> None:
    args = build_condition_parser().parse_args(
        [
            "--dataset",
            "animal5",
            "--data-root",
            "data",
            "--output",
            "output",
            "--split-level",
            "7",
            "--aux-fraction",
            "0.05",
            "--seed",
            "42",
            "--victim-augmentation",
            "none",
            "--auxiliary-augmentation",
            "dynamic_flip",
        ]
    )
    assert args.victim_augmentation == "none"
    assert args.auxiliary_augmentation == "dynamic_flip"
