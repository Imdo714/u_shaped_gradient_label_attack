from __future__ import annotations

from src.shared_pretrained_encoder_drift_attack.pipeline.run_label_inference_improvements import (
    DYNAMIC_CONDITIONS,
    DYNAMIC_VARIANTS,
    augmentation_subsets,
    factorial_conditions,
)


def test_factorial_ablation_covers_all_snapshot_and_augmentation_cases() -> None:
    subsets = augmentation_subsets()
    assert subsets == (
        (),
        ("crop",),
        ("flip",),
        ("color",),
        ("crop", "flip"),
        ("crop", "color"),
        ("flip", "color"),
        ("crop", "flip", "color"),
    )

    conditions = factorial_conditions((0, 1, 5, 10, 20))
    assert len(conditions) == 16
    assert len({condition.variant for condition in conditions}) == 16
    for subset in subsets:
        matching = [
            condition for condition in conditions if condition.augmentations == subset
        ]
        assert {condition.snapshot_mode for condition in matching} == {
            "single_snapshot",
            "multi_snapshot",
        }
        single = next(
            condition
            for condition in matching
            if condition.snapshot_mode == "single_snapshot"
        )
        multi = next(
            condition
            for condition in matching
            if condition.snapshot_mode == "multi_snapshot"
        )
        assert single.snapshot_epochs == (20,)
        assert multi.snapshot_epochs == (0, 1, 5, 10, 20)


def test_controlled_dynamic_ablation_has_three_distinct_schedules() -> None:
    assert tuple(condition.variant for condition in DYNAMIC_CONDITIONS) == (
        DYNAMIC_VARIANTS
    )
    assert {condition.augmentation_schedule for condition in DYNAMIC_CONDITIONS} == {
        "none",
        "dynamic_shared",
        "dynamic_independent",
    }
