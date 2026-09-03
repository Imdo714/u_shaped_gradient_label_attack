from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from src.shared.data.prepare_augmented_aux_dataset import (
    balanced_label_counts,
    build_variant_plan,
    prepare_dataset,
)


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (24, 18), color).save(path)


def test_balanced_label_counts_assigns_exact_total():
    counts = balanced_label_counts(("cat", "dog", "pug"), 10_000)

    assert counts == {"cat": 3334, "dog": 3333, "pug": 3333}
    assert sum(counts.values()) == 10_000


def test_variant_plan_keeps_each_source_once_before_augmentation(tmp_path):
    sources = {
        "cat": [tmp_path / "cat_1.jpg", tmp_path / "cat_2.jpg"],
        "dog": [tmp_path / "dog_1.jpg", tmp_path / "dog_2.jpg"],
    }

    plan = build_variant_plan(sources, target_total=8, seed=42)

    assert len(plan) == 8
    for label in sources:
        label_plan = [item for item in plan if item.label == label]
        assert [item.identity for item in label_plan] == [True, True, False, False]
        assert [item.variant_index for item in label_plan] == [0, 0, 1, 1]


def test_prepared_dataset_keeps_evaluation_out_of_training(tmp_path):
    source = tmp_path / "source"
    for label, color in (("cat", (200, 10, 10)), ("dog", (10, 200, 10))):
        _write_image(source / "train" / label / f"{label}_train.jpg", color)
        _write_image(source / "val" / label / f"{label}_val.jpg", color)
        _write_image(source / "new_holdout" / label / f"{label}_holdout.jpg", color)
    output = tmp_path / "output"

    prepare_dataset(source, output, 6, image_size=16, seed=7, workers=1)

    assert len(list((output / "train").glob("*/*.jpg"))) == 6
    assert len(list((output / "val").glob("*/*.jpg"))) == 2
    assert len(list((output / "new_holdout").glob("*/*.jpg"))) == 2
    train_names = {path.name for path in (output / "train").glob("*/*.jpg")}
    assert "cat_val.jpg" not in train_names
    assert "dog_holdout.jpg" not in train_names
    assert "10,000 independent source photographs" in (
        output / "DATASET_SOURCE.md"
    ).read_text(encoding="utf-8")


def test_preparation_refuses_to_overwrite_existing_output(tmp_path):
    source = tmp_path / "source"
    _write_image(source / "train" / "cat" / "cat.jpg", (1, 2, 3))
    _write_image(source / "train" / "dog" / "dog.jpg", (4, 5, 6))
    output = tmp_path / "output"
    output.mkdir()

    with pytest.raises(FileExistsError):
        prepare_dataset(source, output, 4, image_size=16, seed=42, workers=1)
