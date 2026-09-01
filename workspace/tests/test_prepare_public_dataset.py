from __future__ import annotations

from src.shared.data.prepare_public_dataset import (
    destination_plan,
    prune_existing_files,
)
from src.shared.data.prepare_new_holdouts import (
    balanced_label_counts,
    select_new_holdout_paths,
)


def test_destination_plan_uses_split_specific_counts_and_one_anchor(tmp_path):
    selected = {"cat": [f"cat_{index}.jpg" for index in range(7)]}
    plan = destination_plan(
        tmp_path, selected, train_count=3, val_count=2, test_count=1
    )
    counts = {
        split: sum(row["split"] == split for row in plan)
        for split in ("train", "val", "test", "anchor")
    }
    assert counts == {"train": 3, "val": 2, "test": 1, "anchor": 1}
    assert next(row for row in plan if row["split"] == "anchor")["source_file"] == "cat_6.jpg"


def test_prune_existing_removes_only_obsolete_requested_class_files(tmp_path):
    keep = tmp_path / "workspace/data/dataset/train/cat/keep.jpg"
    obsolete = tmp_path / "workspace/data/dataset/train/cat/obsolete.jpg"
    unrelated = tmp_path / "workspace/data/dataset/train/dog/unrelated.jpg"
    for path in (keep, obsolete, unrelated):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test")
    plan = [
        {
            "source_file": "keep.jpg",
            "source_url": "unused",
            "split": "train",
            "label": "cat",
            "destination": "workspace/data/dataset/train/cat/keep.jpg",
        }
    ]

    removed = prune_existing_files(tmp_path, plan, ("cat",))

    assert removed == 1
    assert keep.is_file()
    assert not obsolete.exists()
    assert unrelated.is_file()


def test_new_holdouts_are_balanced_and_exclude_existing_sources():
    paths = [
        *(f"Cat_{index}.jpg" for index in range(6)),
        *(f"Dog_{index}.jpg" for index in range(6)),
    ]
    selected = select_new_holdout_paths(
        paths,
        {"cat": ("Cat",), "dog": ("Dog",)},
        {"Cat_0.jpg", "Dog_0.jpg"},
        total=5,
        seed=42,
    )

    assert balanced_label_counts(("cat", "dog"), 5) == {"cat": 3, "dog": 2}
    assert {label: len(items) for label, items in selected.items()} == {
        "cat": 3,
        "dog": 2,
    }
    assert not ({"Cat_0.jpg", "Dog_0.jpg"} & set().union(*selected.values()))
