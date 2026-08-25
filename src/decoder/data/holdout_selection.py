from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from ...shared.data.image_dataset import ImageFolderWithID, validate_class_mapping


@dataclass(frozen=True)
class HoldoutRecord:
    sample_id: str
    split: str
    label: str
    label_index: int
    class_index: int
    source_name: str


def select_holdout_records(
    data_dir: str | Path,
    split: str,
    class_names: tuple[str, ...],
    holdouts_per_class: int,
    labels: Iterable[str] | None = None,
    start_index: int = 0,
) -> list[HoldoutRecord]:
    if holdouts_per_class < 1:
        raise ValueError("holdouts_per_class must be positive")
    if start_index < 0:
        raise ValueError("start_index must be non-negative")
    selected_labels = tuple(labels) if labels is not None else class_names
    unknown = sorted(set(selected_labels) - set(class_names))
    if unknown:
        raise ValueError(f"unknown holdout labels: {unknown}")

    root = Path(data_dir) / split
    dataset = ImageFolderWithID(root)
    validate_class_mapping(dataset, class_names)
    by_label: dict[str, list[int]] = {label: [] for label in selected_labels}
    for dataset_index, (_, label_index) in enumerate(dataset.samples):
        label = class_names[label_index]
        if label in by_label:
            by_label[label].append(dataset_index)

    records: list[HoldoutRecord] = []
    for label in selected_labels:
        indices = by_label[label]
        end = start_index + holdouts_per_class
        if end > len(indices):
            raise ValueError(
                f"{split}/{label} has {len(indices)} images; cannot select "
                f"indices {start_index}..{end - 1}"
            )
        for class_index, dataset_index in enumerate(
            indices[start_index:end], start=start_index
        ):
            path, label_index = dataset.samples[dataset_index]
            records.append(
                HoldoutRecord(
                    sample_id=dataset.sample_id(dataset_index),
                    split=split,
                    label=label,
                    label_index=label_index,
                    class_index=class_index,
                    source_name=Path(path).name,
                )
            )
    return records


def write_holdout_records(records: Iterable[HoldoutRecord], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(record) for record in records]
    if not rows:
        raise ValueError("holdout record list is empty")
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output


def assert_holdouts_excluded(
    manifest_paths: Iterable[str | Path], holdout_sample_ids: set[str]
) -> None:
    for manifest_path in manifest_paths:
        path = Path(manifest_path)
        with path.open(newline="", encoding="utf-8") as handle:
            present = {
                row["sample_id"]
                for row in csv.DictReader(handle)
                if row["sample_id"] in holdout_sample_ids
            }
        if present:
            raise RuntimeError(f"holdout leakage in {path}: {sorted(present)}")


__all__ = [
    "HoldoutRecord",
    "assert_holdouts_excluded",
    "select_holdout_records",
    "write_holdout_records",
]
