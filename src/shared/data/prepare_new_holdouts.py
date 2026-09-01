from __future__ import annotations

import argparse
import csv
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ..configuration.workspace_paths import DEFAULT_WORKSPACE_PATHS
from .prepare_public_dataset import (
    RAW_ROOT,
    belongs_to_breed,
    download_one,
    load_class_definitions,
    load_public_image_paths,
)


def balanced_label_counts(labels: tuple[str, ...], total: int) -> dict[str, int]:
    if total < 1:
        raise ValueError("total holdout count must be positive")
    base, extra = divmod(total, len(labels))
    return {
        label: base + int(index < extra)
        for index, label in enumerate(labels)
    }


def select_new_holdout_paths(
    paths: list[str],
    definitions: dict[str, tuple[str, ...]],
    excluded_source_files: set[str],
    total: int,
    seed: int,
) -> dict[str, list[str]]:
    counts = balanced_label_counts(tuple(definitions), total)
    selected: dict[str, list[str]] = {}
    for label, breeds in definitions.items():
        candidates = sorted(
            path
            for path in paths
            if path not in excluded_source_files and belongs_to_breed(path, breeds)
        )
        random.Random(f"{seed}:new-holdout:{label}").shuffle(candidates)
        if len(candidates) < counts[label]:
            raise RuntimeError(
                f"not enough unseen {label} images: need {counts[label]}, "
                f"found {len(candidates)}"
            )
        selected[label] = candidates[: counts[label]]
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download holdout images that do not overlap the tracked public subset."
    )
    parser.add_argument(
        "--project-root", default=Path(__file__).resolve().parents[3], type=Path
    )
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Download only these configured classes.",
    )
    parser.add_argument("--seed", type=int, default=142)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    definitions = load_class_definitions(
        project_root / DEFAULT_WORKSPACE_PATHS.class_config
    )
    requested_labels = tuple(args.labels) if args.labels else tuple(definitions)
    unknown = sorted(set(requested_labels) - set(definitions))
    if unknown:
        parser.error(f"unknown labels: {unknown}; configured: {list(definitions)}")
    requested_definitions = {
        label: definitions[label] for label in requested_labels
    }
    public_manifest = (
        project_root / DEFAULT_WORKSPACE_PATHS.dataset / "public_subset_manifest.csv"
    )
    with public_manifest.open(newline="", encoding="utf-8") as handle:
        excluded = {row["source_file"] for row in csv.DictReader(handle)}

    selected = select_new_holdout_paths(
        load_public_image_paths(), requested_definitions, excluded, args.count, args.seed
    )
    rows: list[dict[str, str]] = []
    output_root = project_root / DEFAULT_WORKSPACE_PATHS.dataset / "new_holdout"
    for label in definitions:
        (output_root / label).mkdir(parents=True, exist_ok=True)
    for label, source_files in selected.items():
        for source_file in source_files:
            destination = output_root / label / Path(source_file).name
            rows.append(
                {
                    "source_file": source_file,
                    "source_url": f"{RAW_ROOT}/{source_file}",
                    "split": "new_holdout",
                    "label": label,
                    "destination": destination.relative_to(project_root).as_posix(),
                }
            )

    completed: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(download_one, project_root, row, args.force)
            for row in rows
        ]
        for index, future in enumerate(as_completed(futures), start=1):
            completed.append(future.result())
            print(f"Verified: {index}/{len(futures)}")

    completed.sort(key=lambda row: (row["label"], row["destination"]))
    manifest = (
        project_root
        / DEFAULT_WORKSPACE_PATHS.results_root
        / "decoder"
        / "new_holdout_manifest.csv"
    )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(completed[0]))
        writer.writeheader()
        writer.writerows(completed)

    print("\nNew holdout dataset ready")
    for label in requested_definitions:
        count = sum(row["label"] == label for row in completed)
        print(f"  {label}: {count}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()


__all__ = ["balanced_label_counts", "select_new_holdout_paths"]
