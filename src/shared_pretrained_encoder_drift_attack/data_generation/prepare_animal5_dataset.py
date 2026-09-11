from __future__ import annotations

import argparse
import csv
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ...shared.data.prepare_public_dataset import (
    RAW_ROOT,
    belongs_to_breed,
    download_one,
    load_public_image_paths,
)


DEFAULT_CLASSES: dict[str, tuple[str, ...]] = {
    "beagle": ("beagle",),
    "bengal": ("Bengal",),
    "pug": ("pug",),
    "samoyed": ("samoyed",),
    "siamese": ("Siamese",),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare five disjoint Oxford-IIIT Pet breed datasets for autoencoder "
            "pretraining, malicious-client fine-tuning, victim fine-tuning, and "
            "a balanced 100-image victim holdout."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("workspace/data/shared_pretrained_encoder_joint_attack/animal5"),
    )
    parser.add_argument("--pretrain-train-per-class", type=int, default=50)
    parser.add_argument("--pretrain-val-per-class", type=int, default=10)
    parser.add_argument("--attacker-train-per-class", type=int, default=40)
    parser.add_argument("--attacker-val-per-class", type=int, default=10)
    parser.add_argument("--victim-train-per-class", type=int, default=40)
    parser.add_argument("--victim-val-per-class", type=int, default=10)
    parser.add_argument("--holdout-per-class", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=8)
    return parser


def _split_counts(args: argparse.Namespace) -> tuple[tuple[str, str, int], ...]:
    return (
        ("pretrain", "train", args.pretrain_train_per_class),
        ("pretrain", "val", args.pretrain_val_per_class),
        ("attacker", "train", args.attacker_train_per_class),
        ("attacker", "val", args.attacker_val_per_class),
        ("victim", "train", args.victim_train_per_class),
        ("victim", "val", args.victim_val_per_class),
        ("victim", "new_holdout", args.holdout_per_class),
    )


def partition_source_paths(
    paths: list[str],
    classes: dict[str, tuple[str, ...]],
    splits: tuple[tuple[str, str, int], ...],
    seed: int,
) -> list[dict[str, str]]:
    if any(count < 1 for _, _, count in splits):
        raise ValueError("all per-class split counts must be positive")
    rows: list[dict[str, str]] = []
    required = sum(count for _, _, count in splits)
    for label, breeds in classes.items():
        candidates = sorted(path for path in paths if belongs_to_breed(path, breeds))
        random.Random(f"{seed}:animal5:{label}").shuffle(candidates)
        if len(candidates) < required:
            raise RuntimeError(
                f"not enough {label} images: need {required}, found {len(candidates)}"
            )
        offset = 0
        for domain, split, count in splits:
            for source_file in candidates[offset : offset + count]:
                rows.append(
                    {
                        "domain": domain,
                        "split": split,
                        "label": label,
                        "source_file": source_file,
                        "source_url": f"{RAW_ROOT}/{source_file}",
                        "destination": f"{domain}/{split}/{label}/{Path(source_file).name}",
                    }
                )
            offset += count
    return rows


def _download_to_staging(
    project_root: Path,
    staging_root: Path,
    row: dict[str, str],
) -> dict[str, str]:
    staging_destination = staging_root / row["destination"]
    download_row = {
        "source_file": row["source_file"],
        "source_url": row["source_url"],
        "split": row["split"],
        "label": row["label"],
        "destination": staging_destination.relative_to(project_root).as_posix(),
    }
    completed = download_one(project_root, download_row, force=False)
    return {
        **row,
        "width": completed["width"],
        "height": completed["height"],
        "bytes": completed["bytes"],
        "sha256": completed["sha256"],
    }


def prepare(args: argparse.Namespace) -> Path:
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.holdout_per_class != 20:
        raise ValueError(
            "this experiment requires exactly 20 holdout images per class (100 total)"
        )
    project_root = Path(__file__).resolve().parents[3]
    output = args.output if args.output.is_absolute() else project_root / args.output
    output = output.resolve()
    try:
        output.relative_to(project_root)
    except ValueError as error:
        raise ValueError("--output must be inside the project workspace") from error
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    staging = output.parent / f".{output.name}.staging"
    if staging.exists():
        raise FileExistsError(f"staging directory already exists: {staging}")
    staging.mkdir(parents=True)

    splits = _split_counts(args)
    rows = partition_source_paths(
        load_public_image_paths(), DEFAULT_CLASSES, splits, args.seed
    )
    completed: list[dict[str, str]] = []
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(_download_to_staging, project_root, staging, row)
                for row in rows
            ]
            for index, future in enumerate(as_completed(futures), start=1):
                completed.append(future.result())
                if index % 50 == 0 or index == len(futures):
                    print(f"Downloaded and verified: {index}/{len(futures)}", flush=True)

        completed.sort(
            key=lambda row: (
                row["domain"], row["split"], row["label"], row["source_file"]
            )
        )
        with (staging / "dataset_manifest.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(completed[0]))
            writer.writeheader()
            writer.writerows(completed)

        holdout_rows = [
            row
            for row in completed
            if row["domain"] == "victim" and row["split"] == "new_holdout"
        ]
        if len(holdout_rows) != 100:
            raise RuntimeError(f"expected 100 holdouts, produced {len(holdout_rows)}")
        holdout_counts = {
            label: sum(row["label"] == label for row in holdout_rows)
            for label in DEFAULT_CLASSES
        }
        if set(holdout_counts.values()) != {20}:
            raise RuntimeError(f"unbalanced holdout: {holdout_counts}")
        if len({row["sha256"] for row in completed}) != len(completed):
            raise RuntimeError("the prepared partitions contain duplicate image bytes")

        lines = [
            "# Five-class animal dataset",
            "",
            "Source: Oxford-IIIT Pet images mirrored by ml4py/dataset-iiit-pet.",
            f"Classes: {', '.join(DEFAULT_CLASSES)}",
            f"Seed: {args.seed}",
            "",
            "All pretraining, attacker, victim, and holdout source images are disjoint.",
            "The victim new_holdout split contains exactly 20 images per class (100 total).",
            "",
        ]
        (staging / "DATASET_SOURCE.md").write_text(
            "\n".join(lines), encoding="utf-8"
        )
        staging.replace(output)
    except Exception:
        print(f"Preparation failed; partial files remain in {staging}")
        raise

    print("\nFive-class animal dataset ready")
    print(f"  output: {output}")
    for domain, split, count in splits:
        print(f"  {domain}/{split}: {count} per class ({count * 5} total)")
    return output


def main() -> None:
    prepare(build_parser().parse_args())


if __name__ == "__main__":
    main()


__all__ = ["DEFAULT_CLASSES", "build_parser", "partition_source_paths", "prepare"]
