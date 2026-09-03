from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import random
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageEnhance, ImageOps


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EVALUATION_SPLITS = ("val", "test", "new_holdout")


@dataclass(frozen=True)
class VariantPlan:
    label: str
    output_index: int
    source: Path
    source_index: int
    variant_index: int
    identity: bool
    seed: int


def balanced_label_counts(labels: tuple[str, ...], total: int) -> dict[str, int]:
    if not labels:
        raise ValueError("at least one label is required")
    if total < len(labels):
        raise ValueError("target count must be at least the number of labels")
    base, remainder = divmod(total, len(labels))
    return {
        label: base + int(index < remainder)
        for index, label in enumerate(labels)
    }


def image_paths(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _stable_seed(seed: int, label: str, source_name: str, variant_index: int) -> int:
    key = f"{seed}:{label}:{source_name}:{variant_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")


def build_variant_plan(
    sources_by_label: dict[str, list[Path]],
    target_total: int,
    seed: int,
) -> list[VariantPlan]:
    labels = tuple(sorted(sources_by_label))
    targets = balanced_label_counts(labels, target_total)
    plan: list[VariantPlan] = []
    for label in labels:
        sources = sorted(sources_by_label[label])
        if not sources:
            raise ValueError(f"no training images found for label {label!r}")
        if targets[label] < len(sources):
            raise ValueError(
                f"target for {label!r} ({targets[label]}) is smaller than its "
                f"source count ({len(sources)}); this tool never discards source images"
            )
        for output_index in range(targets[label]):
            source_index = output_index % len(sources)
            variant_index = output_index // len(sources)
            source = sources[source_index]
            plan.append(
                VariantPlan(
                    label=label,
                    output_index=output_index,
                    source=source,
                    source_index=source_index,
                    variant_index=variant_index,
                    identity=variant_index == 0,
                    seed=_stable_seed(seed, label, source.name, variant_index),
                )
            )
    return plan


def _resample():
    return getattr(Image, "Resampling", Image).LANCZOS


def _transpose_method():
    return getattr(Image, "Transpose", Image).FLIP_LEFT_RIGHT


def _make_image(source: Path, image_size: int, identity: bool, seed: int) -> tuple[Image.Image, dict[str, str]]:
    with Image.open(source) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")

    if identity:
        output = image.resize((image_size, image_size), _resample())
        return output, {
            "crop_left": "0",
            "crop_top": "0",
            "crop_width": str(image.width),
            "crop_height": str(image.height),
            "horizontal_flip": "false",
            "brightness": "1.000000",
            "contrast": "1.000000",
            "saturation": "1.000000",
        }

    rng = random.Random(seed)
    area_scale = rng.uniform(0.72, 0.98)
    aspect_ratio = rng.uniform(0.90, 1.10)
    crop_width = min(image.width, max(1, round(image.width * math.sqrt(area_scale * aspect_ratio))))
    crop_height = min(image.height, max(1, round(image.height * math.sqrt(area_scale / aspect_ratio))))
    crop_left = rng.randint(0, image.width - crop_width)
    crop_top = rng.randint(0, image.height - crop_height)
    image = image.crop(
        (crop_left, crop_top, crop_left + crop_width, crop_top + crop_height)
    )
    horizontal_flip = rng.random() < 0.5
    if horizontal_flip:
        image = image.transpose(_transpose_method())
    brightness = rng.uniform(0.88, 1.12)
    contrast = rng.uniform(0.88, 1.12)
    saturation = rng.uniform(0.88, 1.12)
    image = ImageEnhance.Brightness(image).enhance(brightness)
    image = ImageEnhance.Contrast(image).enhance(contrast)
    image = ImageEnhance.Color(image).enhance(saturation)
    output = image.resize((image_size, image_size), _resample())
    return output, {
        "crop_left": str(crop_left),
        "crop_top": str(crop_top),
        "crop_width": str(crop_width),
        "crop_height": str(crop_height),
        "horizontal_flip": str(horizontal_flip).lower(),
        "brightness": f"{brightness:.6f}",
        "contrast": f"{contrast:.6f}",
        "saturation": f"{saturation:.6f}",
    }


def _write_variant(
    item: VariantPlan,
    source_root: Path,
    staging_root: Path,
    image_size: int,
) -> dict[str, str]:
    destination = (
        staging_root
        / "train"
        / item.label
        / f"{item.output_index:05d}_{item.source.stem}_v{item.variant_index:03d}.jpg"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    image, parameters = _make_image(
        item.source, image_size=image_size, identity=item.identity, seed=item.seed
    )
    image.save(destination, format="JPEG", quality=95, optimize=True)
    output_bytes = destination.read_bytes()
    return {
        "split": "train",
        "label": item.label,
        "destination": destination.relative_to(staging_root).as_posix(),
        "source": item.source.relative_to(source_root).as_posix(),
        "source_index": str(item.source_index),
        "variant_index": str(item.variant_index),
        "identity": str(item.identity).lower(),
        "seed": str(item.seed),
        **parameters,
        "output_sha256": hashlib.sha256(output_bytes).hexdigest(),
        "output_bytes": str(len(output_bytes)),
    }


def _link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def _copy_evaluation_splits(
    source_root: Path,
    staging_root: Path,
    labels: tuple[str, ...],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for split in EVALUATION_SPLITS:
        split_root = source_root / split
        if not split_root.exists():
            continue
        for label in labels:
            source_directory = split_root / label
            destination_directory = staging_root / split / label
            destination_directory.mkdir(parents=True, exist_ok=True)
            if not source_directory.exists():
                continue
            for source in image_paths(source_directory):
                destination = destination_directory / source.name
                mode = _link_or_copy(source, destination)
                rows.append(
                    {
                        "split": split,
                        "label": label,
                        "source": source.relative_to(source_root).as_posix(),
                        "destination": destination.relative_to(staging_root).as_posix(),
                        "copy_mode": mode,
                    }
                )
    return rows


def _write_source_note(
    staging_root: Path,
    source_root: Path,
    plan: list[VariantPlan],
    evaluation_rows: list[dict[str, str]],
    image_size: int,
    seed: int,
) -> None:
    original_count = sum(item.identity for item in plan)
    augmented_count = len(plan) - original_count
    label_counts = {
        label: sum(item.label == label for item in plan)
        for label in sorted({item.label for item in plan})
    }
    lines = [
        "# Auxiliary decoder dataset",
        "",
        f"Source dataset: `{source_root}`",
        f"Training samples: {len(plan)} ({original_count} resized originals + {augmented_count} deterministic augmented views)",
        f"Output resolution: {image_size}x{image_size}",
        f"Seed: {seed}",
        f"Per-class counts: {label_counts}",
        "",
        "The validation, test, and new_holdout images were not used to create training views.",
        f"Evaluation files linked/copied without augmentation: {len(evaluation_rows)}",
        "This dataset contains 10,000 training instances, not 10,000 independent source photographs.",
        "Report both the source-image count and training-instance count in experiments.",
        "",
    ]
    (staging_root / "DATASET_SOURCE.md").write_text("\n".join(lines), encoding="utf-8")


def prepare_dataset(
    source_root: Path,
    output_root: Path,
    target_train_count: int,
    image_size: int,
    seed: int,
    workers: int,
) -> None:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if not (source_root / "train").is_dir():
        raise FileNotFoundError(f"training split not found: {source_root / 'train'}")
    if output_root.exists():
        raise FileExistsError(
            f"output already exists: {output_root}; choose a new --output path"
        )
    if image_size < 16:
        raise ValueError("image_size must be at least 16")
    if workers < 1:
        raise ValueError("workers must be positive")

    labels = tuple(sorted(path.name for path in (source_root / "train").iterdir() if path.is_dir()))
    sources_by_label = {
        label: image_paths(source_root / "train" / label) for label in labels
    }
    plan = build_variant_plan(sources_by_label, target_train_count, seed)
    staging_root = output_root.parent / f".{output_root.name}.staging"
    if staging_root.exists():
        raise FileExistsError(
            f"staging directory already exists after an interrupted run: {staging_root}"
        )
    staging_root.mkdir(parents=True)

    rows: list[dict[str, str]] = []
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _write_variant, item, source_root, staging_root, image_size
                )
                for item in plan
            ]
            for index, future in enumerate(as_completed(futures), start=1):
                rows.append(future.result())
                if index % 500 == 0 or index == len(futures):
                    print(f"Generated training images: {index}/{len(futures)}", flush=True)

        rows.sort(key=lambda row: (row["label"], row["destination"]))
        with (staging_root / "augmentation_manifest.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        evaluation_rows = _copy_evaluation_splits(source_root, staging_root, labels)
        if evaluation_rows:
            with (staging_root / "evaluation_split_manifest.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=list(evaluation_rows[0]))
                writer.writeheader()
                writer.writerows(evaluation_rows)
        _write_source_note(
            staging_root,
            source_root,
            plan,
            evaluation_rows,
            image_size,
            seed,
        )
        staging_root.replace(output_root)
    except Exception:
        print(f"Preparation failed; partial files remain in {staging_root}")
        raise

    unique_outputs = len({row["output_sha256"] for row in rows})
    print("\nAuxiliary dataset ready")
    print(f"  output: {output_root}")
    print(f"  training instances: {len(rows)}")
    print(f"  unique encoded outputs: {unique_outputs}")
    for label in labels:
        print(f"  train {label}: {sum(row['label'] == label for row in rows)}")
    print(f"  evaluation images (unmodified): {len(evaluation_rows)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic 10k-style auxiliary decoder dataset while keeping "
            "validation/test/holdout images isolated."
        )
    )
    parser.add_argument("--source", type=Path, default=Path("workspace/data/dataset"))
    parser.add_argument(
        "--output", type=Path, default=Path("workspace/data/dataset_aux_10k")
    )
    parser.add_argument("--target-train-count", type=int, default=10_000)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    prepare_dataset(
        source_root=args.source,
        output_root=args.output,
        target_train_count=args.target_train_count,
        image_size=args.image_size,
        seed=args.seed,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "VariantPlan",
    "balanced_label_counts",
    "build_variant_plan",
    "prepare_dataset",
]
