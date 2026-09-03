from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


DEFAULT_FONT_CANDIDATES = (
    Path("C:/Windows/Fonts/arialbd.ttf"),
    Path("C:/Windows/Fonts/ariblk.ttf"),
    Path("C:/Windows/Fonts/bahnschrift.ttf"),
    Path("C:/Windows/Fonts/calibrib.ttf"),
    Path("C:/Windows/Fonts/verdana.ttf"),
    Path("C:/Windows/Fonts/timesbd.ttf"),
)


@dataclass(frozen=True)
class SamplePlan:
    domain: str
    split: str
    label: str
    character: str
    index: int
    seed: int


def _stable_seed(base_seed: int, *parts: object) -> int:
    value = ":".join([str(base_seed), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def _balanced_counts(labels: tuple[str, ...], total: int) -> dict[str, int]:
    if total < len(labels):
        raise ValueError("each split must contain at least one sample per class")
    base, remainder = divmod(total, len(labels))
    return {
        label: base + int(index < remainder) for index, label in enumerate(labels)
    }


def _plans(
    letters: tuple[str, ...],
    split_counts: dict[tuple[str, str], int],
    seed: int,
) -> list[SamplePlan]:
    labels = tuple(f"letter_{letter.lower()}" for letter in letters)
    result: list[SamplePlan] = []
    for (domain, split), total in split_counts.items():
        counts = _balanced_counts(labels, total)
        for label, character in zip(labels, letters):
            for index in range(counts[label]):
                result.append(
                    SamplePlan(
                        domain=domain,
                        split=split,
                        label=label,
                        character=character,
                        index=index,
                        seed=_stable_seed(seed, domain, split, label, index),
                    )
                )
    return result


def _fonts(values: list[Path] | None) -> tuple[Path, ...]:
    candidates = tuple(values) if values else DEFAULT_FONT_CANDIDATES
    available = tuple(path.resolve() for path in candidates if path.is_file())
    if not available:
        raise FileNotFoundError(
            "no TrueType fonts found; pass one or more existing paths with --fonts"
        )
    return available


def _render(
    plan: SamplePlan,
    destination: Path,
    image_size: int,
    fonts: tuple[Path, ...],
) -> dict[str, str]:
    rng = random.Random(plan.seed)
    np_rng = np.random.default_rng(plan.seed)
    dark_background = rng.random() < 0.2
    if dark_background:
        background = tuple(rng.randint(8, 55) for _ in range(3))
        foreground = tuple(rng.randint(205, 255) for _ in range(3))
    else:
        background = tuple(rng.randint(205, 255) for _ in range(3))
        foreground = tuple(rng.randint(0, 65) for _ in range(3))

    canvas_size = round(image_size * 1.35)
    image = Image.new("RGB", (canvas_size, canvas_size), background)
    draw = ImageDraw.Draw(image)
    # Low-contrast nuisance lines prevent the generator from producing a single
    # unrealistically clean template while keeping the character readable.
    for _ in range(rng.randint(0, 3)):
        line_color = tuple(
            max(0, min(255, channel + rng.randint(-20, 20)))
            for channel in background
        )
        draw.line(
            (
                rng.randint(0, canvas_size),
                rng.randint(0, canvas_size),
                rng.randint(0, canvas_size),
                rng.randint(0, canvas_size),
            ),
            fill=line_color,
            width=rng.randint(1, 3),
        )

    font_path = rng.choice(fonts)
    font_size = rng.randint(round(image_size * 0.58), round(image_size * 0.9))
    font = ImageFont.truetype(str(font_path), font_size)
    box = draw.textbbox((0, 0), plan.character, font=font, stroke_width=1)
    width, height = box[2] - box[0], box[3] - box[1]
    offset_x = rng.randint(round(-image_size * 0.12), round(image_size * 0.12))
    offset_y = rng.randint(round(-image_size * 0.12), round(image_size * 0.12))
    position = (
        (canvas_size - width) // 2 - box[0] + offset_x,
        (canvas_size - height) // 2 - box[1] + offset_y,
    )
    stroke_width = rng.randint(0, max(1, round(image_size * 0.025)))
    stroke_fill = tuple((a + b) // 2 for a, b in zip(background, foreground))
    draw.text(
        position,
        plan.character,
        font=font,
        fill=foreground,
        stroke_width=stroke_width,
        stroke_fill=stroke_fill,
    )

    angle = rng.uniform(-22.0, 22.0)
    image = image.rotate(
        angle,
        resample=getattr(Image, "Resampling", Image).BICUBIC,
        expand=False,
        fillcolor=background,
    )
    crop_left = (canvas_size - image_size) // 2
    image = image.crop(
        (crop_left, crop_left, crop_left + image_size, crop_left + image_size)
    )
    blur_radius = rng.uniform(0.0, 0.9)
    if blur_radius > 0.15:
        image = image.filter(ImageFilter.GaussianBlur(blur_radius))
    noise_sigma = rng.uniform(0.0, 7.0)
    pixels = np.asarray(image, dtype=np.float32)
    pixels += np_rng.normal(0.0, noise_sigma, size=pixels.shape)
    image = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8), mode="RGB")

    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="JPEG", quality=92, optimize=True)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    return {
        "domain": plan.domain,
        "split": plan.split,
        "label": plan.label,
        "character": plan.character,
        "index": str(plan.index),
        "seed": str(plan.seed),
        "relative_path": destination.as_posix(),
        "font": font_path.name,
        "font_size": str(font_size),
        "angle": f"{angle:.6f}",
        "offset_x": str(offset_x),
        "offset_y": str(offset_y),
        "stroke_width": str(stroke_width),
        "background_rgb": ":".join(map(str, background)),
        "foreground_rgb": ":".join(map(str, foreground)),
        "blur_radius": f"{blur_radius:.6f}",
        "noise_sigma": f"{noise_sigma:.6f}",
        "sha256": digest,
    }


def prepare_letter_dataset(
    output_root: Path,
    *,
    letters: tuple[str, ...] = ("A", "B"),
    image_size: int = 128,
    victim_train: int = 600,
    victim_validation: int = 100,
    victim_test: int = 100,
    victim_holdout: int = 100,
    auxiliary_train: int = 10_000,
    auxiliary_validation: int = 100,
    seed: int = 42,
    workers: int = 8,
    font_paths: list[Path] | None = None,
) -> Path:
    if output_root.exists():
        raise FileExistsError(f"output already exists: {output_root}")
    if len(letters) < 2 or len(set(letters)) != len(letters):
        raise ValueError("provide at least two unique letters")
    if any(len(letter) != 1 or not letter.isalpha() for letter in letters):
        raise ValueError("each class must be one alphabetic character")
    if image_size < 32:
        raise ValueError("image_size must be at least 32")
    if workers < 1:
        raise ValueError("workers must be positive")
    fonts = _fonts(font_paths)
    split_counts = {
        ("victim", "train"): victim_train,
        ("victim", "val"): victim_validation,
        ("victim", "test"): victim_test,
        ("victim", "new_holdout"): victim_holdout,
        ("auxiliary_10k", "train"): auxiliary_train,
        ("auxiliary_10k", "val"): auxiliary_validation,
    }
    plans = _plans(tuple(letter.upper() for letter in letters), split_counts, seed)
    staging = output_root.parent / f".{output_root.name}.staging"
    if staging.exists():
        raise FileExistsError(f"staging output already exists: {staging}")
    staging.mkdir(parents=True)
    rows: list[dict[str, str]] = []
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = []
            for plan in plans:
                relative = (
                    Path(plan.domain)
                    / plan.split
                    / plan.label
                    / f"{plan.character.lower()}_{plan.index:05d}.jpg"
                )
                futures.append(
                    executor.submit(
                        _render, plan, staging / relative, image_size, fonts
                    )
                )
            for completed, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                row["relative_path"] = Path(row["relative_path"]).relative_to(
                    staging
                ).as_posix()
                rows.append(row)
                if completed % 1000 == 0 or completed == len(futures):
                    print(f"Generated letter images: {completed}/{len(futures)}", flush=True)

        rows.sort(key=lambda row: row["relative_path"])
        victim_holdout_hashes = {
            row["sha256"]
            for row in rows
            if row["domain"] == "victim" and row["split"] == "new_holdout"
        }
        auxiliary_train_hashes = {
            row["sha256"]
            for row in rows
            if row["domain"] == "auxiliary_10k" and row["split"] == "train"
        }
        overlap = victim_holdout_hashes & auxiliary_train_hashes
        if overlap:
            raise RuntimeError("auxiliary training and victim holdout contain duplicates")
        with (staging / "generation_manifest.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        metadata = {
            "generator": "deterministic synthetic Latin letter renderer",
            "letters": list(letters),
            "class_names": [f"letter_{letter.lower()}" for letter in letters],
            "image_size": image_size,
            "seed": seed,
            "fonts": [
                {
                    "name": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for path in fonts
            ],
            "split_counts": {
                f"{domain}/{split}": count
                for (domain, split), count in split_counts.items()
            },
            "auxiliary_train_victim_holdout_sha256_overlap": len(overlap),
        }
        with (staging / "dataset_metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
        (staging / "DATASET_SOURCE.md").write_text(
            "# Synthetic letter experiment dataset\n\n"
            "Images are generated locally from the recorded TrueType fonts. Each "
            "sample independently varies font, position, rotation, colors, blur, "
            "noise, and nuisance lines. The auxiliary training and victim holdout "
            "domains use disjoint deterministic seeds; SHA-256 overlap is checked "
            "before the dataset is finalized. These are synthetic instances, not "
            "independent real handwriting samples.\n",
            encoding="utf-8",
        )
        staging.replace(output_root)
    except Exception:
        print(f"Generation failed; partial output remains at {staging}")
        raise
    print(f"Letter dataset ready: {output_root.resolve()}")
    return output_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate disjoint victim and public-auxiliary letter images."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("workspace/data/letter_experiment"),
    )
    parser.add_argument("--letters", nargs="+", default=["A", "B"])
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--victim-train", type=int, default=600)
    parser.add_argument("--victim-validation", type=int, default=100)
    parser.add_argument("--victim-test", type=int, default=100)
    parser.add_argument("--victim-holdout", type=int, default=100)
    parser.add_argument("--auxiliary-train", type=int, default=10_000)
    parser.add_argument("--auxiliary-validation", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--fonts", nargs="+", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    prepare_letter_dataset(
        args.output,
        letters=tuple(args.letters),
        image_size=args.image_size,
        victim_train=args.victim_train,
        victim_validation=args.victim_validation,
        victim_test=args.victim_test,
        victim_holdout=args.victim_holdout,
        auxiliary_train=args.auxiliary_train,
        auxiliary_validation=args.auxiliary_validation,
        seed=args.seed,
        workers=args.workers,
        font_paths=args.fonts,
    )


if __name__ == "__main__":
    main()


__all__ = ["SamplePlan", "build_parser", "prepare_letter_dataset"]
