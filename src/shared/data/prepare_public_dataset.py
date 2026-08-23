from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

from ..configuration.workspace_paths import (
    DEFAULT_WORKSPACE_PATHS,
    WorkspacePaths,
)

REPOSITORY = "ml4py/dataset-iiit-pet"
BRANCH = "master"
TREE_API = (
    "https://api.github.com/repos/ml4py/dataset-iiit-pet/"
    "git/trees/7e58d388403fc156dbd9a69da5cf8f0bce43bd1b?recursive=1"
)
RAW_ROOT = "https://raw.githubusercontent.com/ml4py/dataset-iiit-pet/master/images"

def request_bytes(url: str, timeout: int = 60) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "split-learning-research-prototype"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def load_public_image_paths() -> list[str]:
    tree = json.loads(request_bytes(TREE_API))
    if tree.get("truncated"):
        raise RuntimeError("GitHub tree response was truncated")
    return [item["path"] for item in tree["tree"] if item["path"].lower().endswith(".jpg")]


def belongs_to_breed(filename: str, breeds: tuple[str, ...]) -> bool:
    return any(re.fullmatch(re.escape(breed) + r"_\d+\.jpg", filename) for breed in breeds)


def load_class_definitions(path: Path) -> dict[str, tuple[str, ...]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    definitions: dict[str, tuple[str, ...]] = {}
    owners: dict[str, str] = {}
    for item in payload.get("classes", []):
        name = str(item["name"])
        breeds = tuple(str(breed) for breed in item["breeds"])
        if name in definitions or not breeds:
            raise ValueError(f"Invalid or duplicate class definition: {name!r}")
        for breed in breeds:
            if breed in owners:
                raise ValueError(
                    f"Breed {breed!r} belongs to both {owners[breed]!r} and {name!r}"
                )
            owners[breed] = name
        definitions[name] = breeds
    if tuple(definitions) != tuple(sorted(definitions)):
        raise ValueError("dataset_classes.json classes must be alphabetically sorted")
    if len(definitions) < 2:
        raise ValueError("At least two dataset classes are required")
    return definitions


def select_samples(
    paths: list[str],
    definitions: dict[str, tuple[str, ...]],
    seed: int,
    count: int = 51,
) -> dict[str, list[str]]:
    candidates = {
        name: [path for path in paths if belongs_to_breed(path, breeds)]
        for name, breeds in definitions.items()
    }
    selected: dict[str, list[str]] = {}
    for label, items in candidates.items():
        if len(items) < count:
            raise RuntimeError(f"Not enough {label} images: {len(items)}")
        rng = random.Random(f"{seed}:{label}")
        items = sorted(items)
        rng.shuffle(items)
        selected[label] = items[:count]
    return selected


def merge_manifest(
    manifest: Path,
    completed: list[dict[str, str]],
    replaced_labels: set[str],
) -> list[dict[str, str]]:
    retained: list[dict[str, str]] = []
    if manifest.exists():
        with manifest.open(newline="", encoding="utf-8") as handle:
            retained = [
                dict(row) for row in csv.DictReader(handle) if row["label"] not in replaced_labels
            ]
    merged = retained + completed
    merged.sort(key=lambda row: (row["split"], row["label"], row["destination"]))
    return merged


def destination_plan(project_root: Path, selected: dict[str, list[str]]):
    workspace = WorkspacePaths(project_root / DEFAULT_WORKSPACE_PATHS.root)
    rows: list[dict[str, str]] = []
    split_ranges = (("train", 0, 30), ("val", 30, 40), ("test", 40, 50))
    for label, paths in selected.items():
        for split, start, end in split_ranges:
            for source_name in paths[start:end]:
                destination = workspace.dataset / split / label / source_name
                rows.append(
                    {
                        "source_file": source_name,
                        "source_url": f"{RAW_ROOT}/{source_name}",
                        "split": split,
                        "label": label,
                        "destination": destination.relative_to(project_root).as_posix(),
                    }
                )
        anchor_destination = workspace.anchors / label / f"{label}_anchor.jpg"
        rows.append(
            {
                "source_file": paths[50],
                "source_url": f"{RAW_ROOT}/{paths[50]}",
                "split": "anchor",
                "label": label,
                "destination": anchor_destination.relative_to(project_root).as_posix(),
            }
        )
    return rows


def download_one(project_root: Path, row: dict[str, str], force: bool) -> dict[str, str]:
    destination = project_root / row["destination"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        data = destination.read_bytes()
    else:
        data = request_bytes(row["source_url"])
        temporary = destination.with_suffix(destination.suffix + ".part")
        temporary.write_bytes(data)
        temporary.replace(destination)
    with Image.open(destination) as image:
        image.verify()
        width, height = image.size
        image_format = image.format
    if image_format != "JPEG":
        raise ValueError(f"Not a JPEG image: {destination} ({image_format})")
    return {
        **row,
        "width": str(width),
        "height": str(height),
        "bytes": str(len(data)),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a reproducible Oxford-IIIT Pet subset")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[3], type=Path)
    parser.add_argument(
        "--class-config",
        default=DEFAULT_WORKSPACE_PATHS.class_config.as_posix(),
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        help="Only prepare these classes and preserve other rows in the existing manifest",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    config_path = Path(args.class_config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    definitions = load_class_definitions(config_path)
    requested = tuple(args.labels) if args.labels else tuple(definitions)
    unknown = sorted(set(requested) - set(definitions))
    if unknown:
        parser.error(f"unknown labels: {unknown}; configured labels are {tuple(definitions)}")
    selected = select_samples(
        load_public_image_paths(),
        {name: definitions[name] for name in requested},
        args.seed,
    )
    plan = destination_plan(project_root, selected)
    completed: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(download_one, project_root, row, args.force) for row in plan]
        for index, future in enumerate(as_completed(futures), start=1):
            completed.append(future.result())
            if index % 10 == 0 or index == len(futures):
                print(f"Verified: {index}/{len(futures)}")
    workspace = WorkspacePaths(project_root / DEFAULT_WORKSPACE_PATHS.root)
    manifest = workspace.dataset / "public_subset_manifest.csv"
    completed = merge_manifest(manifest, completed, set(requested))
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=completed[0].keys())
        writer.writeheader()
        writer.writerows(completed)
    print("\nDataset ready")
    for split in ("train", "val", "test", "anchor"):
        for label in definitions:
            count = sum(row["split"] == split and row["label"] == label for row in completed)
            print(f"  {split:6s} {label}: {count}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
