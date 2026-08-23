from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..shared.configuration.workspace_paths import DEFAULT_WORKSPACE_PATHS
from .attacks.anchor_mapping import (
    extract_anchor_gradient,
    extract_anchor_matrix,
    map_anchors_to_clusters,
    print_anchor_mapping,
)
from ..shared.data.class_catalog import checkpoint_class_catalog
from ..shared.data.image_dataset import load_image
from ..split_learning.architecture.split_learning_model import load_split_learning_model


def describe_single_anchor(
    image_path: str, label: int, class_names: tuple[str, ...], model, centroids, device, image_size
):
    feature = extract_anchor_gradient(model, load_image(image_path, image_size), label, device)
    centers = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)
    similarities = centers @ feature
    distances = np.linalg.norm(centers - feature, axis=1)
    assigned = int(np.argmin(distances))
    print(f"Input image: {Path(image_path).name}")
    print(f"Known anchor label: {class_names[label].upper()}\n")
    for cluster, value in enumerate(similarities):
        print(f"cosine similarity to Cluster {cluster}: {value:.8f}")
    print()
    for cluster, value in enumerate(distances):
        print(f"Euclidean distance to Cluster {cluster}: {value:.8f}")
    print(f"\nAnchor assigned to: Cluster {assigned}")
    return feature, assigned


def main() -> None:
    parser = argparse.ArgumentParser(description="Identify cluster semantics with real anchors")
    parser.add_argument("--anchor-dir", default=str(DEFAULT_WORKSPACE_PATHS.anchors))
    parser.add_argument("--image")
    parser.add_argument("--label")
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_WORKSPACE_PATHS.checkpoints / "model.pt"),
    )
    parser.add_argument(
        "--centroids",
        default=str(DEFAULT_WORKSPACE_PATHS.reports / "gradient_centroids_epoch_001.npy"),
    )
    parser.add_argument(
        "--mapping-output",
        default=str(DEFAULT_WORKSPACE_PATHS.reports / "cluster_mapping.json"),
    )
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    model, metadata = load_split_learning_model(args.checkpoint, device)
    catalog = checkpoint_class_catalog(metadata, model.h_model.num_classes)
    image_size = int(metadata.get("config", {}).get("image_size", args.image_size))
    centroids = np.load(args.centroids)
    if args.image:
        if args.label is None:
            parser.error("--image requires --label")
        describe_single_anchor(
            args.image,
            catalog.label(args.label),
            catalog.names,
            model,
            centroids,
            device,
            image_size,
        )
        return
    anchors = extract_anchor_matrix(
        model, catalog.anchor_paths(args.anchor_dir), image_size, device
    )
    result = map_anchors_to_clusters(anchors, centroids)
    print_anchor_mapping(result, list(catalog.names))
    path = Path(args.mapping_output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({str(k): v for k, v in result.cluster_to_label.items()}, indent=2))


if __name__ == "__main__":
    main()
