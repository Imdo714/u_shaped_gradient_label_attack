from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .attacks.anchor_mapping import extract_anchor_gradient
from .attacks.smashed_data_attack import extract_inference_smashed_feature
from ..shared.data.class_catalog import checkpoint_class_catalog
from ..shared.data.image_dataset import load_image
from ..shared.configuration.workspace_paths import DEFAULT_WORKSPACE_PATHS
from ..split_learning.architecture.split_learning_model import load_split_learning_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Demonstrate distinct training and inference attack modes")
    parser.add_argument("--attack-mode", choices=("training_gradient", "inference_smashed"), required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--label", help="Legitimate ClientTail label; training_gradient only")
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_WORKSPACE_PATHS.checkpoints / "model.pt"),
    )
    parser.add_argument("--centroids", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    model, metadata = load_split_learning_model(args.checkpoint, device)
    catalog = checkpoint_class_catalog(metadata, model.h_model.num_classes)
    image_size = int(metadata.get("config", {}).get("image_size", 64))
    image = load_image(args.image, image_size)
    if args.attack_mode == "training_gradient":
        if args.label is None:
            parser.error("training_gradient requires --label (known only to legitimate ClientTail)")
        feature = extract_anchor_gradient(model, image, catalog.label(args.label), device)
    else:
        if args.label is not None:
            parser.error("inference_smashed does not accept a label or fabricate a loss gradient")
        feature = extract_inference_smashed_feature(model, image.to(device))
    centroids = np.load(args.centroids)
    if centroids.shape[1] != feature.shape[0]:
        raise ValueError("Centroids must be built from the same attack mode/features")
    distances = np.linalg.norm(centroids - feature, axis=1)
    print(f"Mode: {args.attack_mode}")
    print(f"Input image: {Path(args.image).name}")
    print(f"Assigned cluster: {int(np.argmin(distances))}")
    print(f"Distances: {distances.tolist()}")


if __name__ == "__main__":
    main()
