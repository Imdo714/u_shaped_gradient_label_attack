from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from ..shared.configuration.workspace_paths import DEFAULT_WORKSPACE_PATHS
from ..shared.data.class_catalog import checkpoint_class_catalog
from ..shared.data.image_dataset import load_image
from ..split_learning.architecture.split_learning_model import (
    SplitLearningModel,
    load_split_learning_model,
)


def label_gradient(
    model: SplitLearningModel,
    image: torch.Tensor,
    label: int,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    model.eval()  # Dropout disabled: both runs use exactly the same logits/model state.
    z = model.f_model(image).detach()
    u = model.g_model(z).detach().requires_grad_(True)
    logits = model.h_model(u)
    loss = nn.CrossEntropyLoss()(logits, torch.tensor([label], device=image.device))
    gradient = torch.autograd.grad(loss, u)[0].detach()
    return float(loss.detach()), logits.detach(), gradient


def experiment_same_image_different_label(
    model: SplitLearningModel,
    image: torch.Tensor,
    class_names: tuple[str, ...],
    image_id: str = "fixed-image",
) -> dict[str, float]:
    """Causal sanity check, not the label-inference attack itself."""
    for component in (model.f_model, model.g_model, model.h_model):
        for parameter in component.parameters():
            parameter.requires_grad_(False)
    observations = [label_gradient(model, image, label) for label in range(len(class_names))]
    reference_logits = observations[0][1]
    if any(not torch.equal(reference_logits, logits) for _, logits, _ in observations[1:]):
        raise AssertionError("Logits changed between label interventions")
    result: dict[str, float] = {}
    for name, (loss, _, gradient) in zip(class_names, observations):
        result[f"loss_{name}"] = loss
        result[f"norm_{name}"] = float(gradient.norm())
    for left, right in combinations(range(len(class_names)), 2):
        left_name, right_name = class_names[left], class_names[right]
        left_grad = observations[left][2].flatten()
        right_grad = observations[right][2].flatten()
        key = f"{left_name}_vs_{right_name}"
        result[f"cosine_{key}"] = float(F.cosine_similarity(left_grad, right_grad, dim=0))
        result[f"distance_{key}"] = float(torch.linalg.vector_norm(left_grad - right_grad))
    print("=" * 42)
    print("SAME IMAGE / DIFFERENT LABEL EXPERIMENT")
    print("=" * 42)
    print(f"Image ID: {image_id}")
    for name in class_names:
        print(
            f"{name}: loss={result[f'loss_{name}']:.8f}, "
            f"gradient_norm={result[f'norm_{name}']:.8f}"
        )
    for left, right in combinations(class_names, 2):
        key = f"{left}_vs_{right}"
        print(
            f"{key}: cosine={result[f'cosine_{key}']:.8f}, "
            f"distance={result[f'distance_{key}']:.8f}"
        )
    print("=" * 42)
    print("This causal sanity check is not itself the label-inference attack.")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_WORKSPACE_PATHS.checkpoints / "model.pt"),
    )
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    model, metadata = load_split_learning_model(args.checkpoint, device)
    catalog = checkpoint_class_catalog(metadata, model.h_model.num_classes)
    image_size = metadata.get("config", {}).get("image_size", args.image_size)
    image = load_image(args.image, image_size).to(device)
    experiment_same_image_different_label(model, image, catalog.names, Path(args.image).name)


if __name__ == "__main__":
    main()
