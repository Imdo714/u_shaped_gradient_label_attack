from __future__ import annotations

import torch
from torch import Tensor


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def denormalize_image(image: Tensor) -> Tensor:
    """Convert an ImageNet-normalized tensor to a displayable [0, 1] tensor."""
    mean = image.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = image.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    batched = image.unsqueeze(0) if image.ndim == 3 else image
    result = (batched * std + mean).clamp(0.0, 1.0)
    return result.squeeze(0) if image.ndim == 3 else result


def normalize_image(image: Tensor) -> Tensor:
    """Convert a [0, 1] RGB tensor to the split model's normalized input space."""
    mean = image.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = image.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    batched = image.unsqueeze(0) if image.ndim == 3 else image
    result = (batched - mean) / std
    return result.squeeze(0) if image.ndim == 3 else result


def l2_normalize_gradient(gradient: Tensor, eps: float = 1e-12) -> Tensor:
    batched = gradient.unsqueeze(0) if gradient.ndim == 3 else gradient
    norm = batched.flatten(1).norm(dim=1).clamp_min(eps).view(-1, 1, 1, 1)
    result = batched / norm
    return result.squeeze(0) if gradient.ndim == 3 else result


__all__ = ["denormalize_image", "normalize_image", "l2_normalize_gradient"]
