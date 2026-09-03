from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional


def conservative_visual_enhancement(
    image: Tensor,
    chroma_denoise: float = 0.15,
    sharpen_amount: float = 0.2,
    contrast: float = 1.03,
    saturation: float = 1.03,
) -> Tensor:
    """Apply weak deterministic enhancement without a generative image prior."""

    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError(f"expected BCHW RGB tensor, found {tuple(image.shape)}")
    if not 0.0 <= chroma_denoise <= 1.0:
        raise ValueError("chroma_denoise must be in [0, 1]")
    if sharpen_amount < 0.0 or contrast <= 0.0 or saturation <= 0.0:
        raise ValueError("enhancement strengths must be non-negative")
    weights = image.new_tensor((0.299, 0.587, 0.114)).view(1, 3, 1, 1)
    luminance = (image * weights).sum(dim=1, keepdim=True)
    chroma = image - luminance
    smooth_chroma = functional.avg_pool2d(
        chroma, kernel_size=3, stride=1, padding=1, count_include_pad=False
    )
    enhanced = luminance + torch.lerp(chroma, smooth_chroma, chroma_denoise)
    blurred = functional.avg_pool2d(
        enhanced, kernel_size=3, stride=1, padding=1, count_include_pad=False
    )
    enhanced = enhanced + sharpen_amount * (enhanced - blurred)
    enhanced = (enhanced - 0.5) * contrast + 0.5
    enhanced_luminance = (enhanced * weights).sum(dim=1, keepdim=True)
    enhanced = enhanced_luminance + saturation * (enhanced - enhanced_luminance)
    return enhanced.clamp(0.0, 1.0)


__all__ = ["conservative_visual_enhancement"]
