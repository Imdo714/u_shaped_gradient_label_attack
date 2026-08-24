from __future__ import annotations

import torch
from torch import Tensor

from ..losses.reconstruction_loss import structural_similarity


def per_sample_metrics(reconstruction: Tensor, target: Tensor) -> dict[str, Tensor]:
    difference = reconstruction - target
    mse = difference.square().flatten(1).mean(dim=1)
    mae = difference.abs().flatten(1).mean(dim=1)
    psnr = 10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))
    ssim_values = torch.stack(
        [structural_similarity(reconstruction[i : i + 1], target[i : i + 1]) for i in range(len(target))]
    )
    return {"mse": mse, "mae": mae, "psnr": psnr, "ssim": ssim_values}


__all__ = ["per_sample_metrics"]
