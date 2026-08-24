from __future__ import annotations

import torch
from torch import Tensor, nn


def structural_similarity(x: Tensor, y: Tensor) -> Tensor:
    """Differentiable global SSIM averaged over batch and RGB channels."""
    dimensions = (-2, -1)
    mean_x = x.mean(dim=dimensions, keepdim=True)
    mean_y = y.mean(dim=dimensions, keepdim=True)
    var_x = ((x - mean_x) ** 2).mean(dim=dimensions, keepdim=True)
    var_y = ((y - mean_y) ** 2).mean(dim=dimensions, keepdim=True)
    covariance = ((x - mean_x) * (y - mean_y)).mean(dim=dimensions, keepdim=True)
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mean_x * mean_y + c1) * (2 * covariance + c2)) / (
        (mean_x.square() + mean_y.square() + c1) * (var_x + var_y + c2)
    )
    return score.mean()


class ReconstructionLoss(nn.Module):
    def __init__(self, l1_weight: float = 1.0, ssim_weight: float = 0.5) -> None:
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight

    def forward(self, reconstruction: Tensor, target: Tensor) -> tuple[Tensor, dict[str, float]]:
        l1 = torch.nn.functional.l1_loss(reconstruction, target)
        ssim_loss = 1.0 - structural_similarity(reconstruction, target)
        total = self.l1_weight * l1 + self.ssim_weight * ssim_loss
        return total, {
            "loss": float(total.detach()),
            "l1": float(l1.detach()),
            "ssim": float((1.0 - ssim_loss).detach()),
        }


__all__ = ["ReconstructionLoss", "structural_similarity"]
