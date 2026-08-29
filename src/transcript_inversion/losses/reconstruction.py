from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional


def structural_similarity(x: Tensor, y: Tensor) -> Tensor:
    dimensions = (-2, -1)
    mean_x = x.mean(dim=dimensions, keepdim=True)
    mean_y = y.mean(dim=dimensions, keepdim=True)
    variance_x = ((x - mean_x) ** 2).mean(dim=dimensions, keepdim=True)
    variance_y = ((y - mean_y) ** 2).mean(dim=dimensions, keepdim=True)
    covariance = ((x - mean_x) * (y - mean_y)).mean(dim=dimensions, keepdim=True)
    c1, c2 = 0.01**2, 0.03**2
    return (
        ((2 * mean_x * mean_y + c1) * (2 * covariance + c2))
        / ((mean_x.square() + mean_y.square() + c1) * (variance_x + variance_y + c2))
    ).mean()


class ReconstructionLoss(nn.Module):
    def __init__(self, l1_weight: float = 1.0, ssim_weight: float = 0.5, edge_weight: float = 0.1) -> None:
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.edge_weight = edge_weight

    @staticmethod
    def _edges(image: Tensor) -> tuple[Tensor, Tensor]:
        return image[..., 1:, :] - image[..., :-1, :], image[..., :, 1:] - image[..., :, :-1]

    def forward(self, reconstruction: Tensor, target: Tensor) -> tuple[Tensor, dict[str, float]]:
        l1 = functional.l1_loss(reconstruction, target)
        ssim = structural_similarity(reconstruction, target)
        reconstruction_edges = self._edges(reconstruction)
        target_edges = self._edges(target)
        edge = sum(
            functional.l1_loss(left, right)
            for left, right in zip(reconstruction_edges, target_edges)
        ) / 2.0
        total = self.l1_weight * l1 + self.ssim_weight * (1.0 - ssim) + self.edge_weight * edge
        return total, {
            "reconstruction_loss": float(total.detach()),
            "l1": float(l1.detach()),
            "ssim": float(ssim.detach()),
            "edge": float(edge.detach()),
        }


__all__ = ["ReconstructionLoss", "structural_similarity"]
