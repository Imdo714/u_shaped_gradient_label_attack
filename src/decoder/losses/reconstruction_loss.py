from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional


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
    def __init__(
        self,
        l1_weight: float = 1.0,
        ssim_weight: float = 0.5,
        edge_weight: float = 0.0,
        perceptual_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.edge_weight = edge_weight
        self.perceptual_weight = perceptual_weight

    @staticmethod
    def _edge_magnitude(image: Tensor) -> Tensor:
        kernel_x = image.new_tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).view(1, 1, 3, 3)
        kernel_y = kernel_x.transpose(-1, -2)
        channels = image.shape[1]
        gradient_x = functional.conv2d(
            image, kernel_x.expand(channels, 1, 3, 3), padding=1, groups=channels
        )
        gradient_y = functional.conv2d(
            image, kernel_y.expand(channels, 1, 3, 3), padding=1, groups=channels
        )
        return torch.sqrt(gradient_x.square() + gradient_y.square() + 1e-12)

    @staticmethod
    def _perceptual_pyramid_l1(reconstruction: Tensor, target: Tensor) -> Tensor:
        losses: list[Tensor] = []
        current_reconstruction = reconstruction
        current_target = target
        for _ in range(3):
            losses.append(functional.l1_loss(current_reconstruction, current_target))
            if min(current_reconstruction.shape[-2:]) < 2:
                break
            current_reconstruction = functional.avg_pool2d(current_reconstruction, 2)
            current_target = functional.avg_pool2d(current_target, 2)
        return torch.stack(losses).mean()

    def forward(self, reconstruction: Tensor, target: Tensor) -> tuple[Tensor, dict[str, float]]:
        l1 = torch.nn.functional.l1_loss(reconstruction, target)
        ssim_loss = 1.0 - structural_similarity(reconstruction, target)
        edge = functional.l1_loss(
            self._edge_magnitude(reconstruction), self._edge_magnitude(target)
        )
        perceptual = self._perceptual_pyramid_l1(reconstruction, target)
        total = (
            self.l1_weight * l1
            + self.ssim_weight * ssim_loss
            + self.edge_weight * edge
            + self.perceptual_weight * perceptual
        )
        return total, {
            "loss": float(total.detach()),
            "l1": float(l1.detach()),
            "ssim": float((1.0 - ssim_loss).detach()),
            "edge": float(edge.detach()),
            "perceptual": float(perceptual.detach()),
        }


__all__ = ["ReconstructionLoss", "structural_similarity"]
