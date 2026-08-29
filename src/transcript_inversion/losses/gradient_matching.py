from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional


def soft_cross_entropy(logits: Tensor, condition: Tensor) -> Tensor:
    """Cross entropy for hard one-hot or inferred soft label conditions."""
    probabilities = condition / condition.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return -(probabilities * functional.log_softmax(logits, dim=1)).sum(dim=1).mean()


class GradientMatchingLoss(nn.Module):
    """Match observed dL/du without overfitting its batch-dependent magnitude."""

    def __init__(
        self,
        direction_weight: float = 1.0,
        normalized_mse_weight: float = 0.25,
        scale_weight: float = 0.05,
    ) -> None:
        super().__init__()
        self.direction_weight = direction_weight
        self.normalized_mse_weight = normalized_mse_weight
        self.scale_weight = scale_weight

    def forward(self, predicted: Tensor, observed: Tensor) -> tuple[Tensor, dict[str, float]]:
        predicted_flat = predicted.flatten(start_dim=1)
        observed_flat = observed.detach().flatten(start_dim=1)
        predicted_norm = predicted_flat.norm(dim=1, keepdim=True).clamp_min(1e-8)
        observed_norm = observed_flat.norm(dim=1, keepdim=True).clamp_min(1e-8)
        predicted_unit = predicted_flat / predicted_norm
        observed_unit = observed_flat / observed_norm
        direction = (1.0 - (predicted_unit * observed_unit).sum(dim=1)).mean()
        normalized_mse = functional.mse_loss(predicted_unit, observed_unit)
        scale = functional.l1_loss(predicted_norm.log(), observed_norm.log())
        total = (
            self.direction_weight * direction
            + self.normalized_mse_weight * normalized_mse
            + self.scale_weight * scale
        )
        return total, {
            "gradient_loss": float(total.detach()),
            "gradient_cosine": float((1.0 - direction).detach()),
            "gradient_normalized_mse": float(normalized_mse.detach()),
            "gradient_log_scale": float(scale.detach()),
        }


__all__ = ["GradientMatchingLoss", "soft_cross_entropy"]
