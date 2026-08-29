from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional


def _compact(value: Tensor, spatial_size: int = 4) -> Tensor:
    if value.ndim == 4:
        value = functional.adaptive_avg_pool2d(value, (spatial_size, spatial_size))
    return value.flatten(start_dim=1)


def _coral(left: Tensor, right: Tensor) -> Tensor:
    mean_loss = functional.mse_loss(left.mean(dim=0), right.mean(dim=0))
    if left.shape[0] < 2 or right.shape[0] < 2:
        return mean_loss
    left_centered = left - left.mean(dim=0, keepdim=True)
    right_centered = right - right.mean(dim=0, keepdim=True)
    left_cov = left_centered.T @ left_centered / (left.shape[0] - 1)
    right_cov = right_centered.T @ right_centered / (right.shape[0] - 1)
    covariance_loss = functional.mse_loss(left_cov, right_cov)
    return mean_loss + covariance_loss


def _mmd(left: Tensor, right: Tensor) -> Tensor:
    combined = torch.cat((left, right), dim=0)
    distances = torch.cdist(combined, combined).square().detach()
    nonzero = distances[distances > 0]
    bandwidth = nonzero.median().clamp_min(1e-6) if nonzero.numel() else left.new_tensor(1.0)

    def kernel(first: Tensor, second: Tensor) -> Tensor:
        return torch.exp(-torch.cdist(first, second).square() / (2.0 * bandwidth))

    return kernel(left, left).mean() + kernel(right, right).mean() - 2.0 * kernel(left, right).mean()


class ConditionalTranscriptAlignmentLoss(nn.Module):
    """Align real and simulated transcript distributions by inferred class."""

    def __init__(self, coral_weight: float = 1.0, mmd_weight: float = 0.25) -> None:
        super().__init__()
        self.coral_weight = coral_weight
        self.mmd_weight = mmd_weight

    def _view_loss(
        self,
        simulated: Tensor,
        observed: Tensor,
        simulated_labels: Tensor,
        observed_labels: Tensor,
    ) -> Tensor:
        simulated = _compact(simulated)
        observed = _compact(observed.detach())
        losses: list[Tensor] = []
        classes = torch.unique(torch.cat((simulated_labels, observed_labels))).tolist()
        for label in classes:
            left = simulated[simulated_labels == label]
            right = observed[observed_labels == label]
            if left.shape[0] and right.shape[0]:
                losses.append(
                    self.coral_weight * _coral(left, right)
                    + self.mmd_weight * _mmd(left, right)
                )
        if not losses:
            return simulated.sum() * 0.0
        return torch.stack(losses).mean()

    def forward(
        self,
        simulated_views: dict[str, Tensor],
        observed_views: dict[str, Tensor],
        simulated_condition: Tensor,
        observed_condition: Tensor,
    ) -> tuple[Tensor, dict[str, float]]:
        shared = sorted(set(simulated_views) & set(observed_views))
        if not shared:
            raise ValueError("no common transcript views to align")
        simulated_labels = simulated_condition.argmax(dim=1)
        observed_labels = observed_condition.argmax(dim=1)
        view_losses = {
            name: self._view_loss(
                simulated_views[name],
                observed_views[name],
                simulated_labels,
                observed_labels,
            )
            for name in shared
        }
        total = torch.stack(list(view_losses.values())).mean()
        metrics = {f"alignment_{name}": float(value.detach()) for name, value in view_losses.items()}
        metrics["alignment_loss"] = float(total.detach())
        return total, metrics


__all__ = ["ConditionalTranscriptAlignmentLoss"]
