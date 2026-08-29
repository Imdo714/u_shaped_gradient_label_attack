from __future__ import annotations

import torch.nn.functional as functional
from torch import Tensor, nn


def as_feature_map(value: Tensor) -> Tensor:
    """Convert vector-valued transcripts to a 1x1 spatial feature map."""
    if value.ndim == 2:
        return value[:, :, None, None]
    if value.ndim != 4:
        raise ValueError(f"expected [B,C] or [B,C,H,W], got {tuple(value.shape)}")
    return value


def group_count(channels: int, maximum: int = 8) -> int:
    groups = min(channels, maximum)
    while channels % groups:
        groups -= 1
    return groups


class SignalEncoder(nn.Module):
    """Encode one transcript view without assuming its original spatial depth."""

    def __init__(self, in_channels: int, out_channels: int, spatial_size: int) -> None:
        super().__init__()
        self.spatial_size = spatial_size
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.GroupNorm(group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, value: Tensor) -> Tensor:
        value = as_feature_map(value)
        value = functional.adaptive_avg_pool2d(
            value, (self.spatial_size, self.spatial_size)
        )
        return self.network(value)


__all__ = ["SignalEncoder", "as_feature_map", "group_count"]
