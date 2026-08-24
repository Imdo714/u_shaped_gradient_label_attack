from __future__ import annotations

import torch.nn.functional as functional
from torch import Tensor, nn


class SignalEncoder(nn.Module):
    """Project one observed communication tensor into a common feature space."""

    def __init__(self, in_channels: int, out_channels: int, spatial_size: int = 8) -> None:
        super().__init__()
        self.spatial_size = spatial_size
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.GroupNorm(4, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(4, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, signal: Tensor) -> Tensor:
        signal = functional.adaptive_avg_pool2d(signal, (self.spatial_size, self.spatial_size))
        return self.network(signal)


__all__ = ["SignalEncoder"]
