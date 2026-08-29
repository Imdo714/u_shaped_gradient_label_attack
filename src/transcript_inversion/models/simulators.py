from __future__ import annotations

from torch import Tensor, nn
from torch.nn import functional

from .signal_encoder import as_feature_map, group_count


class _ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.GroupNorm(group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )


class FrontSimulator(nn.Module):
    """Generic f-hat trained by transcript distribution alignment, not f weights."""

    def __init__(
        self,
        output_channels: int,
        output_spatial_shape: tuple[int, int],
        width: int = 32,
    ) -> None:
        super().__init__()
        if min(output_channels, *output_spatial_shape, width) < 1:
            raise ValueError("front simulator dimensions must be positive")
        self.output_spatial_shape = output_spatial_shape
        self.network = nn.Sequential(
            _ConvBlock(3, width),
            _ConvBlock(width, width * 2, stride=2),
            _ConvBlock(width * 2, width * 2, stride=2),
            nn.Conv2d(width * 2, output_channels, 1),
        )

    def forward(self, image: Tensor) -> Tensor:
        value = self.network(image)
        return functional.interpolate(
            value, size=self.output_spatial_shape, mode="bilinear", align_corners=False
        )


class TailGradientSimulator(nn.Module):
    """Generic h-hat learned from (u, dL/du, pseudo-label) gradient matching."""

    def __init__(self, input_channels: int, num_classes: int, width: int = 64) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, width, 1),
            nn.GroupNorm(group_count(width), width),
            nn.SiLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GroupNorm(group_count(width), width),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(width * 4 * 4, width * 2),
            nn.SiLU(inplace=True),
            nn.Linear(width * 2, num_classes),
        )

    def forward(self, server_output: Tensor) -> Tensor:
        return self.classifier(self.features(as_feature_map(server_output)))


__all__ = ["FrontSimulator", "TailGradientSimulator"]
