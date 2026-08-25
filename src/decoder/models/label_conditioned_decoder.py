from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from ..data.image_scaling import l2_normalize_gradient
from .signal_encoder import SignalEncoder


@dataclass(frozen=True)
class DecoderConfig:
    z_channels: int
    u_channels: int
    gradient_channels: int
    num_classes: int
    image_size: int = 64
    use_z: bool = True
    use_u: bool = True
    use_gradient: bool = True
    signal_spatial_size: int = 8
    signal_channels: int = 32
    label_channels: int = 16
    decoder_base_channels: int = 128
    decoder_min_channels: int = 16
    decoder_refinement_blocks: int = 0

    def to_dict(self) -> dict[str, int | bool]:
        return asdict(self)


class _UpsampleBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        groups = min(4, out_channels)
        while out_channels % groups:
            groups -= 1
        super().__init__(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
        )


class _RefinementBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = min(8, channels)
        while channels % groups:
            groups -= 1
        self.network = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, value: Tensor) -> Tensor:
        return self.activation(value + self.network(value))


class LabelConditionedDecoder(nn.Module):
    """Reconstruct an image from z, u, dL/du and a hard or soft label."""

    def __init__(self, config: DecoderConfig) -> None:
        super().__init__()
        if not any((config.use_z, config.use_u, config.use_gradient)):
            raise ValueError("at least one observed signal must be enabled")
        if config.signal_spatial_size < 1:
            raise ValueError("signal_spatial_size must be positive")
        if config.image_size < config.signal_spatial_size:
            raise ValueError("image_size must not be smaller than signal_spatial_size")
        scale = config.image_size // config.signal_spatial_size
        if config.image_size % config.signal_spatial_size or scale & (scale - 1):
            raise ValueError("image_size / signal_spatial_size must be a power of two")
        if config.decoder_refinement_blocks < 0:
            raise ValueError("decoder_refinement_blocks must be non-negative")
        channel_options = (
            config.signal_channels,
            config.label_channels,
            config.decoder_base_channels,
            config.decoder_min_channels,
        )
        if any(channels < 1 for channels in channel_options):
            raise ValueError("decoder channel counts must be positive")
        self.config = config
        self.z_encoder = (
            SignalEncoder(config.z_channels, config.signal_channels, config.signal_spatial_size)
            if config.use_z else None
        )
        self.u_encoder = (
            SignalEncoder(config.u_channels, config.signal_channels, config.signal_spatial_size)
            if config.use_u else None
        )
        self.gradient_encoder = (
            SignalEncoder(
                config.gradient_channels, config.signal_channels, config.signal_spatial_size
            )
            if config.use_gradient
            else None
        )
        self.label_encoder = nn.Sequential(
            nn.Linear(config.num_classes, config.label_channels), nn.ReLU(inplace=True)
        )
        signal_count = sum((config.use_z, config.use_u, config.use_gradient))
        channels = signal_count * config.signal_channels + config.label_channels
        base_groups = min(8, config.decoder_base_channels)
        while config.decoder_base_channels % base_groups:
            base_groups -= 1
        layers: list[nn.Module] = [
            nn.Conv2d(channels, config.decoder_base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(base_groups, config.decoder_base_channels),
            nn.ReLU(inplace=True),
        ]
        current_size = config.signal_spatial_size
        current_channels = config.decoder_base_channels
        while current_size < config.image_size:
            next_channels = max(config.decoder_min_channels, current_channels // 2)
            layers.append(_UpsampleBlock(current_channels, next_channels))
            current_channels = next_channels
            current_size *= 2
        layers.extend(
            _RefinementBlock(current_channels)
            for _ in range(config.decoder_refinement_blocks)
        )
        layers.extend([nn.Conv2d(current_channels, 3, kernel_size=3, padding=1), nn.Sigmoid()])
        self.image_decoder = nn.Sequential(*layers)

    def forward(
        self,
        smashed_z: Tensor,
        server_output_u: Tensor,
        grad_h_to_g: Tensor,
        label_condition: Tensor,
    ) -> Tensor:
        features: list[Tensor] = []
        if self.z_encoder is not None:
            features.append(self.z_encoder(smashed_z))
        if self.u_encoder is not None:
            features.append(self.u_encoder(server_output_u))
        if self.gradient_encoder is not None:
            features.append(self.gradient_encoder(l2_normalize_gradient(grad_h_to_g)))
        label = self.label_encoder(label_condition).unsqueeze(-1).unsqueeze(-1)
        label = label.expand(
            -1, -1, self.config.signal_spatial_size, self.config.signal_spatial_size
        )
        features.append(label)
        return self.image_decoder(torch.cat(features, dim=1))


__all__ = ["DecoderConfig", "LabelConditionedDecoder"]
