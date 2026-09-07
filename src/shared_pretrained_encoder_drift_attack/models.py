from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from ..decoder.data.image_scaling import l2_normalize_gradient


def _groups(channels: int) -> int:
    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return groups


@dataclass(frozen=True)
class ReconstructionDecoderConfig:
    z_channels: int
    image_size: int
    grad_channels: int | None = None
    use_z: bool = True
    use_gradient: bool = False
    signal_spatial_size: int = 16
    signal_channels: int = 64
    base_channels: int = 256
    min_channels: int = 32

    def to_dict(self) -> dict:
        return asdict(self)


class _Adapter(nn.Module):
    def __init__(self, inputs: int, outputs: int, spatial_size: int) -> None:
        super().__init__()
        self.spatial_size = spatial_size
        self.network = nn.Sequential(
            nn.Conv2d(inputs, outputs, 1),
            nn.GroupNorm(_groups(outputs), outputs),
            nn.SiLU(),
            nn.Conv2d(outputs, outputs, 3, padding=1),
            nn.GroupNorm(_groups(outputs), outputs),
            nn.SiLU(),
        )

    def forward(self, value: Tensor) -> Tensor:
        value = torch.nn.functional.adaptive_avg_pool2d(
            value, (self.spatial_size, self.spatial_size)
        )
        return self.network(value)


class ReconstructionDecoder(nn.Module):
    """Separate z/gradient branches followed by a shared image decoder."""

    def __init__(self, config: ReconstructionDecoderConfig) -> None:
        super().__init__()
        if not config.use_z and not config.use_gradient:
            raise ValueError("at least one transcript signal must be enabled")
        if config.use_gradient and not config.grad_channels:
            raise ValueError("gradient condition requires grad_channels")
        ratio = config.image_size // config.signal_spatial_size
        if (
            config.image_size % config.signal_spatial_size
            or ratio < 1
            or ratio & (ratio - 1)
        ):
            raise ValueError("image_size / signal_spatial_size must be a power of two")
        self.config = config
        self.z_encoder = (
            _Adapter(config.z_channels, config.signal_channels, config.signal_spatial_size)
            if config.use_z
            else None
        )
        self.gradient_encoder = (
            _Adapter(
                int(config.grad_channels),
                config.signal_channels,
                config.signal_spatial_size,
            )
            if config.use_gradient
            else None
        )
        fusion_channels = config.signal_channels * (
            int(config.use_z) + int(config.use_gradient)
        )
        layers: list[nn.Module] = [
            nn.Conv2d(fusion_channels, config.base_channels, 3, padding=1),
            nn.GroupNorm(_groups(config.base_channels), config.base_channels),
            nn.SiLU(),
        ]
        size = config.signal_spatial_size
        channels = config.base_channels
        while size < config.image_size:
            next_channels = max(config.min_channels, channels // 2)
            layers.extend(
                [
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(channels, next_channels, 3, padding=1),
                    nn.GroupNorm(_groups(next_channels), next_channels),
                    nn.SiLU(),
                ]
            )
            channels = next_channels
            size *= 2
        layers.extend((nn.Conv2d(channels, 3, 3, padding=1), nn.Sigmoid()))
        self.decoder = nn.Sequential(*layers)

    def forward(self, z: Tensor | None, gradient: Tensor | None = None) -> Tensor:
        features: list[Tensor] = []
        if self.z_encoder is not None:
            if z is None:
                raise ValueError("z is required by this decoder")
            features.append(self.z_encoder(z))
        if self.gradient_encoder is not None:
            if gradient is None:
                raise ValueError("gradient is required by this decoder")
            features.append(self.gradient_encoder(l2_normalize_gradient(gradient)))
        return self.decoder(torch.cat(features, dim=1))


def decoder_for_condition(
    condition: str,
    z_channels: int,
    image_size: int,
    signal_spatial_size: int = 16,
    signal_channels: int = 64,
    base_channels: int = 256,
    min_channels: int = 32,
) -> ReconstructionDecoder:
    normalized = condition.lower().replace("+", "_")
    settings = {
        "z_only": (True, False),
        "gradient_only": (False, True),
        "z_gradient": (True, True),
        "z_grad": (True, True),
    }
    if normalized not in settings:
        raise ValueError(f"unknown attack condition: {condition}")
    use_z, use_gradient = settings[normalized]
    return ReconstructionDecoder(
        ReconstructionDecoderConfig(
            z_channels=z_channels,
            grad_channels=z_channels,
            image_size=image_size,
            use_z=use_z,
            use_gradient=use_gradient,
            signal_spatial_size=signal_spatial_size,
            signal_channels=signal_channels,
            base_channels=base_channels,
            min_channels=min_channels,
        )
    )


__all__ = [
    "ReconstructionDecoder",
    "ReconstructionDecoderConfig",
    "decoder_for_condition",
]
