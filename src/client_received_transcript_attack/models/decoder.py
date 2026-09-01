from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from ...decoder.data.image_scaling import l2_normalize_gradient


def _group_count(channels: int, maximum: int = 8) -> int:
    groups = min(maximum, channels)
    while channels % groups:
        groups -= 1
    return groups


@dataclass(frozen=True)
class ClientReceivedDecoderConfig:
    u_channels: int
    grad_z_channels: int
    num_classes: int
    image_size: int = 64
    signal_spatial_size: int = 16
    signal_channels: int = 64
    decoder_base_channels: int = 256
    decoder_min_channels: int = 32
    refinement_blocks: int = 1
    use_label_head: bool = False
    label_channels: int = 32

    def to_dict(self) -> dict[str, int | bool]:
        return asdict(self)


class SignalAdapter(nn.Module):
    """Map an observed four-dimensional tensor to an architecture-neutral grid."""

    def __init__(self, in_channels: int, out_channels: int, spatial_size: int) -> None:
        super().__init__()
        self.spatial_size = spatial_size
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, value: Tensor) -> Tensor:
        if value.ndim != 4:
            raise ValueError(f"expected BCHW tensor, found shape {tuple(value.shape)}")
        pooled = torch.nn.functional.adaptive_avg_pool2d(
            value, (self.spatial_size, self.spatial_size)
        )
        return self.network(pooled)


class _UpsampleBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(channels), channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(channels), channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, value: Tensor) -> Tensor:
        return self.activation(value + self.network(value))


class ClientReceivedDecoder(nn.Module):
    """Decode only server output u and server-to-client gradient dL/dz."""

    def __init__(self, config: ClientReceivedDecoderConfig) -> None:
        super().__init__()
        if config.u_channels < 1 or config.grad_z_channels < 1:
            raise ValueError("observed signal channel counts must be positive")
        if config.num_classes < 2:
            raise ValueError("num_classes must be at least two")
        if config.signal_spatial_size < 1 or config.image_size < config.signal_spatial_size:
            raise ValueError("invalid signal or image spatial size")
        scale = config.image_size // config.signal_spatial_size
        if config.image_size % config.signal_spatial_size or scale & (scale - 1):
            raise ValueError("image_size / signal_spatial_size must be a power of two")
        if config.refinement_blocks < 0:
            raise ValueError("refinement_blocks must be non-negative")
        self.config = config
        self.u_encoder = SignalAdapter(
            config.u_channels, config.signal_channels, config.signal_spatial_size
        )
        self.grad_z_encoder = SignalAdapter(
            config.grad_z_channels, config.signal_channels, config.signal_spatial_size
        )
        self.label_classifier: nn.Module | None = None
        self.label_encoder: nn.Module | None = None
        fusion_channels = 2 * config.signal_channels
        if config.use_label_head:
            self.label_classifier = nn.Linear(config.signal_channels, config.num_classes)
            self.label_encoder = nn.Sequential(
                nn.Linear(config.num_classes, config.label_channels),
                nn.SiLU(inplace=True),
            )
            fusion_channels += config.label_channels

        layers: list[nn.Module] = [
            nn.Conv2d(
                fusion_channels, config.decoder_base_channels, kernel_size=3, padding=1
            ),
            nn.GroupNorm(
                _group_count(config.decoder_base_channels), config.decoder_base_channels
            ),
            nn.SiLU(inplace=True),
        ]
        current_size = config.signal_spatial_size
        current_channels = config.decoder_base_channels
        while current_size < config.image_size:
            next_channels = max(config.decoder_min_channels, current_channels // 2)
            layers.append(_UpsampleBlock(current_channels, next_channels))
            current_channels = next_channels
            current_size *= 2
        layers.extend(_ResidualBlock(current_channels) for _ in range(config.refinement_blocks))
        layers.extend(
            [nn.Conv2d(current_channels, 3, kernel_size=3, padding=1), nn.Sigmoid()]
        )
        self.image_decoder = nn.Sequential(*layers)

    def forward(
        self, server_output_u: Tensor, grad_g_to_f: Tensor
    ) -> tuple[Tensor, Tensor | None]:
        u_feature = self.u_encoder(server_output_u)
        grad_feature = self.grad_z_encoder(l2_normalize_gradient(grad_g_to_f))
        features = [u_feature, grad_feature]
        label_logits: Tensor | None = None
        if self.label_classifier is not None and self.label_encoder is not None:
            pooled_gradient = grad_feature.mean(dim=(-2, -1))
            label_logits = self.label_classifier(pooled_gradient)
            soft_label = torch.softmax(label_logits, dim=1)
            label_feature = self.label_encoder(soft_label)[:, :, None, None]
            features.append(
                label_feature.expand(
                    -1,
                    -1,
                    self.config.signal_spatial_size,
                    self.config.signal_spatial_size,
                )
            )
        reconstruction = self.image_decoder(torch.cat(features, dim=1))
        return reconstruction, label_logits


__all__ = [
    "ClientReceivedDecoder",
    "ClientReceivedDecoderConfig",
    "SignalAdapter",
]
