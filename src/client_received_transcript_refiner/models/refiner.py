from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional

from ...client_received_transcript_attack.models.decoder import SignalAdapter
from ...decoder.data.image_scaling import l2_normalize_gradient


def _group_count(channels: int, maximum: int = 8) -> int:
    groups = min(maximum, channels)
    while channels % groups:
        groups -= 1
    return groups


@dataclass(frozen=True)
class ResidualUNetRefinerConfig:
    """Architecture settings for the conservative image-space refiner."""

    image_channels: int = 3
    base_channels: int = 32
    bottleneck_blocks: int = 2
    max_residual: float = 0.1

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class _ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )


class _DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )
        self.refine = _ConvBlock(out_channels, out_channels)

    def forward(self, value: Tensor) -> Tensor:
        return self.refine(self.downsample(value))


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


class _UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )
        self.fuse = _ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, value: Tensor, skip: Tensor) -> Tensor:
        value = functional.interpolate(
            value, size=skip.shape[-2:], mode="bilinear", align_corners=False
        )
        value = self.project(value)
        return self.fuse(torch.cat((value, skip), dim=1))


class ResidualUNetRefiner(nn.Module):
    """Sharpen a coarse reconstruction while limiting how much it may be changed.

    The output head is zero-initialized, so a newly created model begins as the
    identity function. ``max_residual`` bounds every per-pixel correction and
    makes visually plausible but unsupported changes less likely.
    """

    def __init__(self, config: ResidualUNetRefinerConfig) -> None:
        super().__init__()
        self.refiner_type = "image_only"
        if config.image_channels < 1 or config.base_channels < 8:
            raise ValueError("image_channels must be positive and base_channels >= 8")
        if config.bottleneck_blocks < 0:
            raise ValueError("bottleneck_blocks must be non-negative")
        if not 0.0 < config.max_residual <= 1.0:
            raise ValueError("max_residual must be in (0, 1]")
        self.config = config
        base = config.base_channels
        self.encoder_1 = _ConvBlock(config.image_channels, base)
        self.encoder_2 = _DownBlock(base, base * 2)
        self.bottleneck_down = _DownBlock(base * 2, base * 4)
        self.bottleneck = nn.Sequential(
            *(_ResidualBlock(base * 4) for _ in range(config.bottleneck_blocks))
        )
        self.decoder_2 = _UpBlock(base * 4, base * 2, base * 2)
        self.decoder_1 = _UpBlock(base * 2, base, base)
        self.residual_head = nn.Conv2d(
            base, config.image_channels, kernel_size=3, padding=1
        )
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def forward(self, coarse: Tensor) -> tuple[Tensor, Tensor]:
        if coarse.ndim != 4 or coarse.shape[1] != self.config.image_channels:
            raise ValueError(
                "expected BCHW coarse reconstruction with "
                f"{self.config.image_channels} channels, found {tuple(coarse.shape)}"
            )
        encoder_1 = self.encoder_1(coarse)
        encoder_2 = self.encoder_2(encoder_1)
        bottleneck = self.bottleneck(self.bottleneck_down(encoder_2))
        decoded = self.decoder_2(bottleneck, encoder_2)
        decoded = self.decoder_1(decoded, encoder_1)
        residual = self.config.max_residual * torch.tanh(self.residual_head(decoded))
        refined = (coarse + residual).clamp(0.0, 1.0)
        return refined, residual


@dataclass(frozen=True)
class TranscriptConditionedRefinerConfig:
    """Configuration for a refiner conditioned on raw observed transcripts."""

    u_channels: int
    grad_z_channels: int
    image_channels: int = 3
    base_channels: int = 32
    condition_channels: int = 32
    condition_spatial_size: int = 16
    bottleneck_blocks: int = 2
    max_residual: float = 0.1

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class TranscriptConditionedResidualUNetRefiner(nn.Module):
    """Refine a coarse image while reusing attacker-visible u and dL/dz.

    The transcript adapters belong to the attacker model and do not depend on
    the victim architecture. Raw signals are fused at the image bottleneck so
    details discarded by the coarse RGB output can still influence refinement.
    """

    def __init__(self, config: TranscriptConditionedRefinerConfig) -> None:
        super().__init__()
        self.refiner_type = "transcript_conditioned"
        if config.u_channels < 1 or config.grad_z_channels < 1:
            raise ValueError("observed signal channel counts must be positive")
        if config.image_channels < 1 or config.base_channels < 8:
            raise ValueError("image_channels must be positive and base_channels >= 8")
        if config.condition_channels < 1 or config.condition_spatial_size < 1:
            raise ValueError("condition channel and spatial sizes must be positive")
        if config.bottleneck_blocks < 0:
            raise ValueError("bottleneck_blocks must be non-negative")
        if not 0.0 < config.max_residual <= 1.0:
            raise ValueError("max_residual must be in (0, 1]")
        self.config = config
        base = config.base_channels
        self.encoder_1 = _ConvBlock(config.image_channels, base)
        self.encoder_2 = _DownBlock(base, base * 2)
        self.bottleneck_down = _DownBlock(base * 2, base * 4)
        self.u_adapter = SignalAdapter(
            config.u_channels,
            config.condition_channels,
            config.condition_spatial_size,
        )
        self.grad_z_adapter = SignalAdapter(
            config.grad_z_channels,
            config.condition_channels,
            config.condition_spatial_size,
        )
        self.condition_fusion = _ConvBlock(
            base * 4 + 2 * config.condition_channels, base * 4
        )
        self.bottleneck = nn.Sequential(
            *(_ResidualBlock(base * 4) for _ in range(config.bottleneck_blocks))
        )
        self.decoder_2 = _UpBlock(base * 4, base * 2, base * 2)
        self.decoder_1 = _UpBlock(base * 2, base, base)
        self.residual_head = nn.Conv2d(
            base, config.image_channels, kernel_size=3, padding=1
        )
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def forward(
        self,
        coarse: Tensor,
        server_output_u: Tensor,
        grad_g_to_f: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if coarse.ndim != 4 or coarse.shape[1] != self.config.image_channels:
            raise ValueError(
                "expected BCHW coarse reconstruction with "
                f"{self.config.image_channels} channels, found {tuple(coarse.shape)}"
            )
        encoder_1 = self.encoder_1(coarse)
        encoder_2 = self.encoder_2(encoder_1)
        image_bottleneck = self.bottleneck_down(encoder_2)
        u_feature = self.u_adapter(server_output_u)
        grad_feature = self.grad_z_adapter(l2_normalize_gradient(grad_g_to_f))
        target_size = image_bottleneck.shape[-2:]
        if u_feature.shape[-2:] != target_size:
            u_feature = functional.interpolate(
                u_feature, size=target_size, mode="bilinear", align_corners=False
            )
        if grad_feature.shape[-2:] != target_size:
            grad_feature = functional.interpolate(
                grad_feature, size=target_size, mode="bilinear", align_corners=False
            )
        bottleneck = self.condition_fusion(
            torch.cat((image_bottleneck, u_feature, grad_feature), dim=1)
        )
        bottleneck = self.bottleneck(bottleneck)
        decoded = self.decoder_2(bottleneck, encoder_2)
        decoded = self.decoder_1(decoded, encoder_1)
        residual = self.config.max_residual * torch.tanh(self.residual_head(decoded))
        refined = (coarse + residual).clamp(0.0, 1.0)
        return refined, residual


__all__ = [
    "ResidualUNetRefiner",
    "ResidualUNetRefinerConfig",
    "TranscriptConditionedRefinerConfig",
    "TranscriptConditionedResidualUNetRefiner",
]
