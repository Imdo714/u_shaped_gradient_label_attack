from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional

from ...decoder.data.image_scaling import l2_normalize_gradient
from .decoder import SignalAdapter


def _group_count(channels: int, maximum: int = 8) -> int:
    groups = min(maximum, channels)
    while channels % groups:
        groups -= 1
    return groups


def _icnr_initialize(convolution: nn.Conv2d, scale_factor: int = 2) -> None:
    """Initialize sub-pixel convolution to avoid checkerboard startup artifacts."""

    output_channels, input_channels, height, width = convolution.weight.shape
    repeats = scale_factor**2
    if output_channels % repeats:
        raise ValueError("PixelShuffle convolution channels are not divisible by scale²")
    subkernel = convolution.weight.new_empty(
        output_channels // repeats, input_channels, height, width
    )
    nn.init.kaiming_normal_(subkernel)
    with torch.no_grad():
        convolution.weight.copy_(subkernel.repeat_interleave(repeats, dim=0))
        if convolution.bias is not None:
            convolution.bias.zero_()


@dataclass(frozen=True)
class MultiscaleDecoderConfig:
    u_channels: int
    grad_z_channels: int
    num_classes: int
    image_size: int = 128
    signal_spatial_size: int = 16
    signal_channels: int = 64
    decoder_base_channels: int = 256
    decoder_min_channels: int = 32
    refinement_blocks: int = 1
    use_label_head: bool = False
    label_channels: int = 32
    film_strength: float = 0.1

    def to_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)


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


class _ConditionalPixelShuffleBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        condition_channels: int,
        refinement_blocks: int,
        film_strength: float,
    ) -> None:
        super().__init__()
        self.film_strength = film_strength
        subpixel_convolution = nn.Conv2d(
            in_channels, out_channels * 4, kernel_size=3, padding=1
        )
        _icnr_initialize(subpixel_convolution)
        self.upsample = nn.Sequential(
            subpixel_convolution,
            nn.PixelShuffle(2),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )
        self.condition_projection = nn.Conv2d(
            condition_channels, 2 * out_channels, kernel_size=1
        )
        self.refinement = nn.Sequential(
            *(_ResidualBlock(out_channels) for _ in range(refinement_blocks))
        )

    def forward(self, value: Tensor, condition: Tensor) -> Tensor:
        value = self.upsample(value)
        resized_condition = functional.interpolate(
            condition,
            size=value.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        gamma, beta = self.condition_projection(resized_condition).chunk(2, dim=1)
        strength = self.film_strength
        value = value * (1.0 + strength * torch.tanh(gamma))
        value = value + strength * torch.tanh(beta)
        return self.refinement(value)


class MultiscalePixelShuffleDecoder(nn.Module):
    """Decode u/dL-dz with learned upsampling and conditioning at every scale."""

    decoder_type = "multiscale_pixelshuffle"

    def __init__(self, config: MultiscaleDecoderConfig) -> None:
        super().__init__()
        if config.u_channels < 1 or config.grad_z_channels < 1:
            raise ValueError("observed signal channel counts must be positive")
        if config.num_classes < 2:
            raise ValueError("num_classes must be at least two")
        scale = config.image_size // config.signal_spatial_size
        if (
            config.signal_spatial_size < 1
            or config.image_size % config.signal_spatial_size
            or scale < 1
            or scale & (scale - 1)
        ):
            raise ValueError("image_size / signal_spatial_size must be a power of two")
        if config.refinement_blocks < 0:
            raise ValueError("refinement_blocks must be non-negative")
        if not 0.0 <= config.film_strength <= 1.0:
            raise ValueError("film_strength must be in [0, 1]")
        self.config = config
        self.u_encoder = SignalAdapter(
            config.u_channels, config.signal_channels, config.signal_spatial_size
        )
        self.grad_z_encoder = SignalAdapter(
            config.grad_z_channels,
            config.signal_channels,
            config.signal_spatial_size,
        )
        condition_channels = 2 * config.signal_channels
        self.label_classifier: nn.Module | None = None
        self.label_encoder: nn.Module | None = None
        initial_channels = condition_channels
        if config.use_label_head:
            self.label_classifier = nn.Linear(config.signal_channels, config.num_classes)
            self.label_encoder = nn.Sequential(
                nn.Linear(config.num_classes, config.label_channels),
                nn.SiLU(inplace=True),
            )
            initial_channels += config.label_channels

        self.initial_projection = nn.Sequential(
            nn.Conv2d(
                initial_channels,
                config.decoder_base_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.GroupNorm(
                _group_count(config.decoder_base_channels),
                config.decoder_base_channels,
            ),
            nn.SiLU(inplace=True),
            _ResidualBlock(config.decoder_base_channels),
        )
        stages: list[nn.Module] = []
        current_channels = config.decoder_base_channels
        current_size = config.signal_spatial_size
        while current_size < config.image_size:
            next_channels = max(config.decoder_min_channels, current_channels // 2)
            stages.append(
                _ConditionalPixelShuffleBlock(
                    current_channels,
                    next_channels,
                    condition_channels,
                    config.refinement_blocks,
                    config.film_strength,
                )
            )
            current_channels = next_channels
            current_size *= 2
        self.upsampling_stages = nn.ModuleList(stages)
        self.output_head = nn.Sequential(
            nn.Conv2d(current_channels, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(
        self, server_output_u: Tensor, grad_g_to_f: Tensor
    ) -> tuple[Tensor, Tensor | None]:
        u_feature = self.u_encoder(server_output_u)
        grad_feature = self.grad_z_encoder(l2_normalize_gradient(grad_g_to_f))
        condition = torch.cat((u_feature, grad_feature), dim=1)
        initial_features = [condition]
        label_logits: Tensor | None = None
        if self.label_classifier is not None and self.label_encoder is not None:
            label_logits = self.label_classifier(grad_feature.mean(dim=(-2, -1)))
            soft_label = torch.softmax(label_logits, dim=1)
            label_feature = self.label_encoder(soft_label)[:, :, None, None]
            initial_features.append(
                label_feature.expand(
                    -1,
                    -1,
                    self.config.signal_spatial_size,
                    self.config.signal_spatial_size,
                )
            )
        value = self.initial_projection(torch.cat(initial_features, dim=1))
        for stage in self.upsampling_stages:
            value = stage(value, condition)
        return self.output_head(value), label_logits


__all__ = ["MultiscaleDecoderConfig", "MultiscalePixelShuffleDecoder"]
