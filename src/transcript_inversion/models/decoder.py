from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional

from .signal_encoder import SignalEncoder, group_count


def _normalized(value: Tensor, epsilon: float = 1e-8) -> Tensor:
    flat = value.flatten(start_dim=1)
    norm = flat.norm(dim=1, keepdim=True).clamp_min(epsilon)
    return value / norm.view(value.shape[0], *([1] * (value.ndim - 1)))


@dataclass(frozen=True)
class DecoderConfig:
    z_channels: int
    u_channels: int
    num_classes: int
    image_size: int = 64
    signal_spatial_size: int = 8
    signal_channels: int = 32
    label_channels: int = 16
    decoder_base_channels: int = 128
    decoder_min_channels: int = 16
    use_z: bool = True
    use_u: bool = True
    use_grad_u: bool = True
    use_grad_z: bool = True
    use_gated_z: bool = True

    def to_dict(self) -> dict[str, int | bool]:
        return asdict(self)


class _UpsampleBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )


class BidirectionalTranscriptDecoder(nn.Module):
    """Decode z/u and both backward gradients into an image reconstruction."""

    def __init__(self, config: DecoderConfig) -> None:
        super().__init__()
        flags = (
            config.use_z,
            config.use_u,
            config.use_grad_u,
            config.use_grad_z,
            config.use_gated_z,
        )
        if not any(flags):
            raise ValueError("at least one transcript signal must be enabled")
        self.config = config
        self.z_encoder = self._encoder(config.z_channels, config.use_z)
        self.u_encoder = self._encoder(config.u_channels, config.use_u)
        self.grad_u_encoder = self._encoder(config.u_channels, config.use_grad_u)
        self.grad_z_encoder = self._encoder(config.z_channels, config.use_grad_z)
        self.gated_z_encoder = self._encoder(config.z_channels, config.use_gated_z)
        self.label_encoder = nn.Sequential(
            nn.Linear(config.num_classes, config.label_channels), nn.SiLU(inplace=True)
        )
        input_channels = sum(flags) * config.signal_channels + config.label_channels
        layers: list[nn.Module] = [
            nn.Conv2d(input_channels, config.decoder_base_channels, 3, padding=1),
            nn.GroupNorm(
                group_count(config.decoder_base_channels), config.decoder_base_channels
            ),
            nn.SiLU(inplace=True),
        ]
        size = config.signal_spatial_size
        channels = config.decoder_base_channels
        while size < config.image_size:
            next_channels = max(config.decoder_min_channels, channels // 2)
            layers.append(_UpsampleBlock(channels, next_channels))
            channels = next_channels
            size *= 2
        layers.extend([nn.Conv2d(channels, 3, 3, padding=1), nn.Sigmoid()])
        self.image_decoder = nn.Sequential(*layers)

    def _encoder(self, channels: int, enabled: bool) -> SignalEncoder | None:
        if not enabled:
            return None
        return SignalEncoder(
            channels, self.config.signal_channels, self.config.signal_spatial_size
        )

    def forward(
        self,
        smashed_z: Tensor,
        server_output_u: Tensor,
        grad_h_to_g: Tensor,
        grad_g_to_f: Tensor,
        label_condition: Tensor,
    ) -> Tensor:
        features: list[Tensor] = []
        if self.z_encoder is not None:
            features.append(self.z_encoder(smashed_z))
        if self.u_encoder is not None:
            features.append(self.u_encoder(server_output_u))
        if self.grad_u_encoder is not None:
            features.append(self.grad_u_encoder(_normalized(grad_h_to_g)))
        if self.grad_z_encoder is not None:
            features.append(self.grad_z_encoder(_normalized(grad_g_to_f)))
        if self.gated_z_encoder is not None:
            features.append(self.gated_z_encoder(smashed_z * _normalized(grad_g_to_f)))
        label = self.label_encoder(label_condition)[:, :, None, None]
        label = label.expand(
            -1, -1, self.config.signal_spatial_size, self.config.signal_spatial_size
        )
        features.append(label)
        output = self.image_decoder(torch.cat(features, dim=1))
        if output.shape[-2:] != (self.config.image_size, self.config.image_size):
            output = functional.interpolate(
                output,
                size=(self.config.image_size, self.config.image_size),
                mode="bilinear",
                align_corners=False,
            )
        return output


__all__ = ["BidirectionalTranscriptDecoder", "DecoderConfig"]
