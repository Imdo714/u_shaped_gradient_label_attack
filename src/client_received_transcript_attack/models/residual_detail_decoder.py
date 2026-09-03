from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .decoder import ClientReceivedDecoder, ClientReceivedDecoderConfig


@dataclass(frozen=True)
class ResidualDetailDecoderConfig(ClientReceivedDecoderConfig):
    """Configuration for a small detail branch attached to the bilinear decoder."""

    detail_condition_channels: int = 8
    detail_channels: int = 16
    detail_scale: float = 0.25


class ResidualDetailDecoder(ClientReceivedDecoder):
    """Bilinear decoder with a bounded, transcript-conditioned RGB residual.

    The inherited decoder remains state-dict compatible with a baseline checkpoint.
    The final convolution is zero-initialized, so loading a trained baseline produces
    exactly the baseline reconstruction before fine-tuning starts.
    """

    decoder_type = "residual_detail"

    def __init__(self, config: ResidualDetailDecoderConfig) -> None:
        if config.detail_condition_channels < 1 or config.detail_channels < 1:
            raise ValueError("detail branch channel counts must be positive")
        if config.detail_scale <= 0.0:
            raise ValueError("detail_scale must be positive")
        super().__init__(config)
        self.config = config
        self.u_detail_projection = nn.Conv2d(
            config.signal_channels, config.detail_condition_channels, kernel_size=1
        )
        self.grad_z_detail_projection = nn.Conv2d(
            config.signal_channels, config.detail_condition_channels, kernel_size=1
        )
        condition_channels = 3 + 2 * config.detail_condition_channels
        self.detail_head = nn.Sequential(
            nn.Conv2d(
                condition_channels, config.detail_channels, kernel_size=3, padding=1
            ),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                config.detail_channels, config.detail_channels, kernel_size=3, padding=1
            ),
            nn.SiLU(inplace=True),
            nn.Conv2d(config.detail_channels, 3, kernel_size=3, padding=1),
        )
        final = self.detail_head[-1]
        if not isinstance(final, nn.Conv2d):
            raise TypeError("detail head must end with a convolution")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(
        self,
        server_output_u: Tensor,
        grad_g_to_f: Tensor,
        smashed_z: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        features, u_feature, grad_feature, label_logits = self._encode_observations(
            server_output_u, grad_g_to_f, smashed_z
        )
        coarse = self.image_decoder(torch.cat(features, dim=1))
        output_size = coarse.shape[-2:]
        u_condition = F.interpolate(
            self.u_detail_projection(u_feature),
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        grad_condition = F.interpolate(
            self.grad_z_detail_projection(grad_feature),
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        detail = torch.tanh(
            self.detail_head(torch.cat((coarse, u_condition, grad_condition), dim=1))
        )
        coarse_logits = torch.logit(coarse.clamp(1e-5, 1.0 - 1e-5))
        reconstruction = torch.sigmoid(
            coarse_logits + self.config.detail_scale * detail
        )
        return reconstruction, label_logits


__all__ = ["ResidualDetailDecoder", "ResidualDetailDecoderConfig"]
