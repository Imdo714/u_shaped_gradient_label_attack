from __future__ import annotations

from pathlib import Path

import torch

from .decoder import ClientReceivedDecoder, ClientReceivedDecoderConfig, ZUGradZDecoder
from .multiscale_decoder import (
    MultiscaleDecoderConfig,
    MultiscalePixelShuffleDecoder,
)
from .residual_detail_decoder import (
    ResidualDetailDecoder,
    ResidualDetailDecoderConfig,
)


DecoderModel = (
    ClientReceivedDecoder
    | MultiscalePixelShuffleDecoder
    | ResidualDetailDecoder
    | ZUGradZDecoder
)
DecoderConfig = (
    ClientReceivedDecoderConfig | MultiscaleDecoderConfig | ResidualDetailDecoderConfig
)


def decoder_from_config(decoder_type: str, config: dict) -> DecoderModel:
    if decoder_type in {"baseline", "baseline_bilinear"}:
        return ClientReceivedDecoder(ClientReceivedDecoderConfig(**config))
    if decoder_type == "multiscale_pixelshuffle":
        return MultiscalePixelShuffleDecoder(MultiscaleDecoderConfig(**config))
    if decoder_type == "residual_detail":
        return ResidualDetailDecoder(ResidualDetailDecoderConfig(**config))
    if decoder_type == "z_u_grad_z_bilinear":
        return ZUGradZDecoder(ClientReceivedDecoderConfig(**config))
    raise ValueError(f"unsupported decoder architecture: {decoder_type}")


def load_decoder_checkpoint(
    checkpoint_path: str | Path, device: torch.device
) -> tuple[DecoderModel, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    decoder_type = str(checkpoint.get("decoder_type", "baseline_bilinear"))
    decoder = decoder_from_config(decoder_type, checkpoint["decoder_config"]).to(device)
    decoder.load_state_dict(checkpoint["model"])
    decoder.eval()
    return decoder, checkpoint


__all__ = [
    "DecoderConfig",
    "DecoderModel",
    "decoder_from_config",
    "load_decoder_checkpoint",
]
