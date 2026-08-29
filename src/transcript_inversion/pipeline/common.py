from __future__ import annotations

from pathlib import Path

import torch

from ...shared.reproducibility.random_seed import seed_everything
from ...split_learning.g_model.server_middle_g_model import ServerMiddleGModel
from ..models.decoder import BidirectionalTranscriptDecoder, DecoderConfig


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_server_middle_only(checkpoint_path: str | Path, device: torch.device):
    """Load only server-owned g from a split checkpoint; f/h are never instantiated."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint["model"]
    cut_config = str(state.get("cut_config", "middle"))
    server = ServerMiddleGModel(cut_config)
    server.load_state_dict(state["server_middle"])
    server.to(device).eval()
    return server, cut_config


def make_decoder(sample: dict, num_classes: int, image_size: int) -> BidirectionalTranscriptDecoder:
    has_grad_z = bool(sample["has_grad_g_to_f"])
    config = DecoderConfig(
        z_channels=int(sample["smashed_z"].shape[0]),
        u_channels=int(sample["server_output_u"].shape[0]),
        num_classes=num_classes,
        image_size=image_size,
        use_grad_z=has_grad_z,
        use_gated_z=has_grad_z,
    )
    return BidirectionalTranscriptDecoder(config)


__all__ = ["load_server_middle_only", "make_decoder", "resolve_device", "seed_everything"]
