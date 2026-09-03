from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from ...client_received_transcript_attack.models.factory import (
    DecoderModel,
    load_decoder_checkpoint,
)
from ..models.refiner import (
    ResidualUNetRefiner,
    ResidualUNetRefinerConfig,
    TranscriptConditionedRefinerConfig,
    TranscriptConditionedResidualUNetRefiner,
)


def device_from_name(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_decoder(
    checkpoint_path: str | Path, device: torch.device
) -> DecoderModel:
    decoder, _ = load_decoder_checkpoint(checkpoint_path, device)
    decoder.eval()
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)
    return decoder


def load_refiner(
    checkpoint_path: str | Path, device: torch.device
) -> tuple[
    ResidualUNetRefiner | TranscriptConditionedResidualUNetRefiner, dict
]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    refiner_type = checkpoint.get("refiner_type", "image_only")
    if refiner_type == "image_only":
        config = ResidualUNetRefinerConfig(**checkpoint["refiner_config"])
        refiner = ResidualUNetRefiner(config).to(device)
    elif refiner_type == "transcript_conditioned":
        config = TranscriptConditionedRefinerConfig(**checkpoint["refiner_config"])
        refiner = TranscriptConditionedResidualUNetRefiner(config).to(device)
    else:
        raise ValueError(f"unsupported refiner type: {refiner_type}")
    refiner.load_state_dict(checkpoint["model"])
    refiner.eval()
    return refiner, checkpoint


def transcript_ids(dataset) -> set[str]:
    return {str(row["transcript_id"]) for row in dataset.rows}


def require_disjoint_datasets(first, second, description: str) -> None:
    overlap = transcript_ids(first) & transcript_ids(second)
    if overlap:
        examples = ", ".join(sorted(overlap)[:3])
        raise ValueError(
            f"{description} overlap on {len(overlap)} transcript IDs; examples: {examples}"
        )


__all__ = [
    "device_from_name",
    "file_sha256",
    "load_frozen_decoder",
    "load_refiner",
    "require_disjoint_datasets",
    "transcript_ids",
]
