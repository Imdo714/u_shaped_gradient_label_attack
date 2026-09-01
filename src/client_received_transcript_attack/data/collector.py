from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from torch import Tensor, nn

from ...decoder.data.image_scaling import denormalize_image
from ...split_learning.architecture.split_learning_model import SplitLearningModel
from ...split_learning.gradient_flow.gradient_exchange import (
    observe_frozen_gradient_exchange,
)


@dataclass(frozen=True)
class ReceivedTranscript:
    """Only the two server-to-client tensors allowed by the threat model."""

    server_output_u: Tensor
    grad_g_to_f: Tensor


class ReceivedTranscriptProvider(Protocol):
    """Black-box boundary used by the collector.

    A real deployment can implement this protocol with an authorized RPC client.
    The decoder and dataset code never receive f, g, h, z, or dL/du.
    """

    @property
    def provider_name(self) -> str: ...

    def observe(self, image: Tensor, label: Tensor) -> ReceivedTranscript: ...


class FrozenSplitLearningProvider:
    """Local laboratory adapter for a fixed U-shaped Split Learning checkpoint."""

    def __init__(self, model: SplitLearningModel) -> None:
        self._model = model
        self._criterion = nn.CrossEntropyLoss()

    @property
    def provider_name(self) -> str:
        return f"frozen-local-{self._model.cut_config}"

    def observe(self, image: Tensor, label: Tensor) -> ReceivedTranscript:
        exchange = observe_frozen_gradient_exchange(
            self._model, image, label, self._criterion
        )
        return ReceivedTranscript(
            server_output_u=exchange.server_output_u.detach(),
            grad_g_to_f=exchange.grad_g_to_f.detach(),
        )


@dataclass(frozen=True)
class CollectionManifests:
    attacker: Path
    evaluator: Path


def _opaque_transcript_id(sample_id: str, collection_name: str) -> str:
    digest = hashlib.sha256(
        f"{collection_name}\0{sample_id}".encode("utf-8")
    ).hexdigest()[:24]
    return f"transcript_{digest}"


def _write_manifest(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collect_received_transcripts(
    provider: ReceivedTranscriptProvider,
    loader,
    output_dir: str | Path,
    collection_name: str,
    device: torch.device,
    max_samples: int | None = None,
) -> CollectionManifests:
    """Collect paired public data while physically separating attacker and evaluator files."""

    if max_samples is not None and max_samples < 1:
        raise ValueError("max_samples must be positive")
    root = Path(output_dir)
    attacker_dir = root / "attacker_records"
    evaluator_dir = root / "evaluator_targets"
    attacker_dir.mkdir(parents=True, exist_ok=True)
    evaluator_dir.mkdir(parents=True, exist_ok=True)
    attacker_rows: list[dict[str, str]] = []
    evaluator_rows: list[dict[str, str]] = []
    collected = 0

    for images, labels, sample_ids in loader:
        for index, sample_id in enumerate(sample_ids):
            if max_samples is not None and collected >= max_samples:
                break
            image = images[index : index + 1].to(device)
            label = labels[index : index + 1].to(device)
            observed = provider.observe(image, label)
            transcript_id = _opaque_transcript_id(str(sample_id), collection_name)
            filename = f"{transcript_id}.npz"

            np.savez_compressed(
                attacker_dir / filename,
                server_output_u=observed.server_output_u[0].cpu().numpy(),
                grad_g_to_f=observed.grad_g_to_f[0].cpu().numpy(),
            )
            np.savez_compressed(
                evaluator_dir / filename,
                target_image=denormalize_image(image[0]).cpu().numpy(),
                true_label=np.int64(label.item()),
            )
            attacker_rows.append(
                {
                    "transcript_id": transcript_id,
                    "attacker_record": f"attacker_records/{filename}",
                }
            )
            evaluator_rows.append(
                {
                    "transcript_id": transcript_id,
                    "evaluator_target": f"evaluator_targets/{filename}",
                }
            )
            collected += 1
        if max_samples is not None and collected >= max_samples:
            break

    if not attacker_rows:
        raise ValueError("no transcripts were collected")

    attacker_manifest = root / "attacker_manifest.csv"
    evaluator_manifest = root / "evaluator_manifest.csv"
    _write_manifest(
        attacker_manifest,
        ("transcript_id", "attacker_record"),
        attacker_rows,
    )
    _write_manifest(
        evaluator_manifest,
        ("transcript_id", "evaluator_target"),
        evaluator_rows,
    )
    with (root / "collection_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "collection_name": collection_name,
                "provider": provider.provider_name,
                "samples": collected,
                "attacker_visible_signals": ["u", "dL/dz"],
                "excluded_signals": ["x", "y", "z", "dL/du"],
            },
            handle,
            indent=2,
        )
    return CollectionManifests(attacker_manifest, evaluator_manifest)


__all__ = [
    "CollectionManifests",
    "FrozenSplitLearningProvider",
    "ReceivedTranscript",
    "ReceivedTranscriptProvider",
    "collect_received_transcripts",
]
