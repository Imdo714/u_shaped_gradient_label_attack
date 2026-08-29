from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
from torch import Tensor


def _record_name(sample_id: str) -> str:
    return f"{hashlib.sha256(str(sample_id).encode('utf-8')).hexdigest()[:24]}.npz"


class ServerTranscriptWriter:
    """Persist attacker-visible values; this API cannot receive original images."""

    fieldnames = ["sample_id", "attacker_record"]

    def __init__(self, output_dir: str | Path) -> None:
        self.root = Path(output_dir)
        self.records = self.root / "attacker_records"
        self.records.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict[str, str]] = []

    def write(
        self,
        sample_id: str,
        smashed_z: Tensor,
        server_output_u: Tensor,
        grad_h_to_g: Tensor,
        grad_g_to_f: Tensor,
        label_condition: Tensor,
        predicted_label: int,
        confidence: float,
        cluster_id: int = -1,
    ) -> None:
        name = _record_name(sample_id)
        np.savez_compressed(
            self.records / name,
            smashed_z=smashed_z.detach().cpu().numpy(),
            server_output_u=server_output_u.detach().cpu().numpy(),
            grad_h_to_g=grad_h_to_g.detach().cpu().numpy(),
            grad_g_to_f=grad_g_to_f.detach().cpu().numpy(),
            label_condition=label_condition.detach().cpu().numpy(),
            predicted_label=np.int64(predicted_label),
            confidence=np.float32(confidence),
            cluster_id=np.int64(cluster_id),
        )
        self.rows.append(
            {"sample_id": str(sample_id), "attacker_record": f"attacker_records/{name}"}
        )

    def finalize(self) -> Path:
        manifest = self.root / "attacker_manifest.csv"
        with manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)
        return manifest


class EvaluatorTargetWriter:
    """Persist protected targets through a separate evaluator-only interface."""

    def __init__(self, output_dir: str | Path) -> None:
        self.root = Path(output_dir)
        self.targets = self.root / "evaluator_targets"
        self.targets.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict[str, str]] = []

    def write(self, sample_id: str, target_image: Tensor, true_label: int) -> None:
        name = _record_name(sample_id)
        np.savez_compressed(
            self.targets / name,
            target_image=target_image.detach().cpu().numpy(),
            true_label=np.int64(true_label),
        )
        self.rows.append(
            {"sample_id": str(sample_id), "evaluator_target": f"evaluator_targets/{name}"}
        )

    def finalize(self) -> Path:
        manifest = self.root / "evaluator_manifest.csv"
        with manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["sample_id", "evaluator_target"]
            )
            writer.writeheader()
            writer.writerows(self.rows)
        return manifest


__all__ = ["EvaluatorTargetWriter", "ServerTranscriptWriter"]
