from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from torch import Tensor


class ServerGradientTranscriptLogger:
    """Record only gradients and activations visible to the server.

    The log API deliberately accepts no label.
    """

    FIELDNAMES = ("sample_id", "epoch", "batch_id", "record_file")

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root) / "attacker_transcript"
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.csv"
        if not self.index_path.exists():
            with self.index_path.open("w", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=self.FIELDNAMES).writeheader()

    def log(
        self,
        sample_id: str,
        epoch: int,
        batch_id: int,
        smashed_z: Tensor,
        server_output_u: Tensor,
        grad_h_to_g: Tensor,
        grad_g_to_f: Tensor,
    ) -> None:
        safe_id = sample_id.replace("/", "__").replace("\\", "__")
        epoch_dir = self.root / f"epoch_{epoch:03d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        record_file = epoch_dir / f"batch_{batch_id:06d}__{safe_id}.npz"
        raw = grad_h_to_g.detach().cpu().numpy().reshape(-1).astype(np.float32)
        normalized = raw / (np.linalg.norm(raw) + 1e-12)
        np.savez_compressed(
            record_file,
            smashed_z=smashed_z.detach().cpu().numpy().astype(np.float32),
            server_output_u=server_output_u.detach().cpu().numpy().astype(np.float32),
            grad_h_to_g=grad_h_to_g.detach().cpu().numpy().astype(np.float32),
            grad_g_to_f=grad_g_to_f.detach().cpu().numpy().astype(np.float32),
            raw_gradient=raw,
            normalized_gradient=normalized,
        )
        with self.index_path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=self.FIELDNAMES).writerow(
                {
                    "sample_id": sample_id,
                    "epoch": epoch,
                    "batch_id": batch_id,
                    "record_file": record_file.relative_to(self.root).as_posix(),
                }
            )


class EvaluatorGroundTruthLogger:
    """Separate evaluator-only storage; attack code never reads this during fitting."""

    FIELDNAMES = ("sample_id", "epoch", "true_label")

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root) / "evaluator_ground_truth"
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "ground_truth.csv"
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=self.FIELDNAMES).writeheader()

    def log(self, sample_id: str, epoch: int, true_label: int) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=self.FIELDNAMES).writerow(
                {"sample_id": sample_id, "epoch": epoch, "true_label": true_label}
            )


# Compatibility alias for code using the earlier logger name.
ServerTranscriptLogger = ServerGradientTranscriptLogger

__all__ = [
    "ServerGradientTranscriptLogger",
    "ServerTranscriptLogger",
    "EvaluatorGroundTruthLogger",
]
