from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


def _tensor(record: np.lib.npyio.NpzFile, key: str) -> Tensor:
    return torch.from_numpy(record[key]).float()


class TranscriptDataset(Dataset):
    """Read only the signals naturally visible to the honest-but-curious server.

    Evaluator targets are opened only when ``include_target=True``.  Strict
    unpaired training must construct this dataset with the default ``False``.
    Older records without dL/dz are supported and expose an availability flag.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        include_target: bool = False,
        require_grad_z: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        self.include_target = include_target
        self.require_grad_z = require_grad_z
        with self.manifest_path.open("r", newline="", encoding="utf-8") as handle:
            self.rows = list(csv.DictReader(handle))
        if not self.rows:
            raise ValueError(f"empty transcript manifest: {self.manifest_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        row = self.rows[index]
        with np.load(self.root / row["attacker_record"]) as record:
            z = _tensor(record, "smashed_z")
            has_grad_z = "grad_g_to_f" in record.files
            if self.require_grad_z and not has_grad_z:
                raise KeyError(
                    "record has no grad_g_to_f; recollect it with ServerTranscriptWriter "
                    "or disable the dL/dz signal"
                )
            grad_z = _tensor(record, "grad_g_to_f") if has_grad_z else torch.zeros_like(z)
            item: dict[str, Tensor | str] = {
                "sample_id": row["sample_id"],
                "smashed_z": z,
                "server_output_u": _tensor(record, "server_output_u"),
                "grad_h_to_g": _tensor(record, "grad_h_to_g"),
                "grad_g_to_f": grad_z,
                "has_grad_g_to_f": torch.tensor(has_grad_z),
                "label_condition": _tensor(record, "label_condition"),
                "predicted_label": torch.tensor(int(record["predicted_label"])),
                "confidence": torch.tensor(float(record["confidence"])),
            }
        if self.include_target:
            target_path = row.get("evaluator_target")
            if not target_path:
                raise KeyError("manifest has no evaluator_target column")
            with np.load(self.root / target_path) as target:
                item["target_image"] = _tensor(target, "target_image")
                item["true_label"] = torch.tensor(int(target["true_label"]))
        return item


class PublicTargetDataset(Dataset):
    """Read images and labels from an auxiliary/public manifest only.

    This view deliberately never opens ``attacker_record``.  It is used to
    train the architecture-agnostic simulators without accidentally consuming
    a victim transcript paired with the public image.
    """

    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        with self.manifest_path.open("r", newline="", encoding="utf-8") as handle:
            self.rows = list(csv.DictReader(handle))
        if not self.rows:
            raise ValueError(f"empty public manifest: {self.manifest_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        row = self.rows[index]
        target_path = row.get("evaluator_target")
        if not target_path:
            raise KeyError("public manifest has no evaluator_target column")
        with np.load(self.root / target_path) as target:
            return {
                "sample_id": row["sample_id"],
                "image": _tensor(target, "target_image"),
                "label": torch.tensor(int(target["true_label"])),
            }


__all__ = ["PublicTargetDataset", "TranscriptDataset"]
