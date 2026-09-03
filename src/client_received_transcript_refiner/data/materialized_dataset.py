from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ...client_received_transcript_attack.data.dataset import (
    ClientReceivedTranscriptDataset,
)


def _safe_path(root: Path, relative_path: str) -> Path:
    resolved_root = root.resolve()
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"coarse record escapes manifest directory: {relative_path}") from error
    return candidate


class MaterializedCoarseTranscriptDataset(Dataset):
    """Join raw u/dL-dz, public targets, and leakage-safe coarse predictions."""

    def __init__(
        self,
        attacker_manifest: str | Path,
        evaluator_manifest: str | Path,
        coarse_manifest: str | Path,
    ) -> None:
        self.base = ClientReceivedTranscriptDataset(
            attacker_manifest, evaluator_manifest
        )
        self.coarse_manifest = Path(coarse_manifest)
        with self.coarse_manifest.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"empty coarse manifest: {self.coarse_manifest}")
        coarse_by_id = {row["transcript_id"]: row for row in rows}
        base_ids = {str(row["transcript_id"]) for row in self.base.rows}
        if set(coarse_by_id) != base_ids:
            missing = sorted(base_ids - set(coarse_by_id))[:3]
            extra = sorted(set(coarse_by_id) - base_ids)[:3]
            raise ValueError(
                "coarse/base transcript IDs differ; "
                f"missing={missing}, extra={extra}"
            )
        self.rows = [
            {**row, **coarse_by_id[str(row["transcript_id"])]}
            for row in self.base.rows
        ]

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict:
        item = dict(self.base[index])
        row = self.rows[index]
        coarse_path = _safe_path(
            self.coarse_manifest.parent, row["coarse_record"]
        )
        with np.load(coarse_path, allow_pickle=False) as record:
            if set(record.files) != {"coarse_reconstruction"}:
                raise ValueError(
                    f"unexpected keys in coarse record {coarse_path}: {record.files}"
                )
            item["coarse_reconstruction"] = torch.from_numpy(
                record["coarse_reconstruction"]
            ).float()
        item["coarse_fold"] = row.get("fold", "reference")
        return item


__all__ = ["MaterializedCoarseTranscriptDataset"]
