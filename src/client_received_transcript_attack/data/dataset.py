from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


ATTACKER_KEYS = {"server_output_u", "grad_g_to_f"}
EVALUATOR_KEYS = {"target_image", "true_label"}


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _safe_record_path(root: Path, relative_path: str) -> Path:
    resolved_root = root.resolve()
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"record escapes manifest directory: {relative_path}") from error
    return candidate


class ClientReceivedTranscriptDataset(Dataset):
    """Join attacker-visible u/dL-dz with separately stored public/evaluator targets."""

    def __init__(
        self,
        attacker_manifest: str | Path,
        evaluator_manifest: str | Path | None = None,
    ) -> None:
        self.attacker_manifest = Path(attacker_manifest)
        attacker_rows = _read_rows(self.attacker_manifest)
        if not attacker_rows:
            raise ValueError(f"empty attacker manifest: {self.attacker_manifest}")
        evaluator_by_id: dict[str, str] = {}
        self.evaluator_manifest = Path(evaluator_manifest) if evaluator_manifest else None
        if self.evaluator_manifest is not None:
            evaluator_rows = _read_rows(self.evaluator_manifest)
            evaluator_by_id = {
                row["transcript_id"]: row["evaluator_target"] for row in evaluator_rows
            }
        self.rows: list[dict[str, str]] = []
        for row in attacker_rows:
            transcript_id = row["transcript_id"]
            merged = dict(row)
            if self.evaluator_manifest is not None:
                if transcript_id not in evaluator_by_id:
                    raise ValueError(f"missing evaluator target for {transcript_id}")
                merged["evaluator_target"] = evaluator_by_id[transcript_id]
            self.rows.append(merged)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.rows[index]
        attacker_path = _safe_record_path(
            self.attacker_manifest.parent, row["attacker_record"]
        )
        with np.load(attacker_path, allow_pickle=False) as record:
            keys = set(record.files)
            if keys != ATTACKER_KEYS:
                raise ValueError(
                    f"attacker record {attacker_path} has keys {sorted(keys)}; "
                    f"expected only {sorted(ATTACKER_KEYS)}"
                )
            item: dict[str, torch.Tensor | str] = {
                "transcript_id": row["transcript_id"],
                "server_output_u": torch.from_numpy(record["server_output_u"]).float(),
                "grad_g_to_f": torch.from_numpy(record["grad_g_to_f"]).float(),
            }
        if self.evaluator_manifest is not None:
            target_path = _safe_record_path(
                self.evaluator_manifest.parent, row["evaluator_target"]
            )
            with np.load(target_path, allow_pickle=False) as target:
                keys = set(target.files)
                if keys != EVALUATOR_KEYS:
                    raise ValueError(
                        f"evaluator record {target_path} has keys {sorted(keys)}; "
                        f"expected only {sorted(EVALUATOR_KEYS)}"
                    )
                item["target_image"] = torch.from_numpy(target["target_image"]).float()
                item["true_label"] = torch.tensor(int(target["true_label"]), dtype=torch.long)
        return item


__all__ = [
    "ATTACKER_KEYS",
    "EVALUATOR_KEYS",
    "ClientReceivedTranscriptDataset",
]
