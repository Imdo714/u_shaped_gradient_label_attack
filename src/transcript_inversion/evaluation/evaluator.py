from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import torch

from ..losses.reconstruction import structural_similarity
from ..training.paired_decoder_trainer import decoder_forward


class ReconstructionEvaluator:
    """Use protected originals only after training, for paper metrics."""

    def __init__(self, device: torch.device) -> None:
        self.device = device

    def evaluate(self, decoder, loader, output_dir: str | Path) -> dict[str, float | int]:
        decoder.eval()
        rows: list[dict[str, object]] = []
        with torch.no_grad():
            for batch in loader:
                target = batch["target_image"].to(self.device)
                reconstruction = decoder_forward(decoder, batch, self.device)
                difference = reconstruction - target
                mse = difference.square().flatten(1).mean(dim=1)
                mae = difference.abs().flatten(1).mean(dim=1)
                for index, sample_id in enumerate(batch["sample_id"]):
                    sample_ssim = structural_similarity(
                        reconstruction[index : index + 1], target[index : index + 1]
                    )
                    sample_mse = float(mse[index])
                    rows.append(
                        {
                            "sample_id": sample_id,
                            "true_label": int(batch["true_label"][index]),
                            "inferred_label": int(batch["predicted_label"][index]),
                            "mse": sample_mse,
                            "mae": float(mae[index]),
                            "psnr": 10.0 * math.log10(1.0 / max(sample_mse, 1e-12)),
                            "ssim": float(sample_ssim),
                        }
                    )
        if not rows:
            raise ValueError("evaluation loader is empty")
        summary: dict[str, float | int] = {"samples": len(rows)}
        for metric in ("mse", "mae", "psnr", "ssim"):
            values = [float(row[metric]) for row in rows]
            summary[metric] = sum(values) / len(values)
            summary[f"{metric}_std"] = (
                sum((value - float(summary[metric])) ** 2 for value in values) / len(values)
            ) ** 0.5
        summary["inferred_label_accuracy"] = sum(
            int(row["true_label"] == row["inferred_label"]) for row in rows
        ) / len(rows)
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with (output / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        return summary


__all__ = ["ReconstructionEvaluator"]
