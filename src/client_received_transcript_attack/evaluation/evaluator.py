from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from ...decoder.evaluation.image_comparison_writer import ReconstructionComparisonWriter
from ...decoder.evaluation.reconstruction_metrics import per_sample_metrics
from ..models.decoder import ClientReceivedDecoder


def _tensor(batch: dict, key: str, device: torch.device) -> Tensor:
    value = batch[key]
    if not isinstance(value, Tensor):
        raise TypeError(f"batch[{key!r}] must be a tensor")
    return value.to(device)


def evaluate_client_received_decoder(
    model: ClientReceivedDecoder,
    dataset,
    output_dir: str | Path,
    device: torch.device,
    class_names: tuple[str, ...],
    batch_size: int = 8,
    max_grid_images: int = 20,
    save_separate_images: bool = True,
) -> dict[str, float | int | bool]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    writer = ReconstructionComparisonWriter(
        output,
        class_names,
        max_grid_images=max_grid_images,
        save_separate_images=save_separate_images,
    )
    loader = DataLoader(dataset, batch_size=batch_size)
    model.eval()
    rows: list[dict[str, object]] = []
    label_head_enabled = False

    with torch.no_grad():
        for batch in loader:
            reconstruction, label_logits = model(
                _tensor(batch, "server_output_u", device),
                _tensor(batch, "grad_g_to_f", device),
            )
            target = _tensor(batch, "target_image", device)
            true_labels = _tensor(batch, "true_label", device)
            metrics = per_sample_metrics(reconstruction, target)
            if label_logits is None:
                inferred_labels = torch.full_like(true_labels, -1)
            else:
                label_head_enabled = True
                inferred_labels = label_logits.argmax(dim=1)

            for index, transcript_id in enumerate(batch["transcript_id"]):
                true_label = int(true_labels[index])
                inferred_label = int(inferred_labels[index])
                row = {
                    "transcript_id": transcript_id,
                    "true_label": true_label,
                    "inferred_label": inferred_label,
                    "label_correct": int(inferred_label == true_label),
                    "mse": float(metrics["mse"][index]),
                    "mae": float(metrics["mae"][index]),
                    "psnr": float(metrics["psnr"][index]),
                    "ssim": float(metrics["ssim"][index]),
                }
                rows.append(row)
                writer.save(
                    transcript_id,
                    target[index],
                    reconstruction[index],
                    true_label,
                    inferred_label,
                )

    if not rows:
        raise ValueError("evaluation dataset is empty")
    summary: dict[str, float | int | bool] = {
        "samples": len(rows),
        "label_head_enabled": label_head_enabled,
    }
    for key in ("mse", "mae", "psnr", "ssim"):
        summary[key] = sum(float(row[key]) for row in rows) / len(rows)
    if label_head_enabled:
        summary["inferred_label_accuracy"] = sum(
            int(row["label_correct"]) for row in rows
        ) / len(rows)

    with (output / "reconstruction_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        csv_writer.writeheader()
        csv_writer.writerows(rows)
    with (output / "reconstruction_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    writer.finalize()
    return summary


__all__ = ["evaluate_client_received_decoder"]
