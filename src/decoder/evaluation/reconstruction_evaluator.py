from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..data.image_scaling import normalize_image
from .image_comparison_writer import ReconstructionComparisonWriter
from .reconstruction_metrics import per_sample_metrics


def evaluate_reconstructions(
    decoder,
    dataset,
    victim_model,
    output_dir: str | Path,
    device: torch.device,
    batch_size: int = 8,
    max_grid_images: int = 12,
    class_names: tuple[str, ...] | None = None,
) -> dict[str, float | int]:
    output = Path(output_dir)
    comparison_writer = ReconstructionComparisonWriter(
        output,
        class_names or (),
        max_grid_images=max_grid_images,
    )
    loader = DataLoader(dataset, batch_size=batch_size)
    decoder.eval()
    victim_model.eval()
    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for batch in loader:
            reconstruction = decoder(
                batch["smashed_z"].to(device),
                batch["server_output_u"].to(device),
                batch["grad_h_to_g"].to(device),
                batch["label_condition"].to(device),
            )
            target = batch["target_image"].to(device)
            metric = per_sample_metrics(reconstruction, target)
            reconstructed_labels = victim_model.predict(normalize_image(reconstruction)).argmax(dim=1)
            for index, sample_id in enumerate(batch["sample_id"]):
                true_label = int(batch["true_label"][index])
                predicted_label = int(batch["predicted_label"][index])
                row = {
                    "sample_id": sample_id,
                    "true_label": true_label,
                    "inferred_label": predicted_label,
                    "label_correct": int(true_label == predicted_label),
                    "reconstructed_label": int(reconstructed_labels[index]),
                    "mse": float(metric["mse"][index]),
                    "mae": float(metric["mae"][index]),
                    "psnr": float(metric["psnr"][index]),
                    "ssim": float(metric["ssim"][index]),
                }
                rows.append(row)
                comparison_writer.save(
                    sample_id,
                    target[index],
                    reconstruction[index],
                    true_label,
                    predicted_label,
                )
    if not rows:
        raise ValueError("evaluation dataset is empty")
    numeric = ("mse", "mae", "psnr", "ssim")
    summary: dict[str, float | int] = {"samples": len(rows)}
    summary.update({key: sum(float(row[key]) for row in rows) / len(rows) for key in numeric})
    summary["inferred_label_accuracy"] = sum(int(row["label_correct"]) for row in rows) / len(rows)
    summary["reconstruction_class_accuracy"] = sum(
        int(row["reconstructed_label"] == row["true_label"]) for row in rows
    ) / len(rows)
    correct = [row for row in rows if row["label_correct"]]
    incorrect = [row for row in rows if not row["label_correct"]]
    for name, group in (("correct_label", correct), ("wrong_label", incorrect)):
        summary[f"{name}_samples"] = len(group)
        if group:
            summary[f"{name}_psnr"] = sum(float(row["psnr"]) for row in group) / len(group)
            summary[f"{name}_ssim"] = sum(float(row["ssim"]) for row in group) / len(group)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "reconstruction_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        csv_writer.writeheader()
        csv_writer.writerows(rows)
    with (output / "reconstruction_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    comparison_writer.finalize()
    return summary


__all__ = ["evaluate_reconstructions"]
