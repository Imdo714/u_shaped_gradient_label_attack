from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image

from ...client_received_transcript_attack.models.factory import DecoderModel
from ...decoder.evaluation.reconstruction_metrics import per_sample_metrics
from ..models.refiner import (
    ResidualUNetRefiner,
    TranscriptConditionedResidualUNetRefiner,
)


def _tensor(batch: dict, key: str, device: torch.device) -> Tensor:
    value = batch[key]
    if not isinstance(value, Tensor):
        raise TypeError(f"batch[{key!r}] must be a tensor")
    return value.to(device)


class _RefinementComparisonWriter:
    """Write Original/Coarse/Refined panels from individual tensors."""

    def __init__(
        self,
        output_dir: str | Path,
        class_names: tuple[str, ...],
        max_grid_images: int,
        grid_columns: int = 3,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.class_names = class_names
        self.max_grid_images = max_grid_images
        self.grid_columns = grid_columns
        self.originals_dir = self.output_dir / "originals"
        self.coarse_dir = self.output_dir / "coarse_reconstructions"
        self.refined_dir = self.output_dir / "refined_reconstructions"
        self.comparison_dir = self.output_dir / "comparisons"
        for directory in (
            self.originals_dir,
            self.coarse_dir,
            self.refined_dir,
            self.comparison_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self._panels: list[Image.Image] = []

    def _class_name(self, label: int) -> str:
        if 0 <= label < len(self.class_names):
            return self.class_names[label]
        return "unavailable" if label < 0 else str(label)

    def save(
        self,
        sample_id: str,
        original: Tensor,
        coarse: Tensor,
        refined: Tensor,
        true_label: int,
        inferred_label: int,
    ) -> None:
        images = [
            to_pil_image(value.detach().cpu().clamp(0.0, 1.0))
            for value in (original, coarse, refined)
        ]
        images[0].save(self.originals_dir / f"{sample_id}.png")
        images[1].save(self.coarse_dir / f"{sample_id}.png")
        images[2].save(self.refined_dir / f"{sample_id}.png")

        width, height = images[0].size
        header_height = 34
        panel = Image.new("RGB", (width * 3, height + header_height), "white")
        for index, image in enumerate(images):
            panel.paste(image, (index * width, header_height))
        draw = ImageDraw.Draw(panel)
        draw.text((5, 2), "Original", fill="black")
        draw.text((5, 17), f"true={self._class_name(true_label)}", fill="black")
        label_color = "green" if true_label == inferred_label else "red"
        draw.text((width + 5, 2), "Coarse", fill=label_color)
        draw.text(
            (width + 5, 17),
            f"inferred={self._class_name(inferred_label)}",
            fill=label_color,
        )
        draw.text((2 * width + 5, 2), "Refined", fill="blue")
        draw.text((2 * width + 5, 17), "residual", fill="blue")
        panel.save(self.comparison_dir / f"{sample_id}.png")
        if len(self._panels) < self.max_grid_images:
            self._panels.append(panel)

    def finalize(self) -> Path | None:
        if not self._panels:
            return None
        columns = min(self.grid_columns, len(self._panels))
        rows = math.ceil(len(self._panels) / columns)
        panel_width = max(panel.width for panel in self._panels)
        panel_height = max(panel.height for panel in self._panels)
        grid = Image.new("RGB", (columns * panel_width, rows * panel_height), "white")
        for index, panel in enumerate(self._panels):
            grid.paste(
                panel,
                ((index % columns) * panel_width, (index // columns) * panel_height),
            )
        path = self.output_dir / "refinement_comparison_grid.png"
        grid.save(path)
        return path


def evaluate_refiner(
    decoder: DecoderModel,
    refiner: ResidualUNetRefiner | TranscriptConditionedResidualUNetRefiner,
    dataset,
    output_dir: str | Path,
    device: torch.device,
    class_names: tuple[str, ...],
    batch_size: int = 8,
    max_grid_images: int = 20,
) -> dict[str, object]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    writer = _RefinementComparisonWriter(output, class_names, max_grid_images)
    loader = DataLoader(dataset, batch_size=batch_size)
    decoder.eval()
    refiner.eval()
    rows: list[dict[str, object]] = []
    label_head_enabled = False

    with torch.no_grad():
        for batch in loader:
            server_output_u = _tensor(batch, "server_output_u", device)
            grad_g_to_f = _tensor(batch, "grad_g_to_f", device)
            coarse, label_logits = decoder(
                server_output_u,
                grad_g_to_f,
            )
            if isinstance(refiner, TranscriptConditionedResidualUNetRefiner):
                refined, residual = refiner(
                    coarse, server_output_u, grad_g_to_f
                )
            else:
                refined, residual = refiner(coarse)
            target = _tensor(batch, "target_image", device)
            true_labels = _tensor(batch, "true_label", device)
            coarse_metrics = per_sample_metrics(coarse, target)
            refined_metrics = per_sample_metrics(refined, target)
            if label_logits is None:
                inferred_labels = torch.full_like(true_labels, -1)
            else:
                label_head_enabled = True
                inferred_labels = label_logits.argmax(dim=1)

            residual_l1 = residual.abs().flatten(1).mean(dim=1)
            for index, transcript_id in enumerate(batch["transcript_id"]):
                true_label = int(true_labels[index])
                inferred_label = int(inferred_labels[index])
                row: dict[str, object] = {
                    "transcript_id": transcript_id,
                    "true_label": true_label,
                    "inferred_label": inferred_label,
                    "label_correct": int(inferred_label == true_label),
                    "residual_l1": float(residual_l1[index]),
                }
                for key in ("mse", "mae", "psnr", "ssim"):
                    coarse_value = float(coarse_metrics[key][index])
                    refined_value = float(refined_metrics[key][index])
                    row[f"coarse_{key}"] = coarse_value
                    row[f"refined_{key}"] = refined_value
                    row[f"delta_{key}"] = refined_value - coarse_value
                rows.append(row)
                writer.save(
                    str(transcript_id),
                    target[index],
                    coarse[index],
                    refined[index],
                    true_label,
                    inferred_label,
                )

    if not rows:
        raise ValueError("evaluation dataset is empty")
    metric_names = ("mse", "mae", "psnr", "ssim")
    coarse_summary = {
        key: sum(float(row[f"coarse_{key}"]) for row in rows) / len(rows)
        for key in metric_names
    }
    refined_summary = {
        key: sum(float(row[f"refined_{key}"]) for row in rows) / len(rows)
        for key in metric_names
    }
    delta_summary = {
        key: refined_summary[key] - coarse_summary[key] for key in metric_names
    }
    summary: dict[str, object] = {
        "samples": len(rows),
        "refiner_type": refiner.refiner_type,
        "label_head_enabled": label_head_enabled,
        "coarse": coarse_summary,
        "refined": refined_summary,
        "delta_refined_minus_coarse": delta_summary,
        "psnr_improved_fraction": sum(float(row["delta_psnr"]) > 0 for row in rows)
        / len(rows),
        "ssim_improved_fraction": sum(float(row["delta_ssim"]) > 0 for row in rows)
        / len(rows),
        "mean_residual_l1": sum(float(row["residual_l1"]) for row in rows)
        / len(rows),
    }
    if label_head_enabled:
        summary["inferred_label_accuracy"] = sum(
            int(row["label_correct"]) for row in rows
        ) / len(rows)

    with (output / "refinement_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        csv_writer.writeheader()
        csv_writer.writerows(rows)
    with (output / "refinement_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    writer.finalize()
    return summary


__all__ = ["evaluate_refiner"]
