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

from ...decoder.evaluation.reconstruction_metrics import per_sample_metrics
from ..models.factory import DecoderModel


def _tensor(batch: dict, key: str, device: torch.device) -> Tensor:
    value = batch[key]
    if not isinstance(value, Tensor):
        raise TypeError(f"batch[{key!r}] must be a tensor")
    return value.to(device)


class _ZComparisonWriter:
    def __init__(
        self, output_dir: Path, class_names: tuple[str, ...], max_grid_images: int
    ) -> None:
        self.output_dir = output_dir
        self.class_names = class_names
        self.max_grid_images = max_grid_images
        self.directories = {
            "original": output_dir / "originals",
            "u_grad_z": output_dir / "u_grad_z_reconstructions",
            "z_u_grad_z": output_dir / "z_u_grad_z_reconstructions",
            "comparison": output_dir / "comparisons",
        }
        for directory in self.directories.values():
            directory.mkdir(parents=True, exist_ok=True)
        self.panels: list[Image.Image] = []

    def _class_name(self, label: int) -> str:
        if 0 <= label < len(self.class_names):
            return self.class_names[label]
        return "unavailable" if label < 0 else str(label)

    def save(
        self,
        sample_id: str,
        original: Tensor,
        u_grad_z: Tensor,
        z_u_grad_z: Tensor,
        true_label: int,
        inferred_label: int,
    ) -> None:
        images = [
            to_pil_image(value.detach().cpu().clamp(0.0, 1.0))
            for value in (original, u_grad_z, z_u_grad_z)
        ]
        for key, image in zip(("original", "u_grad_z", "z_u_grad_z"), images):
            image.save(self.directories[key] / f"{sample_id}.png")
        width, height = images[0].size
        header_height = 34
        panel = Image.new("RGB", (3 * width, height + header_height), "white")
        for index, image in enumerate(images):
            panel.paste(image, (index * width, header_height))
        draw = ImageDraw.Draw(panel)
        headers = (
            ("Original", f"true={self._class_name(true_label)}", "black"),
            ("u + dL/dz", "downlink baseline", "green"),
            (
                "z + u + dL/dz",
                f"label={self._class_name(inferred_label)}",
                "blue",
            ),
        )
        for index, (title, subtitle, color) in enumerate(headers):
            x = index * width + 5
            draw.text((x, 2), title, fill=color)
            draw.text((x, 17), subtitle, fill=color)
        panel.save(self.directories["comparison"] / f"{sample_id}.png")
        if len(self.panels) < self.max_grid_images:
            self.panels.append(panel)

    def finalize(self) -> Path | None:
        if not self.panels:
            return None
        columns = min(2, len(self.panels))
        rows = math.ceil(len(self.panels) / columns)
        width = max(panel.width for panel in self.panels)
        height = max(panel.height for panel in self.panels)
        grid = Image.new("RGB", (columns * width, rows * height), "white")
        for index, panel in enumerate(self.panels):
            grid.paste(panel, ((index % columns) * width, (index // columns) * height))
        path = self.output_dir / "z_signal_comparison_grid.png"
        grid.save(path)
        return path


def evaluate_z_signal_comparison(
    u_grad_z_decoder: DecoderModel,
    z_u_grad_z_decoder: DecoderModel,
    dataset,
    output_dir: str | Path,
    device: torch.device,
    class_names: tuple[str, ...],
    batch_size: int = 8,
    max_grid_images: int = 20,
) -> dict[str, object]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    writer = _ZComparisonWriter(output, class_names, max_grid_images)
    loader = DataLoader(dataset, batch_size=batch_size)
    u_grad_z_decoder.eval()
    z_u_grad_z_decoder.eval()
    rows: list[dict[str, object]] = []

    with torch.no_grad():
        for batch in loader:
            z = _tensor(batch, "smashed_z", device)
            u = _tensor(batch, "server_output_u", device)
            grad_z = _tensor(batch, "grad_g_to_f", device)
            target = _tensor(batch, "target_image", device)
            true_labels = _tensor(batch, "true_label", device)
            baseline, _ = u_grad_z_decoder(u, grad_z)
            candidate, label_logits = z_u_grad_z_decoder(u, grad_z, z)
            if baseline.shape != target.shape or candidate.shape != target.shape:
                raise ValueError("all reconstructions and targets must have equal shapes")
            inferred_labels = (
                label_logits.argmax(dim=1)
                if label_logits is not None
                else torch.full_like(true_labels, -1)
            )
            outputs = {"u_grad_z": baseline, "z_u_grad_z": candidate}
            metrics = {
                name: per_sample_metrics(value, target)
                for name, value in outputs.items()
            }
            for index, transcript_id in enumerate(batch["transcript_id"]):
                row: dict[str, object] = {
                    "transcript_id": str(transcript_id),
                    "true_label": int(true_labels[index]),
                    "inferred_label": int(inferred_labels[index]),
                    "label_correct": int(
                        inferred_labels[index] == true_labels[index]
                    ),
                }
                for name in outputs:
                    for metric_name in ("mse", "mae", "psnr", "ssim"):
                        row[f"{name}_{metric_name}"] = float(
                            metrics[name][metric_name][index]
                        )
                rows.append(row)
                writer.save(
                    str(transcript_id),
                    target[index],
                    baseline[index],
                    candidate[index],
                    int(true_labels[index]),
                    int(inferred_labels[index]),
                )

    if not rows:
        raise ValueError("evaluation dataset is empty")
    summary: dict[str, object] = {"samples": len(rows)}
    for name in ("u_grad_z", "z_u_grad_z"):
        summary[name] = {
            metric_name: sum(
                float(row[f"{name}_{metric_name}"]) for row in rows
            )
            / len(rows)
            for metric_name in ("mse", "mae", "psnr", "ssim")
        }
    baseline_summary = summary["u_grad_z"]
    candidate_summary = summary["z_u_grad_z"]
    if not isinstance(baseline_summary, dict) or not isinstance(
        candidate_summary, dict
    ):
        raise TypeError("invalid metric summary")
    summary["delta_z_u_grad_z_minus_u_grad_z"] = {
        metric_name: float(candidate_summary[metric_name])
        - float(baseline_summary[metric_name])
        for metric_name in ("mse", "mae", "psnr", "ssim")
    }
    summary["z_psnr_improved_fraction"] = sum(
        float(row["z_u_grad_z_psnr"]) > float(row["u_grad_z_psnr"])
        for row in rows
    ) / len(rows)
    summary["z_ssim_improved_fraction"] = sum(
        float(row["z_u_grad_z_ssim"]) > float(row["u_grad_z_ssim"])
        for row in rows
    ) / len(rows)
    summary["inferred_label_accuracy"] = sum(
        int(row["label_correct"]) for row in rows
    ) / len(rows)

    with (output / "z_signal_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        csv_writer.writeheader()
        csv_writer.writerows(rows)
    with (output / "z_signal_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    writer.finalize()
    return summary


__all__ = ["evaluate_z_signal_comparison"]
