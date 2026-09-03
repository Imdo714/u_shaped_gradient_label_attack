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
from .postprocessing import conservative_visual_enhancement


def _tensor(batch: dict, key: str, device: torch.device) -> Tensor:
    value = batch[key]
    if not isinstance(value, Tensor):
        raise TypeError(f"batch[{key!r}] must be a tensor")
    return value.to(device)


class _FourWayWriter:
    def __init__(
        self,
        output_dir: Path,
        class_names: tuple[str, ...],
        max_grid_images: int,
    ) -> None:
        self.output_dir = output_dir
        self.class_names = class_names
        self.max_grid_images = max_grid_images
        self.directories = {
            "original": output_dir / "originals",
            "baseline": output_dir / "baseline_reconstructions",
            "v2": output_dir / "v2_reconstructions",
            "postprocessed": output_dir / "v2_postprocessed",
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
        baseline: Tensor,
        v2: Tensor,
        postprocessed: Tensor,
        true_label: int,
        inferred_label: int,
    ) -> None:
        tensors = (original, baseline, v2, postprocessed)
        images = [
            to_pil_image(value.detach().cpu().clamp(0.0, 1.0))
            for value in tensors
        ]
        for key, image in zip(
            ("original", "baseline", "v2", "postprocessed"), images
        ):
            image.save(self.directories[key] / f"{sample_id}.png")
        width, height = images[0].size
        header_height = 34
        panel = Image.new("RGB", (4 * width, height + header_height), "white")
        for index, image in enumerate(images):
            panel.paste(image, (index * width, header_height))
        draw = ImageDraw.Draw(panel)
        headers = (
            ("Original", f"true={self._class_name(true_label)}", "black"),
            ("Decoder v1", "bilinear", "green"),
            ("Decoder v2", f"label={self._class_name(inferred_label)}", "blue"),
            ("V2 + visual", "fixed filter", "purple"),
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
        path = self.output_dir / "decoder_v2_comparison_grid.png"
        grid.save(path)
        return path


def evaluate_decoder_v2_comparison(
    baseline_decoder: DecoderModel,
    v2_decoder: DecoderModel,
    dataset,
    output_dir: str | Path,
    device: torch.device,
    class_names: tuple[str, ...],
    batch_size: int = 8,
    max_grid_images: int = 20,
    chroma_denoise: float = 0.15,
    sharpen_amount: float = 0.2,
    contrast: float = 1.03,
    saturation: float = 1.03,
) -> dict[str, object]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    writer = _FourWayWriter(output, class_names, max_grid_images)
    loader = DataLoader(dataset, batch_size=batch_size)
    baseline_decoder.eval()
    v2_decoder.eval()
    rows: list[dict[str, object]] = []

    with torch.no_grad():
        for batch in loader:
            u = _tensor(batch, "server_output_u", device)
            grad_z = _tensor(batch, "grad_g_to_f", device)
            target = _tensor(batch, "target_image", device)
            true_labels = _tensor(batch, "true_label", device)
            baseline, _ = baseline_decoder(u, grad_z)
            v2, label_logits = v2_decoder(u, grad_z)
            postprocessed = conservative_visual_enhancement(
                v2,
                chroma_denoise=chroma_denoise,
                sharpen_amount=sharpen_amount,
                contrast=contrast,
                saturation=saturation,
            )
            if baseline.shape != target.shape or v2.shape != target.shape:
                raise ValueError(
                    "baseline, v2, and target must have identical BCHW shapes"
                )
            inferred_labels = (
                label_logits.argmax(dim=1)
                if label_logits is not None
                else torch.full_like(true_labels, -1)
            )
            outputs = {
                "baseline": baseline,
                "v2": v2,
                "postprocessed": postprocessed,
            }
            metrics = {
                name: per_sample_metrics(value, target)
                for name, value in outputs.items()
            }
            for index, transcript_id in enumerate(batch["transcript_id"]):
                row: dict[str, object] = {
                    "transcript_id": str(transcript_id),
                    "true_label": int(true_labels[index]),
                    "inferred_label": int(inferred_labels[index]),
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
                    v2[index],
                    postprocessed[index],
                    int(true_labels[index]),
                    int(inferred_labels[index]),
                )

    if not rows:
        raise ValueError("evaluation dataset is empty")
    summary: dict[str, object] = {"samples": len(rows)}
    for name in ("baseline", "v2", "postprocessed"):
        summary[name] = {
            metric_name: sum(
                float(row[f"{name}_{metric_name}"]) for row in rows
            )
            / len(rows)
            for metric_name in ("mse", "mae", "psnr", "ssim")
        }
    summary["delta_v2_minus_baseline"] = {
        metric_name: summary["v2"][metric_name]
        - summary["baseline"][metric_name]
        for metric_name in ("mse", "mae", "psnr", "ssim")
    }
    summary["delta_postprocessed_minus_v2"] = {
        metric_name: summary["postprocessed"][metric_name]
        - summary["v2"][metric_name]
        for metric_name in ("mse", "mae", "psnr", "ssim")
    }
    summary["v2_psnr_improved_fraction"] = sum(
        float(row["v2_psnr"]) > float(row["baseline_psnr"]) for row in rows
    ) / len(rows)
    summary["v2_ssim_improved_fraction"] = sum(
        float(row["v2_ssim"]) > float(row["baseline_ssim"]) for row in rows
    ) / len(rows)

    with (output / "decoder_v2_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        csv_writer.writeheader()
        csv_writer.writerows(rows)
    with (output / "decoder_v2_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    writer.finalize()
    return summary


__all__ = ["evaluate_decoder_v2_comparison"]
