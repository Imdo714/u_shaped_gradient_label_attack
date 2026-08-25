from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw
from torch import Tensor
from torchvision.transforms.functional import to_pil_image


class ReconstructionComparisonWriter:
    """Save original/reconstruction pairs without mixing this concern into metrics."""

    def __init__(
        self,
        output_dir: str | Path,
        class_names: tuple[str, ...],
        max_grid_images: int = 12,
        grid_columns: int = 3,
        save_separate_images: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.comparison_dir = self.output_dir / "comparisons"
        self.comparison_dir.mkdir(parents=True, exist_ok=True)
        self.class_names = class_names
        self.max_grid_images = max_grid_images
        self.grid_columns = grid_columns
        self.save_separate_images = save_separate_images
        self._panels: list[Image.Image] = []
        if save_separate_images:
            self.originals_dir = self.output_dir / "originals"
            self.reconstructions_dir = self.output_dir / "reconstructions"
            self.originals_dir.mkdir(parents=True, exist_ok=True)
            self.reconstructions_dir.mkdir(parents=True, exist_ok=True)

    def _class_name(self, label: int) -> str:
        if 0 <= label < len(self.class_names):
            return self.class_names[label]
        return str(label)

    def save(
        self,
        sample_id: str,
        original: Tensor,
        reconstruction: Tensor,
        true_label: int,
        inferred_label: int,
    ) -> None:
        original_image = to_pil_image(original.detach().cpu().clamp(0.0, 1.0))
        reconstructed_image = to_pil_image(
            reconstruction.detach().cpu().clamp(0.0, 1.0)
        )
        if self.save_separate_images:
            original_image.save(self.originals_dir / f"{sample_id}.png")
            reconstructed_image.save(self.reconstructions_dir / f"{sample_id}.png")

        width, height = original_image.size
        header_height = 34
        panel = Image.new("RGB", (width * 2, height + header_height), "white")
        panel.paste(original_image, (0, header_height))
        panel.paste(reconstructed_image, (width, header_height))
        draw = ImageDraw.Draw(panel)
        draw.text((5, 2), "Original", fill="black")
        draw.text((5, 17), f"true={self._class_name(true_label)}", fill="black")
        condition_text = self._class_name(inferred_label)
        condition_color = "green" if true_label == inferred_label else "red"
        draw.text((width + 5, 2), "Restored", fill=condition_color)
        draw.text((width + 5, 17), f"inferred={condition_text}", fill=condition_color)
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
            x = (index % columns) * panel_width
            y = (index // columns) * panel_height
            grid.paste(panel, (x, y))
        path = self.output_dir / "comparison_grid.png"
        grid.save(path)
        return path


__all__ = ["ReconstructionComparisonWriter"]
