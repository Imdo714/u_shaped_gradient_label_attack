from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .workspace_paths import DEFAULT_WORKSPACE_PATHS


@dataclass
class ExperimentConfig:
    data_dir: str = str(DEFAULT_WORKSPACE_PATHS.dataset)
    output_dir: str = str(DEFAULT_WORKSPACE_PATHS.results_root)
    image_size: int = 64
    batch_size: int = 1
    epochs: int = 1
    learning_rate: float = 1e-3
    random_seed: int = 42
    num_workers: int = 0
    cut_config: str = "middle"
    device: str = "auto"
    max_attack_samples: int | None = None
    debug_samples: int = 3
    debug_values: int = 8

    def resolved_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def output_path(self, *parts: str) -> Path:
        return Path(self.output_dir).joinpath(*parts)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
