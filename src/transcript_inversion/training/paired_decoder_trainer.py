from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from ..losses.reconstruction import ReconstructionLoss


@dataclass(frozen=True)
class PairedDecoderConfig:
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    l1_weight: float = 1.0
    ssim_weight: float = 0.5
    edge_weight: float = 0.1


def decoder_forward(model: nn.Module, batch: dict, device: torch.device):
    return model(
        batch["smashed_z"].to(device),
        batch["server_output_u"].to(device),
        batch["grad_h_to_g"].to(device),
        batch["grad_g_to_f"].to(device),
        batch["label_condition"].to(device),
    )


class PairedDecoderTrainer:
    """Train P0-P3 supervised controls; pair construction stays in the dataset."""

    def __init__(self, config: PairedDecoderConfig, device: torch.device) -> None:
        self.config = config
        self.device = device
        self.loss_function = ReconstructionLoss(
            config.l1_weight, config.ssim_weight, config.edge_weight
        )

    def _epoch(self, model: nn.Module, loader, optimizer=None) -> dict[str, float]:
        training = optimizer is not None
        model.train(training)
        total_loss = 0.0
        total_ssim = 0.0
        samples = 0
        for batch in loader:
            target = batch["target_image"].to(self.device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training):
                reconstruction = decoder_forward(model, batch, self.device)
                loss, metrics = self.loss_function(reconstruction, target)
                if optimizer is not None:
                    loss.backward()
                    optimizer.step()
            batch_size = target.shape[0]
            total_loss += float(loss.detach()) * batch_size
            total_ssim += metrics["ssim"] * batch_size
            samples += batch_size
        if samples == 0:
            raise ValueError("paired loader is empty")
        return {"loss": total_loss / samples, "ssim": total_ssim / samples}

    def fit(self, model: nn.Module, train_loader, validation_loader, output_dir: str | Path):
        model.to(self.device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        best_loss = float("inf")
        best_state = copy.deepcopy(model.state_dict())
        history: list[dict[str, float | int]] = []
        for epoch in range(1, self.config.epochs + 1):
            train_metrics = self._epoch(model, train_loader, optimizer)
            validation_metrics = self._epoch(model, validation_loader)
            row: dict[str, float | int] = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_ssim": train_metrics["ssim"],
                "validation_loss": validation_metrics["loss"],
                "validation_ssim": validation_metrics["ssim"],
            }
            history.append(row)
            if validation_metrics["loss"] < best_loss:
                best_loss = validation_metrics["loss"]
                best_state = copy.deepcopy(model.state_dict())
        model.load_state_dict(best_state)
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "decoder_config": model.config.to_dict(),
                "training_config": asdict(self.config),
                "best_validation_loss": best_loss,
            },
            output / "decoder_best.pt",
        )
        with (output / "paired_history.json").open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)
        return history


__all__ = ["PairedDecoderConfig", "PairedDecoderTrainer", "decoder_forward"]
