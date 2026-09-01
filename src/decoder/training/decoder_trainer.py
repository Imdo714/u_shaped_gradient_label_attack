from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..losses.reconstruction_loss import ReconstructionLoss
from ..models.label_conditioned_decoder import LabelConditionedDecoder


@dataclass(frozen=True)
class DecoderTrainingConfig:
    epochs: int = 20
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    l1_weight: float = 1.0
    ssim_weight: float = 0.5
    edge_weight: float = 0.0
    perceptual_weight: float = 0.0


def _forward(model: LabelConditionedDecoder, batch: dict, device: torch.device):
    return model(
        batch["smashed_z"].to(device),
        batch["server_output_u"].to(device),
        batch["grad_h_to_g"].to(device),
        batch["grad_g_to_f"].to(device),
        batch["label_condition"].to(device),
    )


def _epoch(
    model: LabelConditionedDecoder,
    loader: DataLoader,
    loss_function: ReconstructionLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    samples = 0
    for batch in loader:
        target = batch["target_image"].to(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            reconstruction = _forward(model, batch, device)
            loss, metrics = loss_function(reconstruction, target)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
        batch_size = target.shape[0]
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value * batch_size
        samples += batch_size
    if samples == 0:
        raise ValueError("observation dataset is empty")
    return {key: value / samples for key, value in totals.items()}


def train_decoder(
    model: LabelConditionedDecoder,
    train_dataset,
    validation_dataset,
    output_dir: str | Path,
    config: DecoderTrainingConfig,
    device: torch.device,
) -> tuple[Path, list[dict[str, float | int]]]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=config.batch_size)
    loss_function = ReconstructionLoss(
        l1_weight=config.l1_weight,
        ssim_weight=config.ssim_weight,
        edge_weight=config.edge_weight,
        perceptual_weight=config.perceptual_weight,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    history: list[dict[str, float | int]] = []
    for epoch in range(1, config.epochs + 1):
        train_metrics = _epoch(model, train_loader, loss_function, device, optimizer)
        validation_metrics = _epoch(model, validation_loader, loss_function, device, None)
        row: dict[str, float | int] = {"epoch": epoch}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"validation_{key}": value for key, value in validation_metrics.items()})
        history.append(row)
        print(
            f"Decoder epoch {epoch:03d}/{config.epochs:03d}: "
            f"train_loss={train_metrics['loss']:.4f}, "
            f"validation_loss={validation_metrics['loss']:.4f}"
        )
        if validation_metrics["loss"] < best_loss:
            best_loss = validation_metrics["loss"]
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    checkpoint = output / "decoder_best.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "decoder_config": model.config.to_dict(),
            "training_config": asdict(config),
            "best_validation_loss": best_loss,
        },
        checkpoint,
    )
    with (output / "training_history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    return checkpoint, history


__all__ = ["DecoderTrainingConfig", "train_decoder"]
