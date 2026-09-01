from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from ...decoder.losses.reconstruction_loss import ReconstructionLoss
from ..models.decoder import ClientReceivedDecoder


@dataclass(frozen=True)
class AttackTrainingConfig:
    epochs: int = 100
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    l1_weight: float = 1.0
    ssim_weight: float = 0.75
    edge_weight: float = 0.15
    perceptual_weight: float = 0.25
    classification_weight: float = 0.1
    gradient_clip_norm: float = 5.0
    num_workers: int = 0


def _to_device(batch: dict, key: str, device: torch.device) -> Tensor:
    value = batch[key]
    if not isinstance(value, Tensor):
        raise TypeError(f"batch[{key!r}] must be a tensor")
    return value.to(device)


def _run_epoch(
    model: ClientReceivedDecoder,
    loader: DataLoader,
    reconstruction_loss: ReconstructionLoss,
    classification_loss: nn.Module,
    classification_weight: float,
    gradient_clip_norm: float,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {
        "loss": 0.0,
        "reconstruction_loss": 0.0,
        "classification_loss": 0.0,
        "l1": 0.0,
        "ssim": 0.0,
        "edge": 0.0,
        "perceptual": 0.0,
        "label_correct": 0.0,
        "label_samples": 0.0,
    }
    samples = 0
    for batch in loader:
        target = _to_device(batch, "target_image", device)
        true_label = _to_device(batch, "true_label", device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            reconstruction, label_logits = model(
                _to_device(batch, "server_output_u", device),
                _to_device(batch, "grad_g_to_f", device),
            )
            reconstruction_total, reconstruction_metrics = reconstruction_loss(
                reconstruction, target
            )
            label_total = reconstruction_total.new_zeros(())
            if label_logits is not None:
                label_total = classification_loss(label_logits, true_label)
            total = reconstruction_total + classification_weight * label_total
            if optimizer is not None:
                total.backward()
                if gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()

        batch_size = int(target.shape[0])
        samples += batch_size
        totals["loss"] += float(total.detach()) * batch_size
        totals["reconstruction_loss"] += float(reconstruction_total.detach()) * batch_size
        totals["classification_loss"] += float(label_total.detach()) * batch_size
        for key in ("l1", "ssim", "edge", "perceptual"):
            totals[key] += float(reconstruction_metrics[key]) * batch_size
        if label_logits is not None:
            totals["label_correct"] += float(
                (label_logits.detach().argmax(dim=1) == true_label).sum()
            )
            totals["label_samples"] += batch_size

    if samples == 0:
        raise ValueError("training or validation dataset is empty")
    result = {
        key: value / samples
        for key, value in totals.items()
        if key not in ("label_correct", "label_samples")
    }
    if totals["label_samples"]:
        result["label_accuracy"] = totals["label_correct"] / totals["label_samples"]
    return result


def train_client_received_decoder(
    model: ClientReceivedDecoder,
    train_dataset,
    validation_dataset,
    output_dir: str | Path,
    config: AttackTrainingConfig,
    device: torch.device,
) -> tuple[Path, list[dict[str, float | int]]]:
    """Train only the attacker model; the transcript provider is never optimized here."""

    if config.epochs < 1 or config.batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )
    reconstruction_loss = ReconstructionLoss(
        l1_weight=config.l1_weight,
        ssim_weight=config.ssim_weight,
        edge_weight=config.edge_weight,
        perceptual_weight=config.perceptual_weight,
    )
    classification_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    best_validation_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    history: list[dict[str, float | int]] = []

    for epoch in range(1, config.epochs + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            reconstruction_loss,
            classification_loss,
            config.classification_weight,
            config.gradient_clip_norm,
            device,
            optimizer,
        )
        validation_metrics = _run_epoch(
            model,
            validation_loader,
            reconstruction_loss,
            classification_loss,
            config.classification_weight,
            config.gradient_clip_norm,
            device,
            None,
        )
        row: dict[str, float | int] = {"epoch": epoch}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"validation_{key}": value for key, value in validation_metrics.items()})
        history.append(row)
        print(
            f"Attack decoder epoch {epoch:03d}/{config.epochs:03d}: "
            f"train_loss={train_metrics['loss']:.4f}, "
            f"validation_loss={validation_metrics['loss']:.4f}, "
            f"validation_ssim={validation_metrics['ssim']:.4f}"
        )
        if validation_metrics["loss"] < best_validation_loss:
            best_validation_loss = validation_metrics["loss"]
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    checkpoint = output / "client_received_decoder_best.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "decoder_config": model.config.to_dict(),
            "training_config": asdict(config),
            "best_validation_loss": best_validation_loss,
        },
        checkpoint,
    )
    with (output / "training_history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    return checkpoint, history


__all__ = ["AttackTrainingConfig", "train_client_received_decoder"]
