from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional
from torch.utils.data import DataLoader

from ...client_received_transcript_attack.models.factory import DecoderModel
from ...decoder.losses.reconstruction_loss import ReconstructionLoss
from ..models.refiner import (
    ResidualUNetRefiner,
    TranscriptConditionedResidualUNetRefiner,
)


@dataclass(frozen=True)
class RefinerTrainingConfig:
    epochs: int = 40
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    l1_weight: float = 1.0
    ssim_weight: float = 0.75
    edge_weight: float = 0.2
    perceptual_weight: float = 0.1
    residual_weight: float = 0.05
    low_frequency_weight: float = 0.05
    gradient_clip_norm: float = 5.0
    num_workers: int = 0


def _tensor(batch: dict, key: str, device: torch.device) -> Tensor:
    value = batch[key]
    if not isinstance(value, Tensor):
        raise TypeError(f"batch[{key!r}] must be a tensor")
    return value.to(device)


def _freeze_decoder(decoder: DecoderModel) -> None:
    decoder.eval()
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)


def _run_epoch(
    refiner: ResidualUNetRefiner,
    decoder: DecoderModel,
    loader: DataLoader,
    reconstruction_loss: ReconstructionLoss,
    config: RefinerTrainingConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    refiner.train(training)
    decoder.eval()
    totals = {
        "loss": 0.0,
        "reconstruction_loss": 0.0,
        "l1": 0.0,
        "ssim": 0.0,
        "edge": 0.0,
        "perceptual": 0.0,
        "residual": 0.0,
        "low_frequency": 0.0,
    }
    samples = 0
    for batch in loader:
        target = _tensor(batch, "target_image", device)
        with torch.no_grad():
            coarse, _ = decoder(
                _tensor(batch, "server_output_u", device),
                _tensor(batch, "grad_g_to_f", device),
            )
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            refined, residual = refiner(coarse)
            reconstruction_total, reconstruction_metrics = reconstruction_loss(
                refined, target
            )
            residual_loss = residual.abs().mean()
            low_frequency_loss = functional.l1_loss(
                functional.avg_pool2d(refined, kernel_size=4),
                functional.avg_pool2d(coarse, kernel_size=4),
            )
            total = (
                reconstruction_total
                + config.residual_weight * residual_loss
                + config.low_frequency_weight * low_frequency_loss
            )
            if optimizer is not None:
                total.backward()
                if config.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        refiner.parameters(), config.gradient_clip_norm
                    )
                optimizer.step()

        batch_size = int(target.shape[0])
        samples += batch_size
        totals["loss"] += float(total.detach()) * batch_size
        totals["reconstruction_loss"] += (
            float(reconstruction_total.detach()) * batch_size
        )
        for key in ("l1", "ssim", "edge", "perceptual"):
            totals[key] += float(reconstruction_metrics[key]) * batch_size
        totals["residual"] += float(residual_loss.detach()) * batch_size
        totals["low_frequency"] += float(low_frequency_loss.detach()) * batch_size

    if samples == 0:
        raise ValueError("training or validation dataset is empty")
    return {key: value / samples for key, value in totals.items()}


def train_refiner(
    refiner: ResidualUNetRefiner,
    decoder: DecoderModel,
    train_dataset,
    validation_dataset,
    output_dir: str | Path,
    config: RefinerTrainingConfig,
    device: torch.device,
    checkpoint_metadata: dict[str, object] | None = None,
) -> tuple[Path, list[dict[str, float | int]]]:
    """Train the refiner on public targets while keeping the coarse decoder frozen."""

    if config.epochs < 1 or config.batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    _freeze_decoder(decoder)
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
    optimizer = torch.optim.AdamW(
        refiner.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    initial_validation_metrics = _run_epoch(
        refiner,
        decoder,
        validation_loader,
        reconstruction_loss,
        config,
        device,
        None,
    )
    best_validation_loss = initial_validation_metrics["loss"]
    best_epoch = 0
    best_state = copy.deepcopy(refiner.state_dict())
    history: list[dict[str, float | int]] = [{
        "epoch": 0,
        **{
            f"validation_{key}": value
            for key, value in initial_validation_metrics.items()
        },
    }]
    print(
        "Refiner identity baseline: "
        f"validation_loss={initial_validation_metrics['loss']:.4f}, "
        f"validation_ssim={initial_validation_metrics['ssim']:.4f}"
    )

    for epoch in range(1, config.epochs + 1):
        train_metrics = _run_epoch(
            refiner,
            decoder,
            train_loader,
            reconstruction_loss,
            config,
            device,
            optimizer,
        )
        validation_metrics = _run_epoch(
            refiner,
            decoder,
            validation_loader,
            reconstruction_loss,
            config,
            device,
            None,
        )
        row: dict[str, float | int] = {"epoch": epoch}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update(
            {f"validation_{key}": value for key, value in validation_metrics.items()}
        )
        history.append(row)
        print(
            f"Refiner epoch {epoch:03d}/{config.epochs:03d}: "
            f"train_loss={train_metrics['loss']:.4f}, "
            f"validation_loss={validation_metrics['loss']:.4f}, "
            f"validation_ssim={validation_metrics['ssim']:.4f}"
        )
        if validation_metrics["loss"] < best_validation_loss:
            best_validation_loss = validation_metrics["loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(refiner.state_dict())

    refiner.load_state_dict(best_state)
    checkpoint = output / "residual_unet_refiner_best.pt"
    torch.save(
        {
            "model": refiner.state_dict(),
            "refiner_type": refiner.refiner_type,
            "refiner_config": refiner.config.to_dict(),
            "training_config": asdict(config),
            "best_validation_loss": best_validation_loss,
            "best_epoch": best_epoch,
            "metadata": checkpoint_metadata or {},
        },
        checkpoint,
    )
    with (output / "training_history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    return checkpoint, history


def _run_conditioned_epoch(
    refiner: TranscriptConditionedResidualUNetRefiner,
    loader: DataLoader,
    reconstruction_loss: ReconstructionLoss,
    config: RefinerTrainingConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    refiner.train(training)
    totals = {
        "loss": 0.0,
        "reconstruction_loss": 0.0,
        "l1": 0.0,
        "ssim": 0.0,
        "edge": 0.0,
        "perceptual": 0.0,
        "residual": 0.0,
        "low_frequency": 0.0,
    }
    samples = 0
    for batch in loader:
        target = _tensor(batch, "target_image", device)
        coarse = _tensor(batch, "coarse_reconstruction", device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            refined, residual = refiner(
                coarse,
                _tensor(batch, "server_output_u", device),
                _tensor(batch, "grad_g_to_f", device),
            )
            reconstruction_total, reconstruction_metrics = reconstruction_loss(
                refined, target
            )
            residual_loss = residual.abs().mean()
            low_frequency_loss = functional.l1_loss(
                functional.avg_pool2d(refined, kernel_size=4),
                functional.avg_pool2d(coarse, kernel_size=4),
            )
            total = (
                reconstruction_total
                + config.residual_weight * residual_loss
                + config.low_frequency_weight * low_frequency_loss
            )
            if optimizer is not None:
                total.backward()
                if config.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        refiner.parameters(), config.gradient_clip_norm
                    )
                optimizer.step()

        batch_size = int(target.shape[0])
        samples += batch_size
        totals["loss"] += float(total.detach()) * batch_size
        totals["reconstruction_loss"] += (
            float(reconstruction_total.detach()) * batch_size
        )
        for key in ("l1", "ssim", "edge", "perceptual"):
            totals[key] += float(reconstruction_metrics[key]) * batch_size
        totals["residual"] += float(residual_loss.detach()) * batch_size
        totals["low_frequency"] += float(low_frequency_loss.detach()) * batch_size

    if samples == 0:
        raise ValueError("training or validation dataset is empty")
    return {key: value / samples for key, value in totals.items()}


def train_conditioned_refiner(
    refiner: TranscriptConditionedResidualUNetRefiner,
    train_dataset,
    validation_dataset,
    output_dir: str | Path,
    config: RefinerTrainingConfig,
    device: torch.device,
    checkpoint_metadata: dict[str, object] | None = None,
) -> tuple[Path, list[dict[str, float | int]]]:
    """Train from OOF coarse images plus raw attacker-visible u and dL/dz."""

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
    optimizer = torch.optim.AdamW(
        refiner.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    initial_validation_metrics = _run_conditioned_epoch(
        refiner,
        validation_loader,
        reconstruction_loss,
        config,
        device,
        None,
    )
    best_validation_loss = initial_validation_metrics["loss"]
    best_epoch = 0
    best_state = copy.deepcopy(refiner.state_dict())
    history: list[dict[str, float | int]] = [{
        "epoch": 0,
        **{
            f"validation_{key}": value
            for key, value in initial_validation_metrics.items()
        },
    }]
    print(
        "Conditioned refiner identity baseline: "
        f"validation_loss={initial_validation_metrics['loss']:.4f}, "
        f"validation_ssim={initial_validation_metrics['ssim']:.4f}"
    )

    for epoch in range(1, config.epochs + 1):
        train_metrics = _run_conditioned_epoch(
            refiner,
            train_loader,
            reconstruction_loss,
            config,
            device,
            optimizer,
        )
        validation_metrics = _run_conditioned_epoch(
            refiner,
            validation_loader,
            reconstruction_loss,
            config,
            device,
            None,
        )
        row: dict[str, float | int] = {"epoch": epoch}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update(
            {f"validation_{key}": value for key, value in validation_metrics.items()}
        )
        history.append(row)
        print(
            f"Conditioned refiner epoch {epoch:03d}/{config.epochs:03d}: "
            f"train_loss={train_metrics['loss']:.4f}, "
            f"validation_loss={validation_metrics['loss']:.4f}, "
            f"validation_ssim={validation_metrics['ssim']:.4f}"
        )
        if validation_metrics["loss"] < best_validation_loss:
            best_validation_loss = validation_metrics["loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(refiner.state_dict())

    refiner.load_state_dict(best_state)
    checkpoint = output / "transcript_conditioned_refiner_best.pt"
    torch.save(
        {
            "model": refiner.state_dict(),
            "refiner_type": refiner.refiner_type,
            "refiner_config": refiner.config.to_dict(),
            "training_config": asdict(config),
            "best_validation_loss": best_validation_loss,
            "best_epoch": best_epoch,
            "metadata": checkpoint_metadata or {},
        },
        checkpoint,
    )
    with (output / "training_history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    return checkpoint, history


__all__ = [
    "RefinerTrainingConfig",
    "train_conditioned_refiner",
    "train_refiner",
]
