from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from ..losses.gradient_matching import GradientMatchingLoss, soft_cross_entropy


@dataclass(frozen=True)
class TailSimulatorConfig:
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    classification_weight: float = 0.05


class TailSimulatorTrainer:
    """Fit h-hat exclusively from attacker-visible transcript values."""

    def __init__(self, config: TailSimulatorConfig, device: torch.device) -> None:
        self.config = config
        self.device = device
        self.gradient_loss = GradientMatchingLoss()

    def fit(self, model: nn.Module, loader, output_dir: str | Path) -> list[dict[str, float | int]]:
        model.to(self.device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        history: list[dict[str, float | int]] = []
        for epoch in range(1, self.config.epochs + 1):
            model.train()
            totals = {"loss": 0.0, "gradient_cosine": 0.0}
            samples = 0
            for batch in loader:
                server_output = batch["server_output_u"].to(self.device).detach().requires_grad_(True)
                condition = batch["label_condition"].to(self.device)
                observed_gradient = batch["grad_h_to_g"].to(self.device)
                logits = model(server_output)
                classification = soft_cross_entropy(logits, condition)
                gradient_objective = classification * server_output.shape[0]
                predicted_gradient = torch.autograd.grad(
                    gradient_objective, server_output, create_graph=True
                )[0]
                gradient_loss, gradient_metrics = self.gradient_loss(
                    predicted_gradient, observed_gradient
                )
                loss = gradient_loss + self.config.classification_weight * classification
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                batch_size = server_output.shape[0]
                totals["loss"] += float(loss.detach()) * batch_size
                totals["gradient_cosine"] += gradient_metrics["gradient_cosine"] * batch_size
                samples += batch_size
            if samples == 0:
                raise ValueError("transcript loader is empty")
            row: dict[str, float | int] = {
                "epoch": epoch,
                "loss": totals["loss"] / samples,
                "gradient_cosine": totals["gradient_cosine"] / samples,
            }
            history.append(row)
            print(
                f"Tail simulator {epoch:03d}/{self.config.epochs:03d}: "
                f"loss={row['loss']:.4f}, cosine={row['gradient_cosine']:.4f}"
            )
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model": model.state_dict(), "training_config": asdict(self.config)},
            output / "tail_simulator.pt",
        )
        with (output / "tail_history.json").open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)
        return history


__all__ = ["TailSimulatorConfig", "TailSimulatorTrainer"]
