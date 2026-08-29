from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional

from ..losses.alignment import ConditionalTranscriptAlignmentLoss
from ..losses.gradient_matching import GradientMatchingLoss, soft_cross_entropy
from ..losses.reconstruction import ReconstructionLoss
from .paired_decoder_trainer import decoder_forward


@dataclass(frozen=True)
class ABTRConfig:
    epochs: int = 30
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    reconstruction_weight: float = 1.0
    alignment_weight: float = 0.5
    gradient_matching_weight: float = 0.5
    public_classification_weight: float = 0.05
    paired_weight: float = 0.25


def _normalize_public_image(image: Tensor) -> Tensor:
    mean = image.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
    standard_deviation = image.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
    return (image - mean) / standard_deviation


class ABTRTrainer:
    """Strict/semi-unpaired architecture-agnostic bidirectional reconstruction."""

    def __init__(self, config: ABTRConfig, device: torch.device, num_classes: int) -> None:
        self.config = config
        self.device = device
        self.num_classes = num_classes
        self.gradient_loss = GradientMatchingLoss()
        self.alignment_loss = ConditionalTranscriptAlignmentLoss()
        self.reconstruction_loss = ReconstructionLoss()

    def _simulated_transcript(
        self,
        front: nn.Module,
        server: nn.Module,
        tail: nn.Module,
        image: Tensor,
        labels: Tensor,
    ) -> tuple[dict[str, Tensor], Tensor, Tensor]:
        condition = functional.one_hot(labels, self.num_classes).float()
        z = front(_normalize_public_image(image))
        u = server(z)
        classification = functional.cross_entropy(tail(u), labels)
        gradient_objective = classification * image.shape[0]
        grad_u = torch.autograd.grad(gradient_objective, u, create_graph=True, retain_graph=True)[0]
        grad_z = torch.autograd.grad(
            u, z, grad_outputs=grad_u, create_graph=True, retain_graph=True
        )[0]
        return {"z": z, "u": u, "grad_u": grad_u, "grad_z": grad_z}, condition, classification

    def fit(
        self,
        front: nn.Module,
        server: nn.Module,
        tail: nn.Module,
        decoder: nn.Module,
        real_loader,
        public_loader,
        output_dir: str | Path,
        paired_loader=None,
    ) -> list[dict[str, float | int]]:
        modules = (front, server, tail, decoder)
        for module in modules:
            module.to(self.device)
        server.eval()
        for parameter in server.parameters():
            parameter.requires_grad_(False)
        trainable = list(front.parameters()) + list(tail.parameters()) + list(decoder.parameters())
        optimizer = torch.optim.AdamW(
            trainable, lr=self.config.learning_rate, weight_decay=self.config.weight_decay
        )
        history: list[dict[str, float | int]] = []
        for epoch in range(1, self.config.epochs + 1):
            front.train()
            tail.train()
            decoder.train()
            totals = {"loss": 0.0, "reconstruction": 0.0, "alignment": 0.0, "gradient": 0.0}
            samples = 0
            public_iterator = iter(public_loader)
            paired_iterator = iter(paired_loader) if paired_loader is not None else None
            for real_batch in real_loader:
                try:
                    public_batch = next(public_iterator)
                except StopIteration:
                    public_iterator = iter(public_loader)
                    public_batch = next(public_iterator)
                image = public_batch["image"].to(self.device)
                labels = public_batch["label"].to(self.device)
                simulated, simulated_condition, public_classification = self._simulated_transcript(
                    front, server, tail, image, labels
                )
                observed = {
                    "z": real_batch["smashed_z"].to(self.device),
                    "u": real_batch["server_output_u"].to(self.device),
                    "grad_u": real_batch["grad_h_to_g"].to(self.device),
                }
                has_grad_z = bool(real_batch["has_grad_g_to_f"].all())
                if has_grad_z:
                    observed["grad_z"] = real_batch["grad_g_to_f"].to(self.device)
                observed_condition = real_batch["label_condition"].to(self.device)

                reconstruction = decoder(
                    simulated["z"], simulated["u"], simulated["grad_u"], simulated["grad_z"], simulated_condition
                )
                reconstruction_loss, _ = self.reconstruction_loss(reconstruction, image)
                alignment_loss, _ = self.alignment_loss(
                    simulated, observed, simulated_condition, observed_condition
                )

                real_u = observed["u"].detach().requires_grad_(True)
                real_objective = (
                    soft_cross_entropy(tail(real_u), observed_condition) * real_u.shape[0]
                )
                predicted_real_gradient = torch.autograd.grad(
                    real_objective, real_u, create_graph=True
                )[0]
                gradient_loss, _ = self.gradient_loss(
                    predicted_real_gradient, observed["grad_u"]
                )
                total = (
                    self.config.reconstruction_weight * reconstruction_loss
                    + self.config.alignment_weight * alignment_loss
                    + self.config.gradient_matching_weight * gradient_loss
                    + self.config.public_classification_weight * public_classification
                )
                if paired_iterator is not None:
                    try:
                        paired_batch = next(paired_iterator)
                    except StopIteration:
                        paired_iterator = iter(paired_loader)
                        paired_batch = next(paired_iterator)
                    paired_reconstruction = decoder_forward(decoder, paired_batch, self.device)
                    paired_loss, _ = self.reconstruction_loss(
                        paired_reconstruction, paired_batch["target_image"].to(self.device)
                    )
                    total = total + self.config.paired_weight * paired_loss

                optimizer.zero_grad(set_to_none=True)
                total.backward()
                optimizer.step()
                batch_size = image.shape[0]
                totals["loss"] += float(total.detach()) * batch_size
                totals["reconstruction"] += float(reconstruction_loss.detach()) * batch_size
                totals["alignment"] += float(alignment_loss.detach()) * batch_size
                totals["gradient"] += float(gradient_loss.detach()) * batch_size
                samples += batch_size
            if samples == 0:
                raise ValueError("real transcript loader is empty")
            row: dict[str, float | int] = {"epoch": epoch}
            row.update({key: value / samples for key, value in totals.items()})
            history.append(row)
            print(
                f"ABTR {epoch:03d}/{self.config.epochs:03d}: "
                f"loss={row['loss']:.4f}, alignment={row['alignment']:.4f}"
            )

        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "front": front.state_dict(),
                "tail": tail.state_dict(),
                "decoder": decoder.state_dict(),
                "decoder_config": decoder.config.to_dict(),
                "training_config": asdict(self.config),
            },
            output / "abtr_models.pt",
        )
        with (output / "abtr_history.json").open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)
        return history


__all__ = ["ABTRConfig", "ABTRTrainer"]
