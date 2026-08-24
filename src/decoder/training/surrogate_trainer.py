from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..data.image_scaling import normalize_image


def _cosine_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    similarity = torch.nn.functional.cosine_similarity(
        prediction.flatten(1), target.flatten(1), dim=1
    )
    return 1.0 - similarity.mean()


def train_surrogate_f(
    model: nn.Module,
    dataset,
    output_path: str | Path,
    device: torch.device,
    epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
) -> Path:
    """Fit f-hat(x) to the observed smashed data z."""
    model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(1, epochs + 1):
        total = 0.0
        count = 0
        for batch in loader:
            images = normalize_image(batch["target_image"].to(device))
            target = batch["smashed_z"].to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(images)
            loss = torch.nn.functional.mse_loss(prediction, target) + _cosine_loss(
                prediction, target
            )
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * images.shape[0]
            count += images.shape[0]
        print(f"Surrogate f epoch {epoch:03d}/{epochs:03d}: loss={total / count:.5f}")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return path


def train_surrogate_h(
    model: nn.Module,
    dataset,
    output_path: str | Path,
    device: torch.device,
    epochs: int = 5,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
) -> Path:
    """Fit h-hat by matching labels and the observed gradient dL/du."""
    model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    for epoch in range(1, epochs + 1):
        total = 0.0
        count = 0
        for batch in loader:
            u = batch["server_output_u"].to(device).detach().requires_grad_(True)
            labels = batch["true_label"].to(device)
            target_gradient = batch["grad_h_to_g"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(u)
            classification_sum = criterion(logits, labels)
            predicted_gradient = torch.autograd.grad(
                classification_sum, u, create_graph=True
            )[0]
            matching_loss = torch.nn.functional.mse_loss(
                predicted_gradient, target_gradient
            ) + _cosine_loss(predicted_gradient, target_gradient)
            loss = classification_sum / u.shape[0] + matching_loss
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * u.shape[0]
            count += u.shape[0]
        print(f"Surrogate h epoch {epoch:03d}/{epochs:03d}: loss={total / count:.5f}")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return path


__all__ = ["train_surrogate_f", "train_surrogate_h"]
