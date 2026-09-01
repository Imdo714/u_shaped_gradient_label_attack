from __future__ import annotations

import csv
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from ...split_learning.architecture.split_learning_model import SplitLearningModel
from ...split_learning.gradient_flow.gradient_exchange import observe_frozen_gradient_exchange
from .gradient_label_predictor import GradientLabelPredictor
from .image_scaling import denormalize_image, normalize_image


LabelMode = Literal["oracle", "inferred-hard", "inferred-soft"]
ConditionMode = Literal["stored", "oracle", "zero"]


def _one_hot(label: int, num_classes: int) -> Tensor:
    result = torch.zeros(num_classes)
    result[label] = 1.0
    return result


def collect_observations(
    model: SplitLearningModel,
    loader,
    output_dir: str | Path,
    device: torch.device,
    num_classes: int,
    label_mode: LabelMode,
    predictor: GradientLabelPredictor | None = None,
    max_samples: int | None = None,
) -> Path:
    """Persist attacker signals separately from evaluator-only original images."""
    if label_mode != "oracle" and predictor is None:
        raise ValueError(f"label_mode={label_mode} requires a GradientLabelPredictor")
    root = Path(output_dir)
    attacker_dir = root / "attacker_records"
    evaluator_dir = root / "evaluator_targets"
    attacker_dir.mkdir(parents=True, exist_ok=True)
    evaluator_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    criterion = nn.CrossEntropyLoss()
    collected = 0
    for images, labels, sample_ids in loader:
        for index, sample_id in enumerate(sample_ids):
            if max_samples is not None and collected >= max_samples:
                break
            image = images[index : index + 1].to(device)
            label_tensor = labels[index : index + 1].to(device)
            result = observe_frozen_gradient_exchange(model, image, label_tensor, criterion)
            true_label = int(label_tensor.item())
            if label_mode == "oracle":
                predicted_label = true_label
                condition = _one_hot(true_label, num_classes)
                confidence = 1.0
                cluster_id = -1
            else:
                prediction = predictor.predict(
                    result.grad_h_to_g[0], soft=label_mode == "inferred-soft"
                )
                predicted_label = prediction.label
                condition = prediction.probabilities
                confidence = prediction.confidence
                cluster_id = prediction.cluster_id

            attacker_name = f"{sample_id}.npz"
            evaluator_name = f"{sample_id}.npz"
            np.savez_compressed(
                attacker_dir / attacker_name,
                smashed_z=result.smashed_z[0].cpu().numpy(),
                server_output_u=result.server_output_u[0].cpu().numpy(),
                grad_h_to_g=result.grad_h_to_g[0].cpu().numpy(),
                grad_g_to_f=result.grad_g_to_f[0].cpu().numpy(),
                label_condition=condition.cpu().numpy(),
                predicted_label=np.int64(predicted_label),
                confidence=np.float32(confidence),
                cluster_id=np.int64(cluster_id),
            )
            np.savez_compressed(
                evaluator_dir / evaluator_name,
                target_image=denormalize_image(image[0]).cpu().numpy(),
                true_label=np.int64(true_label),
            )
            rows.append(
                {
                    "sample_id": sample_id,
                    "attacker_record": f"attacker_records/{attacker_name}",
                    "evaluator_target": f"evaluator_targets/{evaluator_name}",
                }
            )
            collected += 1
        if max_samples is not None and collected >= max_samples:
            break
    manifest = root / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["sample_id", "attacker_record", "evaluator_target"]
        )
        writer.writeheader()
        writer.writerows(rows)
    return manifest


class ObservationDataset(Dataset):
    """Paired controlled dataset; target files are never placed in attacker records."""

    def __init__(self, manifest_path: str | Path, include_target: bool = True) -> None:
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        self.include_target = include_target
        with self.manifest_path.open("r", newline="", encoding="utf-8") as handle:
            self.rows = list(csv.DictReader(handle))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        row = self.rows[index]
        with np.load(self.root / row["attacker_record"]) as record:
            has_grad_g_to_f = "grad_g_to_f" in record.files
            smashed_z = torch.from_numpy(record["smashed_z"]).float()
            item: dict[str, Tensor | str] = {
                "sample_id": row["sample_id"],
                "smashed_z": smashed_z,
                "server_output_u": torch.from_numpy(record["server_output_u"]).float(),
                "grad_h_to_g": torch.from_numpy(record["grad_h_to_g"]).float(),
                "grad_g_to_f": (
                    torch.from_numpy(record["grad_g_to_f"]).float()
                    if has_grad_g_to_f
                    else torch.zeros_like(smashed_z)
                ),
                "has_grad_g_to_f": torch.tensor(has_grad_g_to_f),
                "label_condition": torch.from_numpy(record["label_condition"]).float(),
                "predicted_label": torch.tensor(int(record["predicted_label"])),
                "confidence": torch.tensor(float(record["confidence"])),
            }
        if self.include_target:
            with np.load(self.root / row["evaluator_target"]) as target:
                item["target_image"] = torch.from_numpy(target["target_image"]).float()
                item["true_label"] = torch.tensor(int(target["true_label"]))
        return item


class ConditionedObservationDataset(Dataset):
    """View observations with an explicit label-conditioning ablation."""

    def __init__(
        self,
        source: Dataset,
        condition_mode: ConditionMode,
        num_classes: int,
    ) -> None:
        if condition_mode not in ("stored", "oracle", "zero"):
            raise ValueError(f"unknown condition mode: {condition_mode}")
        self.source = source
        self.condition_mode = condition_mode
        self.num_classes = num_classes

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        item = dict(self.source[index])
        if self.condition_mode == "zero":
            item["label_condition"] = torch.zeros(self.num_classes)
        elif self.condition_mode == "oracle":
            true_label = int(item["true_label"])
            item["label_condition"] = _one_hot(true_label, self.num_classes)
            item["predicted_label"] = torch.tensor(true_label)
        return item


def collect_surrogate_observations(
    source_dataset: ObservationDataset,
    surrogate_f: nn.Module,
    server_g: nn.Module,
    surrogate_h: nn.Module,
    output_dir: str | Path,
    device: torch.device,
    num_classes: int,
) -> Path:
    """Generate auxiliary records through the learned f-hat/g/h-hat path."""
    root = Path(output_dir)
    attacker_dir = root / "attacker_records"
    evaluator_dir = root / "evaluator_targets"
    attacker_dir.mkdir(parents=True, exist_ok=True)
    evaluator_dir.mkdir(parents=True, exist_ok=True)
    surrogate_f.eval()
    server_g.eval()
    surrogate_h.eval()
    criterion = nn.CrossEntropyLoss()
    rows: list[dict[str, object]] = []
    for batch in DataLoader(source_dataset, batch_size=1, shuffle=False):
        sample_id = batch["sample_id"][0]
        target = batch["target_image"].to(device)
        true_label = batch["true_label"].to(device)
        z = surrogate_f(normalize_image(target))
        u_server = server_g(z)
        u = u_server.detach().requires_grad_(True)
        loss = criterion(surrogate_h(u), true_label)
        gradient = torch.autograd.grad(loss, u)[0]
        grad_g_to_f = torch.autograd.grad(
            u_server, z, grad_outputs=gradient
        )[0]
        label = int(true_label.item())
        condition = _one_hot(label, num_classes)
        name = f"{sample_id}.npz"
        np.savez_compressed(
            attacker_dir / name,
            smashed_z=z[0].detach().cpu().numpy(),
            server_output_u=u[0].detach().cpu().numpy(),
            grad_h_to_g=gradient[0].detach().cpu().numpy(),
            grad_g_to_f=grad_g_to_f[0].detach().cpu().numpy(),
            label_condition=condition.numpy(),
            predicted_label=np.int64(label),
            confidence=np.float32(1.0),
            cluster_id=np.int64(-1),
        )
        np.savez_compressed(
            evaluator_dir / name,
            target_image=target[0].detach().cpu().numpy(),
            true_label=np.int64(label),
        )
        rows.append(
            {
                "sample_id": sample_id,
                "attacker_record": f"attacker_records/{name}",
                "evaluator_target": f"evaluator_targets/{name}",
            }
        )
    manifest = root / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["sample_id", "attacker_record", "evaluator_target"]
        )
        writer.writeheader()
        writer.writerows(rows)
    return manifest


__all__ = [
    "LabelMode",
    "ConditionMode",
    "ConditionedObservationDataset",
    "ObservationDataset",
    "collect_observations",
    "collect_surrogate_observations",
]
