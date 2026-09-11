from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from ...decoder.data.image_scaling import IMAGENET_MEAN, IMAGENET_STD
from ...shared.data.class_catalog import ClassCatalog
from ...shared.data.image_dataset import (
    ImageFolderWithID,
    image_transform,
    make_loader,
    validate_class_mapping,
)
from ...shared.reproducibility.random_seed import seed_everything
from ...split_learning.architecture.split_learning_model import SplitLearningModel
from ...split_learning.f_model.client_front_f_model import ClientFrontFModel
from ...split_learning.g_model.server_middle_g_model import ServerMiddleGModel
from ...split_learning.gradient_flow.gradient_exchange import (
    observe_frozen_gradient_exchange,
    run_gradient_exchange_step,
)
from ...split_learning.h_model.client_tail_h_model import ClientTailHModel
from .run_joint_transcript_attack import (
    _classification_accuracy,
    _device,
    _file_hashes,
    _warmup_classifier,
)


SIGNAL_MODES = ("u_only", "gradient_only", "u_gradient")
AUGMENTATION_MODES = ("none", "fixed", "dynamic")
TEMPORAL_MODES = ("single_latest", "multi_natural", "multi_matched")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a paper-oriented online U-shaped Split Learning label attack. "
            "One live server g is shared by an attacker client and victim clients; "
            "the attack is trained only from time-indexed attacker transcripts."
        )
    )
    base = "workspace/data/shared_pretrained_encoder_joint_attack/animal5"
    parser.add_argument("--pretrained-autoencoder", required=True)
    parser.add_argument("--pretrain-data", default=f"{base}/pretrain")
    parser.add_argument("--attacker-data", default=f"{base}/attacker")
    parser.add_argument("--victim-data", default=f"{base}/victim")
    parser.add_argument(
        "--output",
        default=(
            "workspace/results/shared_pretrained_encoder_drift_attack/"
            "animal5_online_transcript_core"
        ),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--num-victims", type=int, default=2)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--global-rounds", type=int, default=20)
    parser.add_argument(
        "--collection-rounds", nargs="+", type=int, default=[1, 5, 10, 20]
    )
    parser.add_argument(
        "--augmentation-modes",
        nargs="+",
        choices=AUGMENTATION_MODES,
        default=list(AUGMENTATION_MODES),
    )
    parser.add_argument("--attack-epochs", type=int, default=30)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--attack-batch-size", type=int, default=8)
    parser.add_argument("--client-learning-rate", type=float, default=1e-3)
    parser.add_argument("--server-learning-rate", type=float, default=1e-3)
    parser.add_argument("--attack-learning-rate", type=float, default=1e-3)
    parser.add_argument("--signal-spatial-size", type=int, default=8)
    parser.add_argument("--signal-channels", type=int, default=64)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--attack-validation-fraction", type=float, default=0.2)
    parser.add_argument(
        "--matched-budget",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include a multi-window condition downsampled to the latest-window budget.",
    )
    parser.add_argument(
        "--save-transcript-tensors",
        action="store_true",
        help="Persist attack-visible tensors. Off by default because tensors are large.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser


def _validate_args(args: argparse.Namespace) -> tuple[int, ...]:
    if args.num_victims < 1:
        raise ValueError("--num-victims must be positive")
    if args.global_rounds < 1:
        raise ValueError("--global-rounds must be positive")
    rounds = tuple(sorted(set(args.collection_rounds)))
    if not rounds or rounds[0] < 1 or rounds[-1] > args.global_rounds:
        raise ValueError(
            "--collection-rounds must be between 1 and --global-rounds; "
            "round 0 is intentionally excluded from the main online protocol"
        )
    if not 0.0 < args.attack_validation_fraction < 0.5:
        raise ValueError("--attack-validation-fraction must be between 0 and 0.5")
    for name in (
        "client_learning_rate",
        "server_learning_rate",
        "attack_learning_rate",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    return rounds


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _full_augmentation(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size, scale=(0.72, 1.0), ratio=(0.88, 1.12)
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.18, contrast=0.18, saturation=0.18, hue=0.03
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class CachedImageDataset(Dataset):
    def __init__(self, images: Tensor, labels: Tensor, sample_ids: Sequence[str]) -> None:
        if len(images) != len(labels) or len(images) != len(sample_ids):
            raise ValueError("cached image, label, and ID counts must match")
        self.images = images
        self.labels = labels
        self.sample_ids = tuple(sample_ids)
        self.targets = labels.tolist()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, str]:
        return self.images[index], self.labels[index], self.sample_ids[index]


def _materialize_fixed_attacker_dataset(
    data_root: str | Path,
    catalog: ClassCatalog,
    image_size: int,
    seed: int,
) -> CachedImageDataset:
    seed_everything(seed)
    dataset = ImageFolderWithID(
        Path(data_root) / "train",
        transform=_full_augmentation(image_size),
        allow_empty=True,
    )
    validate_class_mapping(dataset, catalog.names)
    images: list[Tensor] = []
    labels: list[Tensor] = []
    sample_ids: list[str] = []
    for index in range(len(dataset)):
        image, label, sample_id = dataset[index]
        images.append(image)
        labels.append(torch.tensor(label, dtype=torch.long))
        sample_ids.append(sample_id)
    return CachedImageDataset(torch.stack(images), torch.stack(labels), sample_ids)


def _attacker_dataset(
    mode: str,
    data_root: str | Path,
    catalog: ClassCatalog,
    image_size: int,
    seed: int,
) -> Dataset:
    if mode == "fixed":
        return _materialize_fixed_attacker_dataset(
            data_root, catalog, image_size, seed
        )
    transform = image_transform(image_size, augment=False)
    if mode == "dynamic":
        transform = _full_augmentation(image_size)
    dataset = ImageFolderWithID(
        Path(data_root) / "train", transform=transform, allow_empty=True
    )
    validate_class_mapping(dataset, catalog.names)
    return dataset


def _partition_victim_dataset(
    data_root: str | Path,
    split: str,
    catalog: ClassCatalog,
    image_size: int,
    num_victims: int,
    seed: int,
    *,
    augment: bool,
) -> tuple[list[Subset], list[dict[str, object]]]:
    dataset = ImageFolderWithID(
        Path(data_root) / split,
        transform=image_transform(image_size, augment=augment),
        allow_empty=True,
    )
    validate_class_mapping(dataset, catalog.names)
    partitions: list[list[int]] = [[] for _ in range(num_victims)]
    rows: list[dict[str, object]] = []
    for label in range(catalog.num_classes):
        indices = [i for i, target in enumerate(dataset.targets) if target == label]
        random.Random(f"{seed}:victim:{split}:{label}").shuffle(indices)
        for position, index in enumerate(indices):
            victim_index = position % num_victims
            partitions[victim_index].append(index)
            path, _ = dataset.samples[index]
            rows.append(
                {
                    "split": split,
                    "victim": f"victim_{victim_index + 1}",
                    "sample_id": dataset.sample_id(index),
                    "class_name": catalog.names[label],
                    "evaluator_path": str(Path(path).resolve()),
                }
            )
    return [Subset(dataset, sorted(indices)) for indices in partitions], rows


def _loader(
    dataset: Dataset,
    batch_size: int,
    seed: int,
    *,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
    )


@dataclass
class TranscriptBundle:
    u: Tensor
    grad_z: Tensor
    labels: Tensor
    sample_ids: tuple[str, ...]
    rounds: Tensor
    server_steps: Tensor

    def __len__(self) -> int:
        return len(self.labels)

    def select(self, indices: Tensor | Sequence[int]) -> "TranscriptBundle":
        if not isinstance(indices, Tensor):
            indices = torch.tensor(indices, dtype=torch.long)
        if indices.dtype == torch.bool:
            index_list = indices.nonzero(as_tuple=False).flatten().tolist()
        else:
            index_list = indices.flatten().tolist()
        return TranscriptBundle(
            self.u[indices],
            self.grad_z[indices],
            self.labels[indices],
            tuple(self.sample_ids[index] for index in index_list),
            self.rounds[indices],
            self.server_steps[indices],
        )


class TranscriptAccumulator:
    def __init__(self) -> None:
        self._u: list[Tensor] = []
        self._grad: list[Tensor] = []
        self._labels: list[Tensor] = []
        self._ids: list[str] = []
        self._rounds: list[Tensor] = []
        self._steps: list[Tensor] = []

    def add(
        self,
        u: Tensor,
        grad_z: Tensor,
        labels: Tensor,
        sample_ids: Sequence[str],
        round_index: int,
        server_step: int,
    ) -> None:
        count = len(labels)
        if len(sample_ids) != count:
            raise ValueError("transcript batch IDs and labels must match")
        self._u.append(u.detach().cpu())
        self._grad.append(grad_z.detach().cpu())
        self._labels.append(labels.detach().cpu())
        self._ids.extend(sample_ids)
        self._rounds.append(torch.full((count,), round_index, dtype=torch.long))
        self._steps.append(torch.full((count,), server_step, dtype=torch.long))

    def bundle(self) -> TranscriptBundle:
        if not self._labels:
            raise ValueError("no transcripts were collected")
        return TranscriptBundle(
            torch.cat(self._u),
            torch.cat(self._grad),
            torch.cat(self._labels),
            tuple(self._ids),
            torch.cat(self._rounds),
            torch.cat(self._steps),
        )


@dataclass
class ClientRuntime:
    name: str
    model: SplitLearningModel
    train_dataset: Dataset
    validation_dataset: Dataset
    optimizer_f: torch.optim.Optimizer
    optimizer_h: torch.optim.Optimizer


def _normalized_weight_distance(module: nn.Module, initial: dict[str, Tensor]) -> float:
    difference = 0.0
    baseline = 0.0
    current = module.state_dict()
    for name, initial_value in initial.items():
        value = current[name].detach().float().cpu()
        reference = initial_value.detach().float().cpu()
        difference += float((value - reference).pow(2).sum())
        baseline += float(reference.pow(2).sum())
    return math.sqrt(difference) / max(math.sqrt(baseline), 1e-12)


def _train_client_local_epoch(
    client: ClientRuntime,
    optimizer_g: torch.optim.Optimizer,
    loader: DataLoader,
    device: torch.device,
    round_index: int,
    server_step: int,
    collector: TranscriptAccumulator | None,
) -> tuple[dict[str, float], int]:
    criterion = nn.CrossEntropyLoss()
    client.model.train()
    correct = samples = 0
    loss_sum = 0.0
    for images, labels, sample_ids in loader:
        images = images.to(device)
        labels = labels.to(device)
        exchange = run_gradient_exchange_step(
            client.model,
            images,
            labels,
            criterion,
            client.optimizer_f,
            optimizer_g,
            client.optimizer_h,
            update=True,
        )
        if collector is not None:
            collector.add(
                exchange.server_output_u,
                exchange.grad_g_to_f,
                labels,
                tuple(sample_ids),
                round_index,
                server_step,
            )
        loss_sum += exchange.loss * len(labels)
        correct += int((exchange.logits.argmax(1) == labels).sum())
        samples += len(labels)
        server_step += 1
    return {"loss": loss_sum / samples, "accuracy": correct / samples}, server_step


def _observe_bundle(
    model: SplitLearningModel,
    loader: DataLoader,
    device: torch.device,
    round_index: int,
    server_step: int,
) -> TranscriptBundle:
    collector = TranscriptAccumulator()
    criterion = nn.CrossEntropyLoss()
    for images, labels, sample_ids in loader:
        images = images.to(device)
        labels = labels.to(device)
        exchange = observe_frozen_gradient_exchange(model, images, labels, criterion)
        collector.add(
            exchange.server_output_u,
            exchange.grad_g_to_f,
            labels,
            tuple(sample_ids),
            round_index,
            server_step,
        )
    return collector.bundle()


def _groups(channels: int) -> int:
    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return groups


class _SignalAdapter(nn.Module):
    def __init__(self, inputs: int, outputs: int, spatial_size: int) -> None:
        super().__init__()
        self.spatial_size = spatial_size
        self.network = nn.Sequential(
            nn.Conv2d(inputs, outputs, 1),
            nn.GroupNorm(_groups(outputs), outputs),
            nn.SiLU(),
            nn.Conv2d(outputs, outputs, 3, padding=1),
            nn.GroupNorm(_groups(outputs), outputs),
            nn.SiLU(),
        )

    def forward(self, value: Tensor) -> Tensor:
        value = nn.functional.adaptive_avg_pool2d(
            value, (self.spatial_size, self.spatial_size)
        )
        return self.network(value)


@dataclass(frozen=True)
class OnlineLabelClassifierConfig:
    u_channels: int
    grad_z_channels: int
    num_classes: int
    signal_mode: str
    signal_spatial_size: int = 8
    signal_channels: int = 64
    hidden_channels: int = 128
    norm_channels: int = 16
    dropout: float = 0.2

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class OnlineLabelClassifier(nn.Module):
    """Unified u/dLdz classifier that retains gradient direction and magnitude."""

    def __init__(self, config: OnlineLabelClassifierConfig) -> None:
        super().__init__()
        if config.signal_mode not in SIGNAL_MODES:
            raise ValueError(f"unknown signal mode: {config.signal_mode}")
        self.config = config
        use_u = config.signal_mode in ("u_only", "u_gradient")
        use_gradient = config.signal_mode in ("gradient_only", "u_gradient")
        self.u_encoder = (
            _SignalAdapter(
                config.u_channels, config.signal_channels, config.signal_spatial_size
            )
            if use_u
            else None
        )
        self.gradient_encoder = (
            _SignalAdapter(
                config.grad_z_channels,
                config.signal_channels,
                config.signal_spatial_size,
            )
            if use_gradient
            else None
        )
        self.norm_encoder = (
            nn.Sequential(
                nn.Linear(1, config.norm_channels),
                nn.SiLU(),
                nn.Linear(config.norm_channels, config.norm_channels),
                nn.SiLU(),
            )
            if use_gradient
            else None
        )
        features = config.signal_channels * (int(use_u) + int(use_gradient))
        features += config.norm_channels if use_gradient else 0
        self.classifier = nn.Sequential(
            nn.Linear(features, config.hidden_channels),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_channels, config.num_classes),
        )

    @staticmethod
    def _pool(value: Tensor) -> Tensor:
        return nn.functional.adaptive_avg_pool2d(value, 1).flatten(1)

    def forward(self, u: Tensor, grad_z: Tensor) -> Tensor:
        features: list[Tensor] = []
        if self.u_encoder is not None:
            features.append(self._pool(self.u_encoder(u)))
        if self.gradient_encoder is not None and self.norm_encoder is not None:
            norm = grad_z.flatten(1).norm(dim=1).clamp_min(1e-12)
            direction = grad_z / norm.view(-1, 1, 1, 1)
            features.append(self._pool(self.gradient_encoder(direction)))
            features.append(self.norm_encoder((torch.log10(norm) / 10.0).unsqueeze(1)))
        return self.classifier(torch.cat(features, dim=1))


class TranscriptDataset(Dataset):
    def __init__(self, bundle: TranscriptBundle) -> None:
        self.bundle = bundle

    def __len__(self) -> int:
        return len(self.bundle)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        return (
            self.bundle.u[index],
            self.bundle.grad_z[index],
            self.bundle.labels[index],
        )


def _attack_validation_ids(
    bundle: TranscriptBundle, num_classes: int, fraction: float, seed: int
) -> set[str]:
    label_by_id: dict[str, int] = {}
    for sample_id, label in zip(bundle.sample_ids, bundle.labels.tolist()):
        label_by_id.setdefault(sample_id, int(label))
    validation: set[str] = set()
    for label in range(num_classes):
        ids = sorted(sample_id for sample_id, value in label_by_id.items() if value == label)
        random.Random(f"{seed}:attack-validation:{label}").shuffle(ids)
        count = max(1, round(len(ids) * fraction))
        validation.update(ids[:count])
    return validation


def _temporal_bundles(
    bundle: TranscriptBundle,
    validation_ids: set[str],
    collection_rounds: tuple[int, ...],
    include_matched: bool,
    seed: int,
) -> dict[str, tuple[TranscriptBundle, TranscriptBundle, str]]:
    is_validation = torch.tensor(
        [sample_id in validation_ids for sample_id in bundle.sample_ids],
        dtype=torch.bool,
    )
    latest = collection_rounds[-1]
    latest_mask = bundle.rounds == latest
    outputs = {
        "single_latest": (
            bundle.select(latest_mask & ~is_validation),
            bundle.select(latest_mask & is_validation),
            "latest_window",
        ),
        "multi_natural": (
            bundle.select(~is_validation),
            bundle.select(is_validation),
            "natural",
        ),
    }
    if not include_matched:
        return outputs

    latest_train_count = int((latest_mask & ~is_validation).sum())
    candidates = (~is_validation).nonzero(as_tuple=False).flatten().tolist()
    by_group: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index in candidates:
        by_group[(int(bundle.rounds[index]), int(bundle.labels[index]))].append(index)
    group_count = len(by_group)
    base = latest_train_count // group_count
    remainder = latest_train_count % group_count
    selected: list[int] = []
    for group_index, (group, indices) in enumerate(sorted(by_group.items())):
        random.Random(f"{seed}:matched:{group[0]}:{group[1]}").shuffle(indices)
        take = base + int(group_index < remainder)
        if len(indices) < take:
            raise ValueError("not enough records to construct matched multi-window budget")
        selected.extend(indices[:take])
    outputs["multi_matched"] = (
        bundle.select(sorted(selected)),
        bundle.select(is_validation),
        "matched_latest_window",
    )
    return outputs


def _train_attack_model(
    train_bundle: TranscriptBundle,
    validation_bundle: TranscriptBundle,
    signal_mode: str,
    num_classes: int,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[OnlineLabelClassifier, list[dict[str, object]]]:
    config = OnlineLabelClassifierConfig(
        u_channels=int(train_bundle.u.shape[1]),
        grad_z_channels=int(train_bundle.grad_z.shape[1]),
        num_classes=num_classes,
        signal_mode=signal_mode,
        signal_spatial_size=args.signal_spatial_size,
        signal_channels=args.signal_channels,
        hidden_channels=args.hidden_channels,
        dropout=args.dropout,
    )
    seed_everything(seed)
    model = OnlineLabelClassifier(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.attack_learning_rate)
    criterion = nn.CrossEntropyLoss()
    train_loader = _loader(
        TranscriptDataset(train_bundle),
        args.attack_batch_size,
        seed + 1,
        shuffle=True,
        num_workers=args.num_workers,
    )
    validation_loader = _loader(
        TranscriptDataset(validation_bundle),
        args.attack_batch_size,
        seed + 2,
        shuffle=False,
        num_workers=args.num_workers,
    )
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    history: list[dict[str, object]] = []
    for epoch in range(1, args.attack_epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = train_samples = 0
        for u, grad_z, labels in train_loader:
            u, grad_z, labels = u.to(device), grad_z.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(u, grad_z)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach()) * len(labels)
            train_correct += int((logits.argmax(1) == labels).sum())
            train_samples += len(labels)
        model.eval()
        validation_loss = 0.0
        validation_correct = validation_samples = 0
        with torch.no_grad():
            for u, grad_z, labels in validation_loader:
                u, grad_z, labels = u.to(device), grad_z.to(device), labels.to(device)
                logits = model(u, grad_z)
                loss = criterion(logits, labels)
                validation_loss += float(loss) * len(labels)
                validation_correct += int((logits.argmax(1) == labels).sum())
                validation_samples += len(labels)
        validation_value = validation_loss / validation_samples
        if validation_value < best_loss:
            best_loss = validation_value
            best_state = copy.deepcopy(model.state_dict())
        history.append(
            {
                "attack_epoch": epoch,
                "train_loss": train_loss / train_samples,
                "train_accuracy": train_correct / train_samples,
                "validation_loss": validation_value,
                "validation_accuracy": validation_correct / validation_samples,
            }
        )
    model.load_state_dict(best_state)
    model.eval()
    return model, history


def _evaluate_attack(
    model: OnlineLabelClassifier,
    bundle: TranscriptBundle,
    catalog: ClassCatalog,
    device: torch.device,
) -> tuple[dict[str, object], list[dict[str, object]], np.ndarray]:
    loader = _loader(
        TranscriptDataset(bundle), 64, 0, shuffle=False, num_workers=0
    )
    predictions: list[int] = []
    labels: list[int] = []
    model.eval()
    with torch.no_grad():
        for u, grad_z, target in loader:
            logits = model(u.to(device), grad_z.to(device))
            predictions.extend(logits.argmax(1).cpu().tolist())
            labels.extend(target.tolist())
    return _metrics_from_predictions(labels, predictions, catalog)


def _metrics_from_predictions(
    labels: Sequence[int], predictions: Sequence[int], catalog: ClassCatalog
) -> tuple[dict[str, object], list[dict[str, object]], np.ndarray]:
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=list(range(catalog.num_classes)), zero_division=0
    )
    summary: dict[str, object] = {
        "accuracy": accuracy_score(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "macro_f1": float(np.mean(f1)),
        "samples": len(labels),
    }
    class_rows = [
        {
            "class_index": index,
            "class_name": catalog.names[index],
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index in range(catalog.num_classes)
    ]
    return summary, class_rows, confusion_matrix(
        labels, predictions, labels=list(range(catalog.num_classes))
    )


def _prototype_features(bundle: TranscriptBundle, signal_mode: str) -> Tensor:
    features: list[Tensor] = []
    if signal_mode in ("u_only", "u_gradient"):
        u = nn.functional.adaptive_avg_pool2d(bundle.u.float(), (4, 4)).flatten(1)
        features.append(nn.functional.normalize(u, dim=1))
    if signal_mode in ("gradient_only", "u_gradient"):
        grad = bundle.grad_z.float()
        norm = grad.flatten(1).norm(dim=1).clamp_min(1e-12)
        direction = grad / norm.view(-1, 1, 1, 1)
        direction = nn.functional.adaptive_avg_pool2d(direction, (4, 4)).flatten(1)
        gradient_features = torch.cat(
            (nn.functional.normalize(direction, dim=1), (torch.log10(norm) / 10.0).unsqueeze(1)),
            dim=1,
        )
        features.append(nn.functional.normalize(gradient_features, dim=1))
    return nn.functional.normalize(torch.cat(features, dim=1), dim=1)


def _evaluate_prototype_attack(
    train_bundle: TranscriptBundle,
    victim_bundle: TranscriptBundle,
    signal_mode: str,
    catalog: ClassCatalog,
) -> tuple[dict[str, object], list[dict[str, object]], np.ndarray]:
    train_features = _prototype_features(train_bundle, signal_mode)
    victim_features = _prototype_features(victim_bundle, signal_mode)
    prototypes: list[Tensor] = []
    for label in range(catalog.num_classes):
        class_features = train_features[train_bundle.labels == label]
        if not len(class_features):
            raise ValueError(f"prototype training data has no class {label}")
        prototypes.append(nn.functional.normalize(class_features.mean(0), dim=0))
    prototype_matrix = torch.stack(prototypes)
    predictions = (victim_features @ prototype_matrix.T).argmax(1).tolist()
    return _metrics_from_predictions(
        victim_bundle.labels.tolist(), predictions, catalog
    )


def _bundle_manifest(
    bundle: TranscriptBundle, catalog: ClassCatalog, *, include_labels: bool
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, sample_id in enumerate(bundle.sample_ids):
        row: dict[str, object] = {
            "record_index": index,
            "sample_id": sample_id,
            "global_round": int(bundle.rounds[index]),
            "server_step": int(bundle.server_steps[index]),
            "u_shape": "x".join(map(str, bundle.u[index].shape)),
            "grad_z_shape": "x".join(map(str, bundle.grad_z[index].shape)),
        }
        if include_labels:
            label = int(bundle.labels[index])
            row["label"] = label
            row["class_name"] = catalog.names[label]
        rows.append(row)
    return rows


def _save_visible_bundle(path: Path, bundle: TranscriptBundle) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "u": bundle.u,
            "grad_z": bundle.grad_z,
            "sample_ids": bundle.sample_ids,
            "rounds": bundle.rounds,
            "server_steps": bundle.server_steps,
        },
        path,
    )


def _build_world(
    initial_f: dict[str, Tensor],
    initial_g: dict[str, Tensor],
    initial_h: dict[str, Tensor],
    cut_config: str,
    catalog: ClassCatalog,
    attacker_dataset: Dataset,
    victim_train: Sequence[Dataset],
    victim_validation: Sequence[Dataset],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[ServerMiddleGModel, list[ClientRuntime], torch.optim.Optimizer]:
    shared_g = ServerMiddleGModel(cut_config).to(device)
    shared_g.load_state_dict(initial_g)
    clients: list[ClientRuntime] = []
    datasets = [attacker_dataset, *victim_train]
    validation = [attacker_dataset, *victim_validation]
    for index, (train_dataset, validation_dataset) in enumerate(zip(datasets, validation)):
        front = ClientFrontFModel(cut_config).to(device)
        front.load_state_dict(initial_f)
        tail = ClientTailHModel(catalog.num_classes).to(device)
        tail.load_state_dict(initial_h)
        model = SplitLearningModel(front, shared_g, tail, cut_config)
        clients.append(
            ClientRuntime(
                name="attacker" if index == 0 else f"victim_{index}",
                model=model,
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                optimizer_f=torch.optim.Adam(
                    front.parameters(), lr=args.client_learning_rate
                ),
                optimizer_h=torch.optim.Adam(
                    tail.parameters(), lr=args.client_learning_rate
                ),
            )
        )
    optimizer_g = torch.optim.Adam(
        shared_g.parameters(), lr=args.server_learning_rate
    )
    return shared_g, clients, optimizer_g


def _run_world(
    mode: str,
    seed: int,
    run_root: Path,
    initial_f: dict[str, Tensor],
    initial_g: dict[str, Tensor],
    initial_h: dict[str, Tensor],
    cut_config: str,
    catalog: ClassCatalog,
    victim_train: Sequence[Dataset],
    victim_validation: Sequence[Dataset],
    victim_holdout: Sequence[Dataset],
    collection_rounds: tuple[int, ...],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[
    TranscriptBundle,
    list[TranscriptBundle],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    attacker_dataset = _attacker_dataset(
        mode,
        args.attacker_data,
        catalog,
        args.image_size,
        seed + 10_000,
    )
    shared_g, clients, optimizer_g = _build_world(
        initial_f,
        initial_g,
        initial_h,
        cut_config,
        catalog,
        attacker_dataset,
        victim_train,
        victim_validation,
        args,
        device,
    )
    attacker_records = TranscriptAccumulator()
    training_rows: list[dict[str, object]] = []
    drift_rows: list[dict[str, object]] = []
    server_step = 0
    for round_index in range(1, args.global_rounds + 1):
        order = list(range(len(clients)))
        random.Random(f"{seed}:client-order:{round_index}").shuffle(order)
        for client_index in order:
            client = clients[client_index]
            data_seed = seed + round_index * 1000 + client_index * 100
            if client_index == 0 and mode == "dynamic":
                data_seed += 37
            seed_everything(data_seed)
            loader = _loader(
                client.train_dataset,
                args.batch_size,
                data_seed,
                shuffle=True,
                num_workers=args.num_workers,
            )
            stats, server_step = _train_client_local_epoch(
                client,
                optimizer_g,
                loader,
                device,
                round_index,
                server_step,
                attacker_records
                if client.name == "attacker" and round_index in collection_rounds
                else None,
            )
            training_rows.append(
                {
                    "seed": seed,
                    "augmentation_mode": mode,
                    "global_round": round_index,
                    "client_order": order.index(client_index),
                    "client": client.name,
                    "loss": stats["loss"],
                    "accuracy": stats["accuracy"],
                    "server_step_after": server_step,
                }
            )
        if round_index in collection_rounds:
            drift_rows.append(
                {
                    "seed": seed,
                    "augmentation_mode": mode,
                    "global_round": round_index,
                    "component": "shared_server_g",
                    "client": "shared_server",
                    "normalized_weight_l2": _normalized_weight_distance(
                        shared_g, initial_g
                    ),
                }
            )
            for client in clients:
                drift_rows.append(
                    {
                        "seed": seed,
                        "augmentation_mode": mode,
                        "global_round": round_index,
                        "component": "client_front_f",
                        "client": client.name,
                        "normalized_weight_l2": _normalized_weight_distance(
                            client.model.f_model, initial_f
                        ),
                    }
                )
        print(
            f"seed={seed} mode={mode} global_round={round_index:02d}/"
            f"{args.global_rounds:02d} server_steps={server_step}",
            flush=True,
        )

    victim_bundles: list[TranscriptBundle] = []
    for victim_index, holdout_dataset in enumerate(victim_holdout, start=1):
        loader = _loader(
            holdout_dataset,
            args.batch_size,
            seed + 80_000 + victim_index,
            shuffle=False,
            num_workers=args.num_workers,
        )
        victim_bundles.append(
            _observe_bundle(
                clients[victim_index].model,
                loader,
                device,
                args.global_rounds,
                server_step,
            )
        )
        task_accuracy = _classification_accuracy(
            clients[victim_index].model, loader, device
        )
        training_rows.append(
            {
                "seed": seed,
                "augmentation_mode": mode,
                "global_round": args.global_rounds,
                "client": f"victim_{victim_index}",
                "stage": "final_holdout",
                "task_accuracy": task_accuracy,
                "server_step_after": server_step,
            }
        )
    return attacker_records.bundle(), victim_bundles, training_rows, drift_rows


def _state_hash(state: dict[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode("utf-8"))
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _aggregate(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    per_seed: dict[tuple[str, str, str, str, int], list[float]] = defaultdict(list)
    balanced_per_seed: dict[tuple[str, str, str, str, int], list[float]] = defaultdict(list)
    f1_per_seed: dict[tuple[str, str, str, str, int], list[float]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["augmentation_mode"]),
            str(row["temporal_mode"]),
            str(row["signal_mode"]),
            str(row["attack_type"]),
            int(row["seed"]),
        )
        per_seed[key].append(float(row["accuracy"]))
        balanced_per_seed[key].append(float(row["balanced_accuracy"]))
        f1_per_seed[key].append(float(row["macro_f1"]))
    grouped: dict[tuple[str, str, str, str], list[tuple[float, float, float]]] = defaultdict(list)
    for key, accuracies in per_seed.items():
        mode, temporal, signal, attack_type, seed = key
        grouped[(mode, temporal, signal, attack_type)].append(
            (
                float(np.mean(accuracies)),
                float(np.mean(balanced_per_seed[key])),
                float(np.mean(f1_per_seed[key])),
            )
        )
    output: list[dict[str, object]] = []
    for (mode, temporal, signal, attack_type), values in sorted(grouped.items()):
        accuracies = [value[0] for value in values]
        balanced_values = [value[1] for value in values]
        f1_values = [value[2] for value in values]
        accuracy_std = statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0
        output.append(
            {
                "augmentation_mode": mode,
                "temporal_mode": temporal,
                "signal_mode": signal,
                "attack_type": attack_type,
                "seed_count": len(values),
                "victims_per_seed": len(
                    {
                        int(row["victim_index"])
                        for row in rows
                        if row["augmentation_mode"] == mode
                        and row["temporal_mode"] == temporal
                        and row["signal_mode"] == signal
                        and row["attack_type"] == attack_type
                    }
                ),
                "mean_accuracy": float(np.mean(accuracies)),
                "std_accuracy_between_seeds": accuracy_std,
                "accuracy_95ci_half_width": (
                    1.96 * accuracy_std / math.sqrt(len(accuracies))
                    if len(accuracies) > 1
                    else 0.0
                ),
                "mean_balanced_accuracy": float(np.mean(balanced_values)),
                "mean_macro_f1": float(np.mean(f1_values)),
                "std_macro_f1_between_seeds": (
                    statistics.stdev(f1_values) if len(f1_values) > 1 else 0.0
                ),
            }
        )
    return output


def _write_readme(output: Path, args: argparse.Namespace, aggregate: Sequence[dict[str, object]]) -> None:
    ranked = sorted(aggregate, key=lambda row: float(row["mean_accuracy"]), reverse=True)
    best_learned = max(
        (row for row in aggregate if row["attack_type"] == "learned_cpsi"),
        key=lambda row: float(row["mean_accuracy"]),
    )
    best_prototype = max(
        (row for row in aggregate if row["attack_type"] == "cosine_class_prototype"),
        key=lambda row: float(row["mean_accuracy"]),
    )
    lines = [
        "# animal5_online_transcript_core",
        "",
        "이 폴더는 전체 모델 Snapshot을 사용하지 않고, 하나의 실제 공유 서버 `g`에서 "
        "수집한 시점별 `u`와 `dL/dz` Transcript로 라벨 추론기를 학습한 결과다.",
        "",
        "## 위협 모델",
        "",
        "- 공격자는 자신의 이미지·라벨과 자신의 정상 통신 Transcript를 안다.",
        "- Victim의 `u`와 `dL/dz`는 명시적인 relay/logging 노출 경계에서만 관측한다.",
        "- 공격기는 Victim 이미지·라벨·로컬 가중치와 서버 `g` 가중치에 접근하지 않는다.",
        "- 연구용 evaluator만 별도 private manifest의 Victim 라벨을 읽는다.",
        "",
        "## 기준 설정",
        "",
        f"- Seeds: `{', '.join(map(str, args.seeds))}`",
        f"- Victim clients: `{args.num_victims}`",
        f"- Global rounds: `{args.global_rounds}`",
        f"- Transcript collection rounds: `{', '.join(map(str, args.collection_rounds))}`",
        f"- Client/Server LR: `{args.client_learning_rate}` / `{args.server_learning_rate}`",
        f"- Split/Attack batch size: `{args.batch_size}` / `{args.attack_batch_size}`",
        f"- Attack epochs: `{args.attack_epochs}`",
        "",
        "## 핵심 결과",
        "",
        f"- 학습형 `C_psi` 최고 조건: `{best_learned['augmentation_mode']} + "
        f"{best_learned['temporal_mode']} + {best_learned['signal_mode']}` = "
        f"**{float(best_learned['mean_accuracy']):.2%}** "
        f"(95% CI ±{float(best_learned['accuracy_95ci_half_width']):.2%}p).",
        f"- 거리 기반 최고 조건: `{best_prototype['augmentation_mode']} + "
        f"{best_prototype['temporal_mode']} + {best_prototype['signal_mode']}` = "
        f"**{float(best_prototype['mean_accuracy']):.2%}** "
        f"(95% CI ±{float(best_prototype['accuracy_95ci_half_width']):.2%}p).",
        "- 5-class 무작위 추측 기준은 20%다.",
        "- `multi_natural`은 640개, `single_latest`와 `multi_matched`는 "
        "각각 160개의 공격 학습 레코드를 사용한다. 따라서 natural 결과는 "
        "시점 다양성과 데이터 수 증가가 함께 반영되며, 순수 Window 효과는 "
        "matched 결과와 함께 해석해야 한다.",
        "- 증강별 실험은 동일 초기 가중치에서 시작한 독립적인 shared-server "
        "학습 world다. 공격자 입력 증강이 shared `g`의 학습 경로에도 영향을 "
        "주므로 증강 효과는 공격기 입력 변화만의 효과가 아니다.",
        "- 논문 본문에 사용하기 전 `split_training_history.csv`의 Victim 정상 task "
        "정확도를 함께 보고하고, 정상 모델 성능이 충분한 추가 설정에서도 공격 "
        "결론이 유지되는지 재검증해야 한다.",
        "",
        "## 집계 결과",
        "",
        "| 순위 | 증강 | Window | 입력 | 공격기 | 평균 정확도 | 95% CI | Macro-F1 |",
        "|---:|---|---|---|---|---:|---:|---:|",
    ]
    for index, row in enumerate(ranked, start=1):
        lines.append(
            f"| {index} | {row['augmentation_mode']} | {row['temporal_mode']} | "
            f"{row['signal_mode']} | {row['attack_type']} | "
            f"{float(row['mean_accuracy']):.2%} | "
            f"±{float(row['accuracy_95ci_half_width']):.2%} | "
            f"{float(row['mean_macro_f1']):.2%} |"
        )
    lines.extend(
        [
            "",
            "## 주요 파일",
            "",
            "- `aggregate_summary.csv`: seed와 Victim을 계층적으로 집계한 핵심 결과",
            "- `label_inference_summary.csv`: seed·Victim·조건별 결과",
            "- `per_class_metrics.csv`: 클래스별 precision/recall/F1",
            "- `condition_matrix.csv`: 공격 조건과 transcript budget",
            "- `split_training_history.csv`: 실제 shared-server 학습 기록",
            "- `drift_metrics.csv`: 초기 가중치 대비 직접 측정한 normalized L2 drift",
            "- `threat_model_audit.json`: 공격기 접근 가능/불가능 정보",
            "",
            "`multi_natural`은 여러 Window의 모든 레코드를 사용하고, "
            "`multi_matched`는 최신 단일 Window와 같은 수로 맞춘 통제 조건이다.",
        ]
    )
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    collection_rounds = _validate_args(args)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = _device(args.device)
    catalog = ClassCatalog.discover(args.pretrain_data)
    if not (
        catalog.names == ClassCatalog.discover(args.attacker_data).names
        == ClassCatalog.discover(args.victim_data).names
    ):
        raise ValueError("pretrain, attacker, and victim class mappings must match")
    checkpoint = torch.load(
        args.pretrained_autoencoder, map_location=device, weights_only=False
    )
    cut_config = str(checkpoint["cut_config"])

    all_results: list[dict[str, object]] = []
    all_class_rows: list[dict[str, object]] = []
    all_training_rows: list[dict[str, object]] = []
    all_drift_rows: list[dict[str, object]] = []
    condition_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []

    for seed in args.seeds:
        seed_everything(seed)
        encoder = ClientFrontFModel(cut_config).to(device)
        encoder.load_state_dict(checkpoint["encoder"])
        base_model = SplitLearningModel(
            encoder,
            ServerMiddleGModel(cut_config).to(device),
            ClientTailHModel(catalog.num_classes).to(device),
            cut_config,
        )
        pretrain_loader = make_loader(
            args.pretrain_data,
            "train",
            args.image_size,
            args.batch_size,
            args.num_workers,
            shuffle=True,
            augment=True,
            class_names=catalog.names,
        )
        warmup_history = _warmup_classifier(
            base_model,
            pretrain_loader,
            args.warmup_epochs,
            args.client_learning_rate,
            device,
        )
        for row in warmup_history:
            all_training_rows.append({"seed": seed, "stage": "warmup", **row})
        initial_f = copy.deepcopy(base_model.f_model.state_dict())
        initial_g = copy.deepcopy(base_model.g_model.state_dict())
        initial_h = copy.deepcopy(base_model.h_model.state_dict())

        victim_train, train_split_rows = _partition_victim_dataset(
            args.victim_data,
            "train",
            catalog,
            args.image_size,
            args.num_victims,
            2026,
            augment=True,
        )
        victim_validation, validation_split_rows = _partition_victim_dataset(
            args.victim_data,
            "val",
            catalog,
            args.image_size,
            args.num_victims,
            2026,
            augment=False,
        )
        victim_holdout, holdout_split_rows = _partition_victim_dataset(
            args.victim_data,
            "new_holdout",
            catalog,
            args.image_size,
            args.num_victims,
            2026,
            augment=False,
        )
        if not split_rows:
            split_rows.extend(train_split_rows)
            split_rows.extend(validation_split_rows)
            split_rows.extend(holdout_split_rows)

        for mode_index, augmentation_mode in enumerate(args.augmentation_modes):
            run_root = output / f"seed_{seed}" / augmentation_mode
            run_root.mkdir(parents=True, exist_ok=True)
            attacker_bundle, victim_bundles, training_rows, drift_rows = _run_world(
                augmentation_mode,
                seed,
                run_root,
                initial_f,
                initial_g,
                initial_h,
                cut_config,
                catalog,
                victim_train,
                victim_validation,
                victim_holdout,
                collection_rounds,
                args,
                device,
            )
            all_training_rows.extend(training_rows)
            all_drift_rows.extend(drift_rows)
            validation_ids = _attack_validation_ids(
                attacker_bundle,
                catalog.num_classes,
                args.attack_validation_fraction,
                seed,
            )
            temporal = _temporal_bundles(
                attacker_bundle,
                validation_ids,
                collection_rounds,
                args.matched_budget,
                seed,
            )

            _write_csv(
                run_root / "transcripts" / "attacker_visible_manifest.csv",
                _bundle_manifest(attacker_bundle, catalog, include_labels=True),
            )
            if args.save_transcript_tensors:
                path = run_root / "transcripts" / "attacker_visible_tensors.pt"
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "u": attacker_bundle.u,
                        "grad_z": attacker_bundle.grad_z,
                        "labels": attacker_bundle.labels,
                        "sample_ids": attacker_bundle.sample_ids,
                        "rounds": attacker_bundle.rounds,
                        "server_steps": attacker_bundle.server_steps,
                    },
                    path,
                )
            for victim_index, victim_bundle in enumerate(victim_bundles, start=1):
                visible_root = run_root / "transcripts" / f"victim_{victim_index}"
                _write_csv(
                    visible_root / "attacker_visible_manifest.csv",
                    _bundle_manifest(victim_bundle, catalog, include_labels=False),
                )
                private_rows = [
                    {
                        "record_index": index,
                        "sample_id": sample_id,
                        "true_label": int(victim_bundle.labels[index]),
                        "class_name": catalog.names[int(victim_bundle.labels[index])],
                    }
                    for index, sample_id in enumerate(victim_bundle.sample_ids)
                ]
                _write_csv(
                    visible_root / "evaluator_private_labels.csv", private_rows
                )
                if args.save_transcript_tensors:
                    _save_visible_bundle(
                        visible_root / "attacker_visible_tensors.pt", victim_bundle
                    )

            for temporal_index, (temporal_mode, values) in enumerate(temporal.items()):
                train_bundle, validation_bundle, budget = values
                condition_rows.append(
                    {
                        "seed": seed,
                        "augmentation_mode": augmentation_mode,
                        "temporal_mode": temporal_mode,
                        "budget": budget,
                        "collection_rounds": " ".join(map(str, collection_rounds)),
                        "training_records": len(train_bundle),
                        "validation_records": len(validation_bundle),
                        "server_snapshot_access": False,
                        "victim_label_access": False,
                    }
                )
                for signal_index, signal_mode in enumerate(SIGNAL_MODES):
                    attack_seed = seed + 100_000 + temporal_index * 100 + signal_index
                    model, history = _train_attack_model(
                        train_bundle,
                        validation_bundle,
                        signal_mode,
                        catalog.num_classes,
                        args,
                        device,
                        attack_seed,
                    )
                    condition_root = (
                        run_root / "conditions" / temporal_mode / signal_mode
                    )
                    condition_root.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "config": model.config.to_dict(),
                            "access": ["attacker_u", "attacker_dL_dz", "attacker_y"],
                        },
                        condition_root / "attack_checkpoint.pt",
                    )
                    _write_csv(
                        condition_root / "attack_training_history.csv",
                        [
                            {
                                "seed": seed,
                                "augmentation_mode": augmentation_mode,
                                "temporal_mode": temporal_mode,
                                "signal_mode": signal_mode,
                                **row,
                            }
                            for row in history
                        ],
                    )
                    for victim_index, victim_bundle in enumerate(victim_bundles, start=1):
                        common = {
                            "seed": seed,
                            "augmentation_mode": augmentation_mode,
                            "temporal_mode": temporal_mode,
                            "signal_mode": signal_mode,
                            "budget": budget,
                            "training_records": len(train_bundle),
                            "victim_index": victim_index,
                            "victim_round": args.global_rounds,
                        }
                        for attack_type in ("learned_cpsi", "cosine_class_prototype"):
                            if attack_type == "learned_cpsi":
                                summary, class_metrics, matrix = _evaluate_attack(
                                    model, victim_bundle, catalog, device
                                )
                            else:
                                summary, class_metrics, matrix = _evaluate_prototype_attack(
                                    train_bundle, victim_bundle, signal_mode, catalog
                                )
                            typed_common = {**common, "attack_type": attack_type}
                            all_results.append({**typed_common, **summary})
                            all_class_rows.extend(
                                {**typed_common, **row} for row in class_metrics
                            )
                            np.savetxt(
                                condition_root
                                / f"victim_{victim_index}_{attack_type}_confusion.csv",
                                matrix,
                                delimiter=",",
                                fmt="%d",
                            )
                    print(
                        f"seed={seed} mode={augmentation_mode} temporal={temporal_mode} "
                        f"signal={signal_mode} attack complete",
                        flush=True,
                    )

        seed_root = output / f"seed_{seed}"
        (seed_root / "initial_state_hashes.json").write_text(
            json.dumps(
                {
                    "client_front_f": _state_hash(initial_f),
                    "shared_server_g": _state_hash(initial_g),
                    "client_tail_h": _state_hash(initial_h),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    aggregate = _aggregate(all_results)
    _write_csv(output / "label_inference_summary.csv", all_results)
    _write_csv(output / "aggregate_summary.csv", aggregate)
    _write_csv(output / "per_class_metrics.csv", all_class_rows)
    _write_csv(output / "condition_matrix.csv", condition_rows)
    _write_csv(output / "split_training_history.csv", all_training_rows)
    _write_csv(output / "drift_metrics.csv", all_drift_rows)
    _write_csv(output / "victim_partition_manifest.csv", split_rows)
    training_hashes = _file_hashes(
        [
            Path(args.pretrain_data) / "train",
            Path(args.pretrain_data) / "val",
            Path(args.attacker_data) / "train",
            Path(args.attacker_data) / "val",
            Path(args.victim_data) / "train",
            Path(args.victim_data) / "val",
        ]
    )
    holdout_hashes = _file_hashes([Path(args.victim_data) / "new_holdout"])
    overlap = training_hashes & holdout_hashes
    if overlap:
        raise RuntimeError(
            f"data leakage audit found {len(overlap)} train/holdout hash overlaps"
        )
    (output / "data_leakage_audit.json").write_text(
        json.dumps(
            {
                "training_and_validation_images": len(training_hashes),
                "victim_holdout_images": len(holdout_hashes),
                "sha256_overlap": 0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    threat_model = {
        "attacker_role": "malicious but protocol-conforming client",
        "shared_server_instances_per_seed_and_augmentation_condition": 1,
        "independent_worlds_for_augmentation_ablation": True,
        "server_model_snapshot_access": False,
        "server_weight_access": False,
        "victim_front_or_tail_weight_access": False,
        "attacker_training_access": ["own_u", "own_dL_dz", "own_label"],
        "victim_attack_access": ["victim_u", "victim_dL_dz", "timestamp"],
        "victim_visibility_assumption": (
            "decrypted tensors exposed at an explicitly stated shared relay or "
            "orchestration logging boundary; ordinary client membership alone is insufficient"
        ),
        "victim_label_storage": "evaluator-private manifest only",
        "model_snapshot_replay": False,
        "attack_training_uses_cached_transcripts_only": True,
    }
    (output / "threat_model_audit.json").write_text(
        json.dumps(threat_model, indent=2), encoding="utf-8"
    )
    (output / "run_config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "resolved_output": str(output),
                "resolved_device": str(device),
                "class_names": list(catalog.names),
                "collection_rounds": list(collection_rounds),
                "signal_modes": list(SIGNAL_MODES),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_readme(output, args, aggregate)
    print(f"Online transcript results written to {output}", flush=True)


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()


__all__ = [
    "AUGMENTATION_MODES",
    "OnlineLabelClassifier",
    "OnlineLabelClassifierConfig",
    "SIGNAL_MODES",
    "TEMPORAL_MODES",
    "TranscriptAccumulator",
    "TranscriptBundle",
    "_temporal_bundles",
    "build_parser",
    "run",
]
