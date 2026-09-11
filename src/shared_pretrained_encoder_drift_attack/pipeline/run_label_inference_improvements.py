from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from ...decoder.data.image_scaling import IMAGENET_MEAN, IMAGENET_STD
from ...shared.data.class_catalog import ClassCatalog
from ...shared.data.image_dataset import (
    ImageFolderWithID,
    make_loader,
    validate_class_mapping,
)
from ...shared.reproducibility.random_seed import seed_everything
from ...split_learning.architecture.split_learning_model import SplitLearningModel
from ...split_learning.f_model.client_front_f_model import ClientFrontFModel
from ...split_learning.g_model.server_middle_g_model import ServerMiddleGModel
from ...split_learning.gradient_flow.gradient_exchange import (
    observe_frozen_gradient_exchange,
)
from ...split_learning.h_model.client_tail_h_model import ClientTailHModel
from ..models import (
    DriftRobustLabelClassifier,
    DriftRobustLabelClassifierConfig,
)
from .run_joint_transcript_attack import (
    _balanced_holdout_loader,
    _classification_accuracy,
    _device,
    _file_hashes,
    _fine_tune_epoch,
    _save_confusion_plot,
    _warmup_classifier,
    _write_csv,
)


VARIANT_BASELINE = "baseline_gradient_direction"
VARIANT_MAGNITUDE = "gradient_direction_norm"
VARIANT_MULTI = "multi_snapshot_gradient_norm"
VARIANT_AUGMENTED = "augmented_multi_snapshot_gradient_norm"
VARIANT_GATED = "gated_u_gradient"
VARIANTS = (
    VARIANT_BASELINE,
    VARIANT_MAGNITUDE,
    VARIANT_MULTI,
    VARIANT_AUGMENTED,
    VARIANT_GATED,
)

AUGMENTATION_COMPONENTS = ("crop", "flip", "color")
DYNAMIC_NONE = "multi_none"
DYNAMIC_SHARED = "multi_dyn_shared"
DYNAMIC_INDEPENDENT = "multi_dyn_independent"
DYNAMIC_VARIANTS = (DYNAMIC_NONE, DYNAMIC_SHARED, DYNAMIC_INDEPENDENT)

LabelModel = DriftRobustLabelClassifier


@dataclass(frozen=True)
class TrainingGroup:
    name: str
    variants: tuple[str, ...]
    snapshot_epochs: tuple[int, ...]
    augment: bool


@dataclass(frozen=True)
class FactorialCondition:
    """One controlled snapshot x augmentation ablation condition."""

    variant: str
    snapshot_mode: str
    snapshot_epochs: tuple[int, ...]
    augmentations: tuple[str, ...]


@dataclass(frozen=True)
class DynamicCondition:
    """One controlled full-augmentation scheduling condition."""

    variant: str
    augmentation_schedule: str


DYNAMIC_CONDITIONS = (
    DynamicCondition(DYNAMIC_NONE, "none"),
    DynamicCondition(DYNAMIC_SHARED, "dynamic_shared"),
    DynamicCondition(DYNAMIC_INDEPENDENT, "dynamic_independent"),
)


def augmentation_subsets() -> tuple[tuple[str, ...], ...]:
    """Return the complete power set in stable 0/1/2/3-factor order."""

    return tuple(
        subset
        for size in range(len(AUGMENTATION_COMPONENTS) + 1)
        for subset in combinations(AUGMENTATION_COMPONENTS, size)
    )


def _augmentation_key(augmentations: tuple[str, ...]) -> str:
    return "none" if not augmentations else "_".join(augmentations)


def factorial_conditions(
    snapshot_epochs: tuple[int, ...],
) -> tuple[FactorialCondition, ...]:
    """Build 2 snapshot modes x all 8 augmentation subsets = 16 conditions."""

    conditions: list[FactorialCondition] = []
    for augmentations in augmentation_subsets():
        augmentation_key = _augmentation_key(augmentations)
        for snapshot_mode, epochs in (
            ("single_snapshot", (snapshot_epochs[-1],)),
            ("multi_snapshot", snapshot_epochs),
        ):
            conditions.append(
                FactorialCondition(
                    variant=f"{snapshot_mode}__aug_{augmentation_key}",
                    snapshot_mode=snapshot_mode,
                    snapshot_epochs=epochs,
                    augmentations=augmentations,
                )
            )
    return tuple(conditions)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare gradient magnitude, multi-snapshot training, transcript "
            "augmentation, and gated u+gradient fusion for label inference."
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
            "animal5_label_improvements"
        ),
    )
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--attacker-finetune-epochs", type=int, default=20)
    parser.add_argument(
        "--attacker-snapshot-epochs",
        nargs="+",
        type=int,
        default=[0, 1, 5, 10, 20],
    )
    parser.add_argument("--victim-finetune-epochs", type=int, default=20)
    parser.add_argument(
        "--capture-epochs", nargs="+", type=int, default=[0, 1, 5, 10, 20]
    )
    parser.add_argument("--attack-epochs", type=int, default=30)
    parser.add_argument("--holdout-per-class", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--attack-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--victim-learning-rate",
        type=float,
        default=None,
        help="Victim-only learning rate; defaults to --learning-rate.",
    )
    parser.add_argument(
        "--victim-observation-batch-size",
        type=int,
        default=None,
        help="Holdout transcript batch size; defaults to --batch-size.",
    )
    parser.add_argument("--attack-learning-rate", type=float, default=1e-3)
    parser.add_argument("--classifier-spatial-size", type=int, default=8)
    parser.add_argument("--signal-channels", type=int, default=64)
    parser.add_argument("--classifier-hidden-channels", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    experiment = parser.add_mutually_exclusive_group()
    experiment.add_argument(
        "--factorial-augmentation-ablation",
        action="store_true",
        help=(
            "Run all 16 combinations of single/multi snapshot training and the "
            "crop/flip/color augmentation power set. Each source image receives "
            "one fixed augmented view that is reused across snapshots and attack epochs."
        ),
    )
    experiment.add_argument(
        "--controlled-dynamic-ablation",
        action="store_true",
        help=(
            "Compare multi-snapshot no augmentation, dynamic full augmentation "
            "shared across snapshots, and dynamic full augmentation independently "
            "sampled per snapshot under matched training conditions."
        ),
    )
    return parser


def _attack_loader(
    data_root: str | Path,
    catalog: ClassCatalog,
    args: argparse.Namespace,
    *,
    augment: bool,
    shuffle: bool,
) -> DataLoader:
    if not augment:
        return make_loader(
            data_root,
            "train" if shuffle else "val",
            args.image_size,
            args.attack_batch_size,
            args.num_workers,
            shuffle=shuffle,
            augment=False,
            class_names=catalog.names,
        )
    transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                args.image_size, scale=(0.72, 1.0), ratio=(0.88, 1.12)
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.18, contrast=0.18, saturation=0.18, hue=0.03
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    dataset = ImageFolderWithID(
        Path(data_root) / "train", transform=transform, allow_empty=True
    )
    validate_class_mapping(dataset, catalog.names)
    return DataLoader(
        dataset,
        batch_size=args.attack_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )


def _factorial_attack_loader(
    data_root: str | Path,
    catalog: ClassCatalog,
    args: argparse.Namespace,
    augmentations: tuple[str, ...],
) -> DataLoader:
    """Create an ordered loader for one controlled augmentation subset."""

    unknown = set(augmentations) - set(AUGMENTATION_COMPONENTS)
    if unknown:
        raise ValueError(f"unknown augmentation components: {sorted(unknown)}")
    operations: list[object] = []
    if "crop" in augmentations:
        operations.append(
            transforms.RandomResizedCrop(
                args.image_size, scale=(0.72, 1.0), ratio=(0.88, 1.12)
            )
        )
    else:
        operations.append(transforms.Resize((args.image_size, args.image_size)))
    if "flip" in augmentations:
        operations.append(transforms.RandomHorizontalFlip())
    if "color" in augmentations:
        operations.append(
            transforms.ColorJitter(
                brightness=0.18, contrast=0.18, saturation=0.18, hue=0.03
            )
        )
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    dataset = ImageFolderWithID(
        Path(data_root) / "train",
        transform=transforms.Compose(operations),
        allow_empty=True,
    )
    validate_class_mapping(dataset, catalog.names)
    return DataLoader(
        dataset,
        batch_size=args.attack_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )


def _materialize_fixed_views(
    data_root: str | Path,
    catalog: ClassCatalog,
    args: argparse.Namespace,
    augmentations: tuple[str, ...],
    seed: int,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor, tuple[str, ...]]], list[dict[str, object]]]:
    """Generate each augmented training image once and cache it on CPU.

    The cached tensors are reused by the single- and multi-snapshot conditions,
    by every provider snapshot, and by every attack epoch. This prevents a
    different random crop/color draw from being confounded with snapshot age.
    """

    loader = _factorial_attack_loader(data_root, catalog, args, augmentations)
    batches: list[tuple[torch.Tensor, torch.Tensor, tuple[str, ...]]] = []
    manifest: list[dict[str, object]] = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for images, labels, sample_ids in loader:
            cpu_images = images.detach().cpu().contiguous()
            cpu_labels = labels.detach().cpu()
            ids = tuple(str(sample_id) for sample_id in sample_ids)
            batches.append((cpu_images, cpu_labels, ids))
            for image, label, sample_id in zip(cpu_images, cpu_labels, ids):
                digest = hashlib.sha256(image.numpy().tobytes()).hexdigest()
                manifest.append(
                    {
                        "sample_id": sample_id,
                        "label": int(label),
                        "augmentations": _augmentation_key(augmentations),
                        "view_seed": seed,
                        "tensor_sha256": digest,
                    }
                )
    return batches, manifest


def _build_models(
    sample_u: torch.Tensor,
    sample_grad_z: torch.Tensor,
    catalog: ClassCatalog,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, LabelModel]:
    u_channels = int(sample_u.shape[1])
    grad_z_channels = int(sample_grad_z.shape[1])
    models: dict[str, LabelModel] = {
        VARIANT_BASELINE: DriftRobustLabelClassifier(
            DriftRobustLabelClassifierConfig(
                u_channels=u_channels,
                grad_z_channels=grad_z_channels,
                num_classes=catalog.num_classes,
                use_u=False,
                use_gradient_norm=False,
                signal_spatial_size=args.classifier_spatial_size,
                signal_channels=args.signal_channels,
                hidden_channels=args.classifier_hidden_channels,
            )
        ).to(device)
    }
    for variant in (VARIANT_MAGNITUDE, VARIANT_MULTI, VARIANT_AUGMENTED):
        models[variant] = DriftRobustLabelClassifier(
            DriftRobustLabelClassifierConfig(
                u_channels=u_channels,
                grad_z_channels=grad_z_channels,
                num_classes=catalog.num_classes,
                use_u=False,
                use_gradient_norm=True,
                signal_spatial_size=args.classifier_spatial_size,
                signal_channels=args.signal_channels,
                hidden_channels=args.classifier_hidden_channels,
            )
        ).to(device)
    models[VARIANT_GATED] = DriftRobustLabelClassifier(
        DriftRobustLabelClassifierConfig(
            u_channels=u_channels,
            grad_z_channels=grad_z_channels,
            num_classes=catalog.num_classes,
            use_u=True,
            use_gradient_norm=True,
            signal_spatial_size=args.classifier_spatial_size,
            signal_channels=args.signal_channels,
            hidden_channels=args.classifier_hidden_channels,
        )
    ).to(device)
    return models


def _build_factorial_models(
    sample_u: torch.Tensor,
    sample_grad_z: torch.Tensor,
    catalog: ClassCatalog,
    args: argparse.Namespace,
    conditions: tuple[FactorialCondition, ...],
    device: torch.device,
) -> dict[str, LabelModel]:
    """Build identically initialized classifiers for a controlled ablation."""

    base = DriftRobustLabelClassifier(
        DriftRobustLabelClassifierConfig(
            u_channels=int(sample_u.shape[1]),
            grad_z_channels=int(sample_grad_z.shape[1]),
            num_classes=catalog.num_classes,
            use_u=False,
            use_gradient_norm=True,
            signal_spatial_size=args.classifier_spatial_size,
            signal_channels=args.signal_channels,
            hidden_channels=args.classifier_hidden_channels,
        )
    ).to(device)
    return {
        condition.variant: copy.deepcopy(base).to(device) for condition in conditions
    }


def _build_dynamic_models(
    sample_u: torch.Tensor,
    sample_grad_z: torch.Tensor,
    catalog: ClassCatalog,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, LabelModel]:
    """Build the three scheduling classifiers with identical initial weights."""

    base = DriftRobustLabelClassifier(
        DriftRobustLabelClassifierConfig(
            u_channels=int(sample_u.shape[1]),
            grad_z_channels=int(sample_grad_z.shape[1]),
            num_classes=catalog.num_classes,
            use_u=False,
            use_gradient_norm=True,
            signal_spatial_size=args.classifier_spatial_size,
            signal_channels=args.signal_channels,
            hidden_channels=args.classifier_hidden_channels,
        )
    ).to(device)
    return {variant: copy.deepcopy(base).to(device) for variant in DYNAMIC_VARIANTS}


def _forward(
    model: LabelModel, u: torch.Tensor, grad_z: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor | None]:
    return model.forward_with_gate(u if model.config.use_u else None, grad_z)


def _train_group(
    group: TrainingGroup,
    models: dict[str, LabelModel],
    snapshots: dict[int, SplitLearningModel],
    attacker_data: str | Path,
    catalog: ClassCatalog,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, object]]:
    train_loader = _attack_loader(
        attacker_data, catalog, args, augment=group.augment, shuffle=True
    )
    validation_loader = make_loader(
        attacker_data,
        "val",
        args.image_size,
        args.attack_batch_size,
        args.num_workers,
        shuffle=False,
        augment=False,
        class_names=catalog.names,
    )
    optimizers = {
        variant: torch.optim.AdamW(
            models[variant].parameters(), lr=args.attack_learning_rate
        )
        for variant in group.variants
    }
    best = {
        variant: (float("inf"), copy.deepcopy(models[variant].state_dict()))
        for variant in group.variants
    }
    criterion = nn.CrossEntropyLoss()
    history: list[dict[str, object]] = []
    providers = [snapshots[epoch] for epoch in group.snapshot_epochs]

    for attack_epoch in range(1, args.attack_epochs + 1):
        for variant in group.variants:
            models[variant].train()
        order = torch.randperm(len(providers)).tolist()
        for provider_index in order:
            provider = providers[provider_index]
            provider.eval()
            for images, labels, _ in train_loader:
                images, labels = images.to(device), labels.to(device)
                exchange = observe_frozen_gradient_exchange(
                    provider, images, labels, criterion
                )
                for variant in group.variants:
                    optimizers[variant].zero_grad(set_to_none=True)
                    logits, _ = _forward(
                        models[variant],
                        exchange.server_output_u,
                        exchange.grad_g_to_f,
                    )
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizers[variant].step()

        validation = {
            variant: {"loss": 0.0, "correct": 0, "gates": []}
            for variant in group.variants
        }
        samples = 0
        for variant in group.variants:
            models[variant].eval()
        for provider in providers:
            provider.eval()
            for images, labels, _ in validation_loader:
                images, labels = images.to(device), labels.to(device)
                exchange = observe_frozen_gradient_exchange(
                    provider, images, labels, criterion
                )
                with torch.no_grad():
                    for variant in group.variants:
                        logits, gate = _forward(
                            models[variant],
                            exchange.server_output_u,
                            exchange.grad_g_to_f,
                        )
                        validation[variant]["loss"] += float(
                            criterion(logits, labels)
                        ) * len(images)
                        validation[variant]["correct"] += int(
                            (logits.argmax(1) == labels).sum()
                        )
                        if gate is not None:
                            validation[variant]["gates"].extend(
                                gate.flatten().cpu().tolist()
                            )
                samples += len(images)

        statuses: list[str] = []
        for variant in group.variants:
            value = float(validation[variant]["loss"]) / samples
            accuracy = int(validation[variant]["correct"]) / samples
            gates = list(validation[variant]["gates"])
            row = {
                "group": group.name,
                "attack_epoch": attack_epoch,
                "variant": variant,
                "snapshots": " ".join(map(str, group.snapshot_epochs)),
                "augmentation": group.augment,
                "training_transcripts_per_epoch": len(train_loader.dataset)
                * len(providers),
                "validation_loss": value,
                "validation_accuracy": accuracy,
                "mean_gate": float(np.mean(gates)) if gates else "",
            }
            history.append(row)
            if value < best[variant][0]:
                best[variant] = (value, copy.deepcopy(models[variant].state_dict()))
            statuses.append(f"{variant}={accuracy:.2%}")
        print(
            f"{group.name} {attack_epoch:03d}/{args.attack_epochs:03d}: "
            + ", ".join(statuses),
            flush=True,
        )

    for variant in group.variants:
        models[variant].load_state_dict(best[variant][1])
        models[variant].eval()
    return history


def _train_factorial_pair(
    conditions: tuple[FactorialCondition, FactorialCondition],
    models: dict[str, LabelModel],
    snapshots: dict[int, SplitLearningModel],
    fixed_batches: list[tuple[torch.Tensor, torch.Tensor, tuple[str, ...]]],
    validation_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, object]]:
    """Train the single/multi pair on exactly the same fixed image views."""

    optimizers = {
        condition.variant: torch.optim.AdamW(
            models[condition.variant].parameters(), lr=args.attack_learning_rate
        )
        for condition in conditions
    }
    best = {
        condition.variant: (
            float("inf"),
            copy.deepcopy(models[condition.variant].state_dict()),
        )
        for condition in conditions
    }
    criterion = nn.CrossEntropyLoss()
    history: list[dict[str, object]] = []
    source_samples = sum(len(labels) for _, labels, _ in fixed_batches)
    provider_passes = max(len(condition.snapshot_epochs) for condition in conditions)

    # Give every augmentation subset the same dropout/random initialization
    # stream. Augmented tensors themselves are already fixed and cached.
    seed_everything(args.seed + 10_000)
    for attack_epoch in range(1, args.attack_epochs + 1):
        statuses: list[str] = []
        for condition in conditions:
            variant = condition.variant
            model = models[variant]
            model.train()
            validation_providers = [
                snapshots[epoch] for epoch in condition.snapshot_epochs
            ]
            training_providers = [
                validation_providers[index % len(validation_providers)]
                for index in range(provider_passes)
            ]
            for provider in training_providers:
                provider.eval()
                for images, labels, _ in fixed_batches:
                    images = images.to(device)
                    labels = labels.to(device)
                    exchange = observe_frozen_gradient_exchange(
                        provider, images, labels, criterion
                    )
                    optimizers[variant].zero_grad(set_to_none=True)
                    logits, _ = _forward(
                        model,
                        exchange.server_output_u,
                        exchange.grad_g_to_f,
                    )
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizers[variant].step()

            model.eval()
            validation_loss = 0.0
            validation_correct = 0
            validation_samples = 0
            for provider in validation_providers:
                provider.eval()
                for images, labels, _ in validation_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    exchange = observe_frozen_gradient_exchange(
                        provider, images, labels, criterion
                    )
                    with torch.no_grad():
                        logits, _ = _forward(
                            model,
                            exchange.server_output_u,
                            exchange.grad_g_to_f,
                        )
                        validation_loss += float(criterion(logits, labels)) * len(
                            images
                        )
                        validation_correct += int(
                            (logits.argmax(1) == labels).sum()
                        )
                    validation_samples += len(images)

            value = validation_loss / validation_samples
            accuracy = validation_correct / validation_samples
            history.append(
                {
                    "group": _augmentation_key(condition.augmentations),
                    "attack_epoch": attack_epoch,
                    "variant": variant,
                    "snapshot_mode": condition.snapshot_mode,
                    "snapshots": " ".join(map(str, condition.snapshot_epochs)),
                    "augmentation_count": len(condition.augmentations),
                    "augmentations": _augmentation_key(condition.augmentations),
                    "crop": "crop" in condition.augmentations,
                    "flip": "flip" in condition.augmentations,
                    "color": "color" in condition.augmentations,
                    "view_policy": "fixed_per_source_shared_across_snapshots_and_attack_epochs",
                    "unique_source_images": source_samples,
                    "unique_snapshot_count": len(condition.snapshot_epochs),
                    "provider_passes_per_epoch": provider_passes,
                    "training_transcripts_per_epoch": source_samples * provider_passes,
                    "matched_training_exposure": True,
                    "validation_loss": value,
                    "validation_accuracy": accuracy,
                }
            )
            if value < best[variant][0]:
                best[variant] = (value, copy.deepcopy(model.state_dict()))
            statuses.append(f"{condition.snapshot_mode}={accuracy:.2%}")
        print(
            f"Factorial {_augmentation_key(conditions[0].augmentations)} "
            f"{attack_epoch:03d}/{args.attack_epochs:03d}: " + ", ".join(statuses),
            flush=True,
        )

    for condition in conditions:
        models[condition.variant].load_state_dict(best[condition.variant][1])
        models[condition.variant].eval()
    return history


def _train_controlled_dynamic(
    models: dict[str, LabelModel],
    snapshots: dict[int, SplitLearningModel],
    snapshot_epochs: tuple[int, ...],
    clean_batches: list[tuple[torch.Tensor, torch.Tensor, tuple[str, ...]]],
    validation_loader: DataLoader,
    attacker_data: str | Path,
    catalog: ClassCatalog,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Train three augmentation schedules with every non-schedule factor matched."""

    optimizers = {
        variant: torch.optim.AdamW(
            models[variant].parameters(), lr=args.attack_learning_rate
        )
        for variant in DYNAMIC_VARIANTS
    }
    best = {
        variant: (float("inf"), copy.deepcopy(models[variant].state_dict()))
        for variant in DYNAMIC_VARIANTS
    }
    criterion = nn.CrossEntropyLoss()
    history: list[dict[str, object]] = []
    view_schedule: list[dict[str, object]] = []
    source_samples = sum(len(labels) for _, labels, _ in clean_batches)
    providers = {epoch: snapshots[epoch] for epoch in snapshot_epochs}

    for attack_epoch in range(1, args.attack_epochs + 1):
        order_generator = torch.Generator().manual_seed(
            args.seed + 40_000 + attack_epoch
        )
        provider_order = [
            snapshot_epochs[index]
            for index in torch.randperm(
                len(snapshot_epochs), generator=order_generator
            ).tolist()
        ]
        batch_generator = torch.Generator().manual_seed(
            args.seed + 50_000 + attack_epoch
        )
        batch_order = torch.randperm(
            len(clean_batches), generator=batch_generator
        ).tolist()

        shared_seed = args.seed + 60_000 + attack_epoch
        shared_batches, _ = _materialize_fixed_views(
            attacker_data,
            catalog,
            args,
            AUGMENTATION_COMPONENTS,
            shared_seed,
        )
        view_schedule.append(
            {
                "attack_epoch": attack_epoch,
                "variant": DYNAMIC_SHARED,
                "snapshot_epoch": "all",
                "view_seed": shared_seed,
                "sharing": "same augmented tensor shared by every snapshot",
            }
        )
        independent_batches: dict[
            int, list[tuple[torch.Tensor, torch.Tensor, tuple[str, ...]]]
        ] = {}
        for snapshot_index, snapshot_epoch in enumerate(snapshot_epochs):
            independent_seed = (
                args.seed + 70_000 + attack_epoch * 100 + snapshot_index
            )
            batches, _ = _materialize_fixed_views(
                attacker_data,
                catalog,
                args,
                AUGMENTATION_COMPONENTS,
                independent_seed,
            )
            independent_batches[snapshot_epoch] = batches
            view_schedule.append(
                {
                    "attack_epoch": attack_epoch,
                    "variant": DYNAMIC_INDEPENDENT,
                    "snapshot_epoch": snapshot_epoch,
                    "view_seed": independent_seed,
                    "sharing": "independent augmented tensor for this snapshot",
                }
            )

        batches_for_variant = {
            DYNAMIC_NONE: {epoch: clean_batches for epoch in snapshot_epochs},
            DYNAMIC_SHARED: {epoch: shared_batches for epoch in snapshot_epochs},
            DYNAMIC_INDEPENDENT: independent_batches,
        }
        statuses: list[str] = []
        for condition in DYNAMIC_CONDITIONS:
            variant = condition.variant
            model = models[variant]
            # Identical seed gives all three models the same dropout stream.
            seed_everything(args.seed + 80_000 + attack_epoch)
            model.train()
            for snapshot_epoch in provider_order:
                provider = providers[snapshot_epoch]
                provider.eval()
                batches = batches_for_variant[variant][snapshot_epoch]
                for batch_index in batch_order:
                    images, labels, _ = batches[batch_index]
                    images = images.to(device)
                    labels = labels.to(device)
                    exchange = observe_frozen_gradient_exchange(
                        provider, images, labels, criterion
                    )
                    optimizers[variant].zero_grad(set_to_none=True)
                    logits, _ = _forward(
                        model,
                        exchange.server_output_u,
                        exchange.grad_g_to_f,
                    )
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizers[variant].step()

            model.eval()
            validation_loss = 0.0
            validation_correct = 0
            validation_samples = 0
            for snapshot_epoch in provider_order:
                provider = providers[snapshot_epoch]
                provider.eval()
                for images, labels, _ in validation_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    exchange = observe_frozen_gradient_exchange(
                        provider, images, labels, criterion
                    )
                    with torch.no_grad():
                        logits, _ = _forward(
                            model,
                            exchange.server_output_u,
                            exchange.grad_g_to_f,
                        )
                        validation_loss += float(criterion(logits, labels)) * len(
                            images
                        )
                        validation_correct += int(
                            (logits.argmax(1) == labels).sum()
                        )
                    validation_samples += len(images)

            value = validation_loss / validation_samples
            accuracy = validation_correct / validation_samples
            history.append(
                {
                    "attack_epoch": attack_epoch,
                    "variant": variant,
                    "augmentation_schedule": condition.augmentation_schedule,
                    "snapshots": " ".join(map(str, snapshot_epochs)),
                    "provider_order": " ".join(map(str, provider_order)),
                    "batch_order_seed": args.seed + 50_000 + attack_epoch,
                    "unique_source_images": source_samples,
                    "training_transcripts_per_epoch": (
                        source_samples * len(snapshot_epochs)
                    ),
                    "matched_training_conditions": True,
                    "validation_loss": value,
                    "validation_accuracy": accuracy,
                }
            )
            if value < best[variant][0]:
                best[variant] = (value, copy.deepcopy(model.state_dict()))
            statuses.append(f"{condition.augmentation_schedule}={accuracy:.2%}")
        print(
            f"Controlled dynamic {attack_epoch:03d}/{args.attack_epochs:03d}: "
            + ", ".join(statuses),
            flush=True,
        )

    for variant in DYNAMIC_VARIANTS:
        models[variant].load_state_dict(best[variant][1])
        models[variant].eval()
    return history, view_schedule


def _evaluate(
    victim: SplitLearningModel,
    models: dict[str, LabelModel],
    holdout_loader: DataLoader,
    catalog: ClassCatalog,
    epoch: int,
    output: Path,
    device: torch.device,
    variants: tuple[str, ...] = VARIANTS,
    separate_condition_outputs: bool = False,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    criterion = nn.CrossEntropyLoss()
    root = output / "victim" / f"epoch_{epoch:03d}"
    predictions: dict[str, list[int]] = {variant: [] for variant in variants}
    gates: dict[str, list[float]] = {variant: [] for variant in variants}
    sample_rows_by_variant: dict[str, list[dict[str, object]]] = {
        variant: [] for variant in variants
    }
    true_labels: list[int] = []
    per_sample_rows: list[dict[str, object]] = []
    victim.eval()
    for images, labels, sample_ids in holdout_loader:
        images, labels = images.to(device), labels.to(device)
        exchange = observe_frozen_gradient_exchange(victim, images, labels, criterion)
        true_labels.extend(labels.cpu().tolist())
        with torch.no_grad():
            for variant in variants:
                logits, gate = _forward(
                    models[variant],
                    exchange.server_output_u,
                    exchange.grad_g_to_f,
                )
                probabilities = logits.softmax(1)
                inferred = logits.argmax(1)
                predictions[variant].extend(inferred.cpu().tolist())
                if gate is not None:
                    gates[variant].extend(gate.flatten().cpu().tolist())
                for index, sample_id in enumerate(sample_ids):
                    prediction = int(inferred[index])
                    row: dict[str, object] = {
                        "epoch": epoch,
                        "variant": variant,
                        "sample_id": sample_id,
                        "true_label": int(labels[index]),
                        "true_class": catalog.names[int(labels[index])],
                        "inferred_label": prediction,
                        "inferred_class": catalog.names[prediction],
                        "correct": int(prediction == int(labels[index])),
                        "confidence": float(probabilities[index, prediction]),
                        "gate": float(gate[index]) if gate is not None else "",
                    }
                    for class_index, class_name in enumerate(catalog.names):
                        row[f"prob_{class_name}"] = float(
                            probabilities[index, class_index]
                        )
                    per_sample_rows.append(row)
                    sample_rows_by_variant[variant].append(row)
    _write_csv(root / "label_predictions.csv", per_sample_rows)

    summary: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    label_indices = list(range(catalog.num_classes))
    for variant in variants:
        inferred = predictions[variant]
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            true_labels, inferred, average="macro", zero_division=0
        )
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, inferred, labels=label_indices, zero_division=0
        )
        summary_row = {
            "epoch": epoch,
            "variant": variant,
            "samples": len(true_labels),
            "accuracy": accuracy_score(true_labels, inferred),
            "balanced_accuracy": balanced_accuracy_score(true_labels, inferred),
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "mean_gate": float(np.mean(gates[variant])) if gates[variant] else "",
            "random_accuracy_baseline": 1.0 / catalog.num_classes,
        }
        summary.append(summary_row)
        variant_class_rows: list[dict[str, object]] = []
        for label, class_name in enumerate(catalog.names):
            class_row = {
                "epoch": epoch,
                "variant": variant,
                "label": label,
                "class_name": class_name,
                "precision": precision[label],
                "recall": recall[label],
                "f1": f1[label],
                "support": int(support[label]),
            }
            class_rows.append(class_row)
            variant_class_rows.append(class_row)
        matrix = confusion_matrix(true_labels, inferred, labels=label_indices)
        matrix_rows = [
            {
                "true_class": catalog.names[row],
                **{
                    catalog.names[column]: int(matrix[row, column])
                    for column in label_indices
                },
            }
            for row in label_indices
        ]
        variant_root = (
            output / "conditions" / variant / "victim" / f"epoch_{epoch:03d}"
            if separate_condition_outputs
            else root / "label_inference" / variant
        )
        if separate_condition_outputs:
            _write_csv(variant_root / "label_predictions.csv", sample_rows_by_variant[variant])
            _write_csv(variant_root / "label_summary.csv", [summary_row])
            _write_csv(variant_root / "label_class_metrics.csv", variant_class_rows)
        _write_csv(variant_root / "confusion_matrix.csv", matrix_rows)
        _save_confusion_plot(
            matrix,
            catalog.names,
            variant_root / "confusion_matrix.png",
            f"Victim epoch {epoch}: {variant}",
        )
    _write_csv(root / "label_summary.csv", summary)
    _write_csv(root / "label_class_metrics.csv", class_rows)
    return summary, class_rows


def run(args: argparse.Namespace) -> Path:
    seed_everything(args.seed)
    device = _device(args.device)
    victim_learning_rate = (
        args.learning_rate
        if args.victim_learning_rate is None
        else args.victim_learning_rate
    )
    victim_observation_batch_size = (
        args.batch_size
        if args.victim_observation_batch_size is None
        else args.victim_observation_batch_size
    )
    snapshot_epochs = tuple(sorted(set(args.attacker_snapshot_epochs)))
    capture_epochs = tuple(
        sorted(set(args.capture_epochs) | {0, args.victim_finetune_epochs})
    )
    if snapshot_epochs[0] < 0 or snapshot_epochs[-1] > args.attacker_finetune_epochs:
        raise ValueError("attacker snapshot epochs must be within fine-tuning epochs")
    if 0 not in snapshot_epochs or args.attacker_finetune_epochs not in snapshot_epochs:
        raise ValueError("attacker snapshots must include epoch 0 and the final epoch")
    if capture_epochs[0] < 0 or capture_epochs[-1] > args.victim_finetune_epochs:
        raise ValueError("victim capture epochs must be within fine-tuning epochs")
    if min(
        args.warmup_epochs,
        args.attacker_finetune_epochs,
        args.victim_finetune_epochs,
        args.attack_epochs,
        victim_observation_batch_size,
    ) < 1:
        raise ValueError("all training epoch counts and batch sizes must be positive")
    if victim_learning_rate <= 0:
        raise ValueError("victim learning rate must be positive")

    checkpoint = torch.load(
        args.pretrained_autoencoder, map_location=device, weights_only=False
    )
    if int(checkpoint["image_size"]) != args.image_size:
        raise ValueError("pretrained autoencoder image size differs from --image-size")
    catalog = ClassCatalog.discover(args.pretrain_data)
    if catalog.num_classes != 5:
        raise ValueError(f"expected five classes, found {catalog.num_classes}")
    for data_root in (args.attacker_data, args.victim_data):
        if ClassCatalog.discover(data_root).names != catalog.names:
            raise ValueError("pretrain, attacker, and victim class catalogs must match")
    if list(catalog.names) != checkpoint["class_names"]:
        raise ValueError("autoencoder checkpoint class catalog does not match")

    holdout_loader, holdout_rows = _balanced_holdout_loader(
        args.victim_data,
        catalog,
        args.image_size,
        victim_observation_batch_size,
        args.num_workers,
        args.holdout_per_class,
        args.seed + 199,
    )
    if len(holdout_rows) != 100 or args.holdout_per_class != 20:
        raise ValueError("this experiment requires 20 x 5 = 100 holdouts")
    holdout_hashes = _file_hashes([Path(args.victim_data) / "new_holdout"])
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
    overlap = holdout_hashes & training_hashes
    if overlap:
        raise RuntimeError(f"holdout leakage detected: {len(overlap)} files")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "holdout_manifest.csv", holdout_rows)
    (output / "data_leakage_audit.json").write_text(
        json.dumps(
            {
                "holdout_images": len(holdout_hashes),
                "training_images": len(training_hashes),
                "sha256_overlap": len(overlap),
                "balanced_holdout_counts": {
                    name: sum(row["class_name"] == name for row in holdout_rows)
                    for name in catalog.names
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    cut_config = str(checkpoint["cut_config"])
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
    training_history = _warmup_classifier(
        base_model,
        pretrain_loader,
        args.warmup_epochs,
        args.learning_rate,
        device,
    )
    initial_encoder_state = copy.deepcopy(base_model.f_model.state_dict())
    initial_tail_state = copy.deepcopy(base_model.h_model.state_dict())

    attacker = SplitLearningModel(
        copy.deepcopy(base_model.f_model),
        copy.deepcopy(base_model.g_model),
        copy.deepcopy(base_model.h_model),
        cut_config,
    ).to(device)
    attacker_train_loader = make_loader(
        args.attacker_data,
        "train",
        args.image_size,
        args.batch_size,
        args.num_workers,
        shuffle=True,
        augment=True,
        class_names=catalog.names,
    )
    attacker_validation_loader = make_loader(
        args.attacker_data,
        "val",
        args.image_size,
        args.batch_size,
        args.num_workers,
        shuffle=False,
        class_names=catalog.names,
    )
    attacker_optimizers = (
        torch.optim.Adam(attacker.f_model.parameters(), lr=args.learning_rate),
        torch.optim.Adam(attacker.g_model.parameters(), lr=args.learning_rate),
        torch.optim.Adam(attacker.h_model.parameters(), lr=args.learning_rate),
    )
    snapshots: dict[int, SplitLearningModel] = {0: copy.deepcopy(attacker).eval()}
    for epoch in range(1, args.attacker_finetune_epochs + 1):
        stats = _fine_tune_epoch(
            attacker, attacker_train_loader, attacker_optimizers, device
        )
        validation_accuracy = _classification_accuracy(
            attacker, attacker_validation_loader, device
        )
        training_history.append(
            {
                "stage": "attacker_finetune",
                "epoch": epoch,
                "loss": stats["loss"],
                "accuracy": stats["accuracy"],
                "validation_accuracy": validation_accuracy,
            }
        )
        if epoch in snapshot_epochs:
            snapshots[epoch] = copy.deepcopy(attacker).eval()
        print(
            f"Attacker fine-tune {epoch:03d}/{args.attacker_finetune_epochs:03d}: "
            f"train_acc={stats['accuracy']:.2%}, val_acc={validation_accuracy:.2%}",
            flush=True,
        )

    sample_loader = _attack_loader(
        args.attacker_data, catalog, args, augment=False, shuffle=True
    )
    sample_images, sample_labels, _ = next(iter(sample_loader))
    sample_exchange = observe_frozen_gradient_exchange(
        snapshots[snapshot_epochs[-1]],
        sample_images[:1].to(device),
        sample_labels[:1].to(device),
        nn.CrossEntropyLoss(),
    )
    attack_history: list[dict[str, object]] = []
    conditions: tuple[FactorialCondition, ...] = ()
    dynamic_conditions: tuple[DynamicCondition, ...] = ()
    if args.factorial_augmentation_ablation:
        conditions = factorial_conditions(snapshot_epochs)
        variants = tuple(condition.variant for condition in conditions)
        models = _build_factorial_models(
            sample_exchange.server_output_u,
            sample_exchange.grad_g_to_f,
            catalog,
            args,
            conditions,
            device,
        )
        validation_loader = make_loader(
            args.attacker_data,
            "val",
            args.image_size,
            args.attack_batch_size,
            args.num_workers,
            shuffle=False,
            augment=False,
            class_names=catalog.names,
        )
        condition_rows: list[dict[str, object]] = []
        for augmentation_index, augmentations in enumerate(augmentation_subsets()):
            augmentation_key = _augmentation_key(augmentations)
            view_seed = args.seed + 20_000 + augmentation_index
            fixed_batches, view_manifest = _materialize_fixed_views(
                args.attacker_data,
                catalog,
                args,
                augmentations,
                view_seed,
            )
            view_root = output / "augmentation_views" / augmentation_key
            _write_csv(view_root / "fixed_view_manifest.csv", view_manifest)
            pair = tuple(
                condition
                for condition in conditions
                if condition.augmentations == augmentations
            )
            if len(pair) != 2:
                raise RuntimeError(
                    f"expected single/multi pair for {augmentation_key}, found {len(pair)}"
                )
            attack_history.extend(
                _train_factorial_pair(
                    pair,  # type: ignore[arg-type]
                    models,
                    snapshots,
                    fixed_batches,
                    validation_loader,
                    args,
                    device,
                )
            )
            for condition in pair:
                condition_row = {
                    "variant": condition.variant,
                    "snapshot_mode": condition.snapshot_mode,
                    "snapshot_epochs": " ".join(map(str, condition.snapshot_epochs)),
                    "unique_snapshot_count": len(condition.snapshot_epochs),
                    "provider_passes_per_attack_epoch": len(snapshot_epochs),
                    "training_transcripts_per_attack_epoch": (
                        len(view_manifest) * len(snapshot_epochs)
                    ),
                    "matched_training_exposure": True,
                    "augmentation_count": len(condition.augmentations),
                    "augmentations": augmentation_key,
                    "crop": "crop" in condition.augmentations,
                    "flip": "flip" in condition.augmentations,
                    "color": "color" in condition.augmentations,
                    "fixed_view_seed": view_seed,
                    "view_policy": "fixed_per_source_shared_across_snapshots_and_attack_epochs",
                    "fixed_view_manifest": str(
                        view_root / "fixed_view_manifest.csv"
                    ),
                }
                condition_rows.append(condition_row)
                condition_root = output / "conditions" / condition.variant
                condition_root.mkdir(parents=True, exist_ok=True)
                (condition_root / "condition_config.json").write_text(
                    json.dumps(condition_row, indent=2), encoding="utf-8"
                )
        _write_csv(output / "condition_matrix.csv", condition_rows)
    elif args.controlled_dynamic_ablation:
        dynamic_conditions = DYNAMIC_CONDITIONS
        variants = DYNAMIC_VARIANTS
        models = _build_dynamic_models(
            sample_exchange.server_output_u,
            sample_exchange.grad_g_to_f,
            catalog,
            args,
            device,
        )
        validation_loader = make_loader(
            args.attacker_data,
            "val",
            args.image_size,
            args.attack_batch_size,
            args.num_workers,
            shuffle=False,
            augment=False,
            class_names=catalog.names,
        )
        clean_batches, clean_manifest = _materialize_fixed_views(
            args.attacker_data,
            catalog,
            args,
            (),
            args.seed + 90_000,
        )
        _write_csv(
            output / "augmentation_views" / "none" / "clean_view_manifest.csv",
            clean_manifest,
        )
        attack_history, dynamic_view_schedule = _train_controlled_dynamic(
            models,
            snapshots,
            snapshot_epochs,
            clean_batches,
            validation_loader,
            args.attacker_data,
            catalog,
            args,
            device,
        )
        _write_csv(output / "dynamic_view_schedule.csv", dynamic_view_schedule)
        dynamic_rows: list[dict[str, object]] = []
        for condition in dynamic_conditions:
            condition_row = {
                "variant": condition.variant,
                "snapshot_mode": "multi_snapshot",
                "snapshot_epochs": " ".join(map(str, snapshot_epochs)),
                "augmentation_components": "crop flip color"
                if condition.augmentation_schedule != "none"
                else "none",
                "augmentation_schedule": condition.augmentation_schedule,
                "shared_snapshot_order": True,
                "shared_batch_order": True,
                "identical_classifier_initialization": True,
                "matched_training_exposure": True,
                "training_transcripts_per_attack_epoch": (
                    len(clean_manifest) * len(snapshot_epochs)
                ),
            }
            dynamic_rows.append(condition_row)
            condition_root = output / "conditions" / condition.variant
            condition_root.mkdir(parents=True, exist_ok=True)
            (condition_root / "condition_config.json").write_text(
                json.dumps(condition_row, indent=2), encoding="utf-8"
            )
        _write_csv(output / "condition_matrix.csv", dynamic_rows)
    else:
        variants = VARIANTS
        models = _build_models(
            sample_exchange.server_output_u,
            sample_exchange.grad_g_to_f,
            catalog,
            args,
            device,
        )
        groups = (
            TrainingGroup(
                "single_snapshot_original",
                (VARIANT_BASELINE, VARIANT_MAGNITUDE),
                (snapshot_epochs[-1],),
                False,
            ),
            TrainingGroup(
                "multi_snapshot_original", (VARIANT_MULTI,), snapshot_epochs, False
            ),
            TrainingGroup(
                "multi_snapshot_augmented",
                (VARIANT_AUGMENTED, VARIANT_GATED),
                snapshot_epochs,
                True,
            ),
        )
        for group in groups:
            attack_history.extend(
                _train_group(
                    group,
                    models,
                    snapshots,
                    args.attacker_data,
                    catalog,
                    args,
                    device,
                )
            )
    _write_csv(output / "attack_training_history.csv", attack_history)
    checkpoint_root = output / "attack_checkpoints"
    checkpoint_root.mkdir(exist_ok=True)
    for variant, model in models.items():
        payload = {
            "model": model.state_dict(),
            "config": model.config.to_dict(),
            "variant": variant,
            "inputs": "u and/or dL/dz; no z, dL/du, logits, or victim label",
        }
        if args.factorial_augmentation_ablation or args.controlled_dynamic_ablation:
            condition_root = output / "conditions" / variant
            torch.save(payload, condition_root / "attack_checkpoint.pt")
            _write_csv(
                condition_root / "attack_training_history.csv",
                [row for row in attack_history if row["variant"] == variant],
            )
        else:
            torch.save(payload, checkpoint_root / f"{variant}.pt")

    # Keep victim training reproducible regardless of how many attack ablations ran.
    seed_everything(args.seed + 30_000)
    victim_front = ClientFrontFModel(cut_config).to(device)
    victim_front.load_state_dict(initial_encoder_state)
    victim_server = ServerMiddleGModel(cut_config).to(device)
    victim_server.load_state_dict(attacker.g_model.state_dict())
    victim_tail = ClientTailHModel(catalog.num_classes).to(device)
    victim_tail.load_state_dict(initial_tail_state)
    victim = SplitLearningModel(victim_front, victim_server, victim_tail, cut_config)
    victim_train_loader = make_loader(
        args.victim_data,
        "train",
        args.image_size,
        args.batch_size,
        args.num_workers,
        shuffle=True,
        augment=True,
        class_names=catalog.names,
    )
    victim_optimizers = (
        torch.optim.Adam(victim.f_model.parameters(), lr=victim_learning_rate),
        torch.optim.Adam(victim.g_model.parameters(), lr=victim_learning_rate),
        torch.optim.Adam(victim.h_model.parameters(), lr=victim_learning_rate),
    )
    all_summary: list[dict[str, object]] = []
    all_class_rows: list[dict[str, object]] = []
    victim_task_rows: list[dict[str, object]] = []

    def capture(epoch: int) -> None:
        task_accuracy = _classification_accuracy(victim, holdout_loader, device)
        summary, class_rows = _evaluate(
            victim,
            models,
            holdout_loader,
            catalog,
            epoch,
            output,
            device,
            variants=variants,
            separate_condition_outputs=(
                args.factorial_augmentation_ablation
                or args.controlled_dynamic_ablation
            ),
        )
        all_summary.extend(summary)
        all_class_rows.extend(class_rows)
        victim_task_rows.append(
            {"epoch": epoch, "victim_holdout_task_accuracy": task_accuracy}
        )
        print(
            f"Captured victim epoch {epoch:03d}: task_accuracy={task_accuracy:.2%}",
            flush=True,
        )

    capture(0)
    for epoch in range(1, args.victim_finetune_epochs + 1):
        stats = _fine_tune_epoch(victim, victim_train_loader, victim_optimizers, device)
        training_history.append(
            {
                "stage": "victim_finetune",
                "epoch": epoch,
                "loss": stats["loss"],
                "accuracy": stats["accuracy"],
            }
        )
        if epoch in capture_epochs:
            capture(epoch)
        print(
            f"Victim fine-tune {epoch:03d}/{args.victim_finetune_epochs:03d}: "
            f"accuracy={stats['accuracy']:.2%}",
            flush=True,
        )

    _write_csv(output / "training_history.csv", training_history)
    _write_csv(output / "label_inference_improvement_summary.csv", all_summary)
    _write_csv(output / "label_inference_class_metrics.csv", all_class_rows)
    _write_csv(output / "victim_task_accuracy.csv", victim_task_rows)
    if args.factorial_augmentation_ablation:
        condition_by_variant = {
            condition.variant: condition for condition in conditions
        }
        aggregate_rows: list[dict[str, object]] = []
        for condition in conditions:
            condition_root = output / "conditions" / condition.variant
            condition_summary = [
                row for row in all_summary if row["variant"] == condition.variant
            ]
            _write_csv(
                condition_root / "label_inference_summary.csv",
                condition_summary,
            )
            _write_csv(
                condition_root / "label_inference_class_metrics.csv",
                [
                    row
                    for row in all_class_rows
                    if row["variant"] == condition.variant
                ],
            )
            _write_csv(
                condition_root / "victim_task_accuracy.csv", victim_task_rows
            )
            final_row = max(condition_summary, key=lambda row: int(row["epoch"]))
            aggregate_row = {
                "variant": condition.variant,
                "snapshot_mode": condition.snapshot_mode,
                "snapshot_epochs": " ".join(map(str, condition.snapshot_epochs)),
                "augmentation_count": len(condition.augmentations),
                "augmentations": _augmentation_key(condition.augmentations),
                "evaluation_epochs": " ".join(
                    str(row["epoch"]) for row in condition_summary
                ),
                "mean_accuracy": float(
                    np.mean([float(row["accuracy"]) for row in condition_summary])
                ),
                "min_accuracy": min(
                    float(row["accuracy"]) for row in condition_summary
                ),
                "max_accuracy": max(
                    float(row["accuracy"]) for row in condition_summary
                ),
                "final_epoch": int(final_row["epoch"]),
                "final_accuracy": float(final_row["accuracy"]),
                "mean_macro_f1": float(
                    np.mean([float(row["macro_f1"]) for row in condition_summary])
                ),
                "final_macro_f1": float(final_row["macro_f1"]),
            }
            aggregate_rows.append(aggregate_row)
            _write_csv(condition_root / "aggregate_summary.csv", [aggregate_row])
        if set(condition_by_variant) != {row["variant"] for row in aggregate_rows}:
            raise RuntimeError("factorial aggregate is missing one or more conditions")
        _write_csv(output / "factorial_comparison_summary.csv", aggregate_rows)
    elif args.controlled_dynamic_ablation:
        dynamic_aggregate_rows: list[dict[str, object]] = []
        for condition in dynamic_conditions:
            condition_root = output / "conditions" / condition.variant
            condition_summary = [
                row for row in all_summary if row["variant"] == condition.variant
            ]
            condition_class_rows = [
                row
                for row in all_class_rows
                if row["variant"] == condition.variant
            ]
            _write_csv(
                condition_root / "label_inference_summary.csv", condition_summary
            )
            _write_csv(
                condition_root / "label_inference_class_metrics.csv",
                condition_class_rows,
            )
            _write_csv(
                condition_root / "victim_task_accuracy.csv", victim_task_rows
            )
            final_row = max(condition_summary, key=lambda row: int(row["epoch"]))
            aggregate_row = {
                "variant": condition.variant,
                "augmentation_schedule": condition.augmentation_schedule,
                "snapshot_epochs": " ".join(map(str, snapshot_epochs)),
                "evaluation_epochs": " ".join(
                    str(row["epoch"]) for row in condition_summary
                ),
                "mean_accuracy": float(
                    np.mean([float(row["accuracy"]) for row in condition_summary])
                ),
                "std_accuracy_across_victim_epochs": float(
                    np.std([float(row["accuracy"]) for row in condition_summary])
                ),
                "min_accuracy": min(
                    float(row["accuracy"]) for row in condition_summary
                ),
                "max_accuracy": max(
                    float(row["accuracy"]) for row in condition_summary
                ),
                "final_epoch": int(final_row["epoch"]),
                "final_accuracy": float(final_row["accuracy"]),
                "mean_macro_f1": float(
                    np.mean([float(row["macro_f1"]) for row in condition_summary])
                ),
                "final_macro_f1": float(final_row["macro_f1"]),
            }
            dynamic_aggregate_rows.append(aggregate_row)
            _write_csv(condition_root / "aggregate_summary.csv", [aggregate_row])
        _write_csv(
            output / "dynamic_comparison_summary.csv", dynamic_aggregate_rows
        )
    (output / "run_config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "resolved_device": str(device),
                "resolved_victim_learning_rate": victim_learning_rate,
                "resolved_victim_observation_batch_size": (
                    victim_observation_batch_size
                ),
                "cut_config": cut_config,
                "class_names": list(catalog.names),
                "variants": list(variants),
                "variant_design": (
                    {
                        condition.variant: {
                            "snapshot_mode": condition.snapshot_mode,
                            "snapshot_epochs": list(condition.snapshot_epochs),
                            "augmentations": list(condition.augmentations),
                        }
                        for condition in conditions
                    }
                    if args.factorial_augmentation_ablation
                    else (
                        {
                            condition.variant: {
                                "snapshot_mode": "multi_snapshot",
                                "snapshot_epochs": list(snapshot_epochs),
                                "augmentation_schedule": (
                                    condition.augmentation_schedule
                                ),
                            }
                            for condition in dynamic_conditions
                        }
                        if args.controlled_dynamic_ablation
                        else {
                            VARIANT_BASELINE: (
                                "normalized dL/dz direction, final attacker snapshot, "
                                "same classifier backbone as magnitude ablation"
                            ),
                            VARIANT_MAGNITUDE: (
                                "direction + log10 gradient norm, final snapshot"
                            ),
                            VARIANT_MULTI: "direction + norm, all attacker snapshots",
                            VARIANT_AUGMENTED: (
                                "direction + norm, all snapshots, random crop/flip/color views"
                            ),
                            VARIANT_GATED: (
                                "gradient direction + norm primary logits with learned "
                                "gated u logits"
                            ),
                        }
                    )
                ),
                "attack_inputs": ["server_output_u", "grad_g_to_f=dL/dz"],
                "forbidden_attack_inputs": ["z", "dL/du", "logits", "victim label"],
                "victim_label_usage": (
                    "inside the simulated victim loss only; withheld from attack models"
                ),
                "holdout_used_for_training_or_selection": False,
                "holdout_samples": len(holdout_rows),
                "random_accuracy_baseline": 1.0 / catalog.num_classes,
                "factorial_design": (
                    {
                        "snapshot_modes": ["single_snapshot", "multi_snapshot"],
                        "augmentation_components": list(AUGMENTATION_COMPONENTS),
                        "augmentation_subsets": [
                            _augmentation_key(subset)
                            for subset in augmentation_subsets()
                        ],
                        "condition_count": len(conditions),
                        "view_policy": (
                            "one deterministic augmented tensor per source image and "
                            "augmentation subset; reused across snapshot providers and "
                            "all attack epochs"
                        ),
                        "classifier_inputs": "gradient direction + log10 gradient norm",
                        "identical_classifier_initialization": True,
                        "matched_training_exposure": True,
                        "single_snapshot_control": (
                            "repeat the final snapshot for the same number of provider "
                            "passes as multi-snapshot training"
                        ),
                    }
                    if args.factorial_augmentation_ablation
                    else None
                ),
                "controlled_dynamic_design": (
                    {
                        "conditions": [
                            condition.augmentation_schedule
                            for condition in dynamic_conditions
                        ],
                        "augmentation_components": list(AUGMENTATION_COMPONENTS),
                        "snapshot_epochs": list(snapshot_epochs),
                        "shared_provider_order": True,
                        "shared_batch_order": True,
                        "identical_classifier_initialization": True,
                        "matched_training_exposure": True,
                        "dynamic_shared": (
                            "one new augmented view per source and attack epoch, "
                            "shared by every snapshot"
                        ),
                        "dynamic_independent": (
                            "one new augmented view per source, attack epoch, and "
                            "snapshot"
                        ),
                    }
                    if args.controlled_dynamic_ablation
                    else None
                ),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"Label improvement results: {output.resolve()}")
    return output


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()


__all__ = [
    "AUGMENTATION_COMPONENTS",
    "DYNAMIC_CONDITIONS",
    "DYNAMIC_VARIANTS",
    "VARIANTS",
    "augmentation_subsets",
    "build_parser",
    "factorial_conditions",
    "run",
]
