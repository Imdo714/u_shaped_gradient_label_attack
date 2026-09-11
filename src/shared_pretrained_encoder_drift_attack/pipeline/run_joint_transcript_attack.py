from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "split_learning_matplotlib")
)

import matplotlib
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch import nn
from torch.utils.data import DataLoader, Subset

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from ...decoder.data.image_scaling import denormalize_image
from ...decoder.evaluation.image_comparison_writer import ReconstructionComparisonWriter
from ...decoder.evaluation.reconstruction_metrics import per_sample_metrics
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
from ..models import (
    LabelInferenceClassifier,
    ReconstructionDecoder,
    ReconstructionDecoderConfig,
    label_classifier_for_condition,
)


ATTACK_CONDITIONS = ("u_only", "grad_z_only", "u_grad_z")


@dataclass
class JointAttackModels:
    reconstruction: dict[str, ReconstructionDecoder]
    label: dict[str, LabelInferenceClassifier]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train image- and label-inference attacks on a malicious client's "
            "(u, dL/dz) transcript, then evaluate fixed attacks on a fine-tuning victim."
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
            "animal5_joint_label_reconstruction"
        ),
    )
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--attacker-finetune-epochs", type=int, default=10)
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
    parser.add_argument("--attack-learning-rate", type=float, default=1e-3)
    parser.add_argument("--signal-spatial-size", type=int, default=16)
    parser.add_argument("--signal-channels", type=int, default=64)
    parser.add_argument("--decoder-base-channels", type=int, default=256)
    parser.add_argument("--decoder-min-channels", type=int, default=32)
    parser.add_argument("--classifier-spatial-size", type=int, default=8)
    parser.add_argument("--classifier-hidden-channels", type=int, default=128)
    parser.add_argument("--max-grid-images", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    fieldnames.extend(
        key for row in rows for key in row if key not in fieldnames
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _balanced_holdout_loader(
    data_root: str | Path,
    catalog: ClassCatalog,
    image_size: int,
    batch_size: int,
    num_workers: int,
    per_class: int,
    seed: int,
) -> tuple[DataLoader, list[dict[str, object]]]:
    if per_class < 1:
        raise ValueError("--holdout-per-class must be positive")
    dataset = ImageFolderWithID(
        Path(data_root) / "new_holdout",
        transform=image_transform(image_size),
        allow_empty=True,
    )
    validate_class_mapping(dataset, catalog.names)
    by_label: list[list[int]] = []
    rng = np.random.default_rng(seed)
    for label in range(catalog.num_classes):
        indices = np.flatnonzero(np.asarray(dataset.targets) == label)
        rng.shuffle(indices)
        if len(indices) < per_class:
            raise ValueError(
                f"holdout class {catalog.names[label]!r} needs {per_class} images, "
                f"found {len(indices)}"
            )
        by_label.append(indices[:per_class].tolist())

    # Round-robin ordering makes the first comparison grid class-balanced too.
    selected = [
        by_label[label][position]
        for position in range(per_class)
        for label in range(catalog.num_classes)
    ]
    rows: list[dict[str, object]] = []
    for order, index in enumerate(selected):
        path, label = dataset.samples[index]
        rows.append(
            {
                "order": order,
                "sample_id": dataset.sample_id(index),
                "true_label": label,
                "class_name": catalog.names[label],
                "evaluator_path": str(Path(path).resolve()),
            }
        )
    return (
        DataLoader(
            Subset(dataset, selected),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        rows,
    )


def _classification_accuracy(
    model: SplitLearningModel, loader: DataLoader, device: torch.device
) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels, _ in loader:
            images, labels = images.to(device), labels.to(device)
            correct += int((model.predict(images).argmax(1) == labels).sum())
            total += len(images)
    return correct / total


def _warmup_classifier(
    model: SplitLearningModel,
    loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: torch.device,
) -> list[dict[str, object]]:
    optimizer = torch.optim.AdamW(
        [*model.g_model.parameters(), *model.h_model.parameters()], lr=learning_rate
    )
    criterion = nn.CrossEntropyLoss()
    history: list[dict[str, object]] = []
    model.f_model.eval()
    for epoch in range(1, epochs + 1):
        model.g_model.train()
        model.h_model.train()
        correct = samples = 0
        loss_sum = 0.0
        for images, labels, _ in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                z = model.f_model(images)
            logits = model.h_model(model.g_model(z))
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(images)
            correct += int((logits.argmax(1) == labels).sum())
            samples += len(images)
        row = {
            "stage": "classifier_warmup",
            "epoch": epoch,
            "loss": loss_sum / samples,
            "accuracy": correct / samples,
        }
        history.append(row)
        print(
            f"Classifier warmup {epoch:03d}/{epochs:03d}: "
            f"loss={row['loss']:.4f}, accuracy={row['accuracy']:.2%}",
            flush=True,
        )
    return history


def _fine_tune_epoch(
    model: SplitLearningModel,
    loader: DataLoader,
    optimizers: tuple[torch.optim.Optimizer, torch.optim.Optimizer, torch.optim.Optimizer],
    device: torch.device,
) -> dict[str, float]:
    criterion = nn.CrossEntropyLoss()
    model.train()
    correct = samples = 0
    loss_sum = 0.0
    optimizer_f, optimizer_g, optimizer_h = optimizers
    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)
        exchange = run_gradient_exchange_step(
            model,
            images,
            labels,
            criterion,
            optimizer_f,
            optimizer_g,
            optimizer_h,
            update=True,
        )
        loss_sum += exchange.loss * len(images)
        correct += int((exchange.logits.argmax(1) == labels).sum())
        samples += len(images)
    return {"loss": loss_sum / samples, "accuracy": correct / samples}


def _reconstruction_decoder(
    condition: str,
    u_channels: int,
    grad_z_channels: int,
    args: argparse.Namespace,
    device: torch.device,
) -> ReconstructionDecoder:
    use_u, use_gradient = {
        "u_only": (True, False),
        "grad_z_only": (False, True),
        "u_grad_z": (True, True),
    }[condition]
    return ReconstructionDecoder(
        ReconstructionDecoderConfig(
            z_channels=u_channels,
            grad_channels=grad_z_channels,
            image_size=args.image_size,
            use_z=use_u,
            use_gradient=use_gradient,
            signal_spatial_size=args.signal_spatial_size,
            signal_channels=args.signal_channels,
            base_channels=args.decoder_base_channels,
            min_channels=args.decoder_min_channels,
        )
    ).to(device)


def _signals(condition: str, u: torch.Tensor, grad_z: torch.Tensor):
    if condition == "u_only":
        return u, None
    if condition == "grad_z_only":
        return None, grad_z
    return u, grad_z


def _reconstruction_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return nn.functional.l1_loss(output, target) + 0.5 * nn.functional.mse_loss(
        output, target
    )


def _train_joint_attacks(
    provider: SplitLearningModel,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    catalog: ClassCatalog,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[JointAttackModels, list[dict[str, object]]]:
    sample_images, sample_labels, _ = next(iter(train_loader))
    sample = observe_frozen_gradient_exchange(
        provider,
        sample_images[:1].to(device),
        sample_labels[:1].to(device),
        nn.CrossEntropyLoss(),
    )
    u_channels = int(sample.server_output_u.shape[1])
    grad_z_channels = int(sample.grad_g_to_f.shape[1])
    reconstruction = {
        condition: _reconstruction_decoder(
            condition, u_channels, grad_z_channels, args, device
        )
        for condition in ATTACK_CONDITIONS
    }
    label = {
        condition: label_classifier_for_condition(
            condition,
            u_channels,
            grad_z_channels,
            catalog.num_classes,
            signal_spatial_size=args.classifier_spatial_size,
            signal_channels=args.signal_channels,
            hidden_channels=args.classifier_hidden_channels,
        ).to(device)
        for condition in ATTACK_CONDITIONS
    }
    reconstruction_optimizers = {
        name: torch.optim.AdamW(model.parameters(), lr=args.attack_learning_rate)
        for name, model in reconstruction.items()
    }
    label_optimizers = {
        name: torch.optim.AdamW(model.parameters(), lr=args.attack_learning_rate)
        for name, model in label.items()
    }
    best_reconstruction = {
        name: (float("inf"), copy.deepcopy(model.state_dict()))
        for name, model in reconstruction.items()
    }
    best_label = {
        name: (float("inf"), copy.deepcopy(model.state_dict()))
        for name, model in label.items()
    }
    criterion = nn.CrossEntropyLoss()
    history: list[dict[str, object]] = []
    provider.eval()

    for epoch in range(1, args.attack_epochs + 1):
        for model in [*reconstruction.values(), *label.values()]:
            model.train()
        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            exchange = observe_frozen_gradient_exchange(provider, images, labels, criterion)
            targets = denormalize_image(images)
            for condition in ATTACK_CONDITIONS:
                u, grad_z = _signals(
                    condition, exchange.server_output_u, exchange.grad_g_to_f
                )
                reconstruction_optimizers[condition].zero_grad(set_to_none=True)
                reconstruction_loss = _reconstruction_loss(
                    reconstruction[condition](u, grad_z), targets
                )
                reconstruction_loss.backward()
                reconstruction_optimizers[condition].step()

                label_optimizers[condition].zero_grad(set_to_none=True)
                label_loss = criterion(label[condition](u, grad_z), labels)
                label_loss.backward()
                label_optimizers[condition].step()

        validation = {
            condition: {"reconstruction": 0.0, "label": 0.0, "correct": 0}
            for condition in ATTACK_CONDITIONS
        }
        samples = 0
        for model in [*reconstruction.values(), *label.values()]:
            model.eval()
        for images, labels, _ in validation_loader:
            images, labels = images.to(device), labels.to(device)
            exchange = observe_frozen_gradient_exchange(provider, images, labels, criterion)
            targets = denormalize_image(images)
            with torch.no_grad():
                for condition in ATTACK_CONDITIONS:
                    u, grad_z = _signals(
                        condition, exchange.server_output_u, exchange.grad_g_to_f
                    )
                    restored = reconstruction[condition](u, grad_z)
                    logits = label[condition](u, grad_z)
                    validation[condition]["reconstruction"] += float(
                        _reconstruction_loss(restored, targets)
                    ) * len(images)
                    validation[condition]["label"] += float(
                        criterion(logits, labels)
                    ) * len(images)
                    validation[condition]["correct"] += int(
                        (logits.argmax(1) == labels).sum()
                    )
            samples += len(images)

        status: list[str] = []
        for condition in ATTACK_CONDITIONS:
            reconstruction_value = float(validation[condition]["reconstruction"]) / samples
            label_value = float(validation[condition]["label"]) / samples
            accuracy = int(validation[condition]["correct"]) / samples
            history.append(
                {
                    "epoch": epoch,
                    "condition": condition,
                    "validation_reconstruction_loss": reconstruction_value,
                    "validation_label_loss": label_value,
                    "validation_label_accuracy": accuracy,
                }
            )
            if reconstruction_value < best_reconstruction[condition][0]:
                best_reconstruction[condition] = (
                    reconstruction_value,
                    copy.deepcopy(reconstruction[condition].state_dict()),
                )
            if label_value < best_label[condition][0]:
                best_label[condition] = (
                    label_value,
                    copy.deepcopy(label[condition].state_dict()),
                )
            status.append(
                f"{condition}: recon={reconstruction_value:.4f}, "
                f"label_acc={accuracy:.2%}"
            )
        print(
            f"Joint attack {epoch:03d}/{args.attack_epochs:03d}: " + "; ".join(status),
            flush=True,
        )

    for condition in ATTACK_CONDITIONS:
        reconstruction[condition].load_state_dict(best_reconstruction[condition][1])
        reconstruction[condition].eval()
        label[condition].load_state_dict(best_label[condition][1])
        label[condition].eval()
    return JointAttackModels(reconstruction, label), history


def _save_confusion_plot(
    matrix: np.ndarray,
    class_names: tuple[str, ...],
    path: Path,
    title: str,
) -> None:
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Inferred label",
        title=title,
    )
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right")
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(int(matrix[row, column])),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _evaluate(
    model: SplitLearningModel,
    attacks: JointAttackModels,
    holdout_loader: DataLoader,
    catalog: ClassCatalog,
    epoch: int,
    output: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    root = output / "victim" / f"epoch_{epoch:03d}"
    attacker_records = root / "transcripts" / "attacker_records"
    evaluator_targets = root / "transcripts" / "evaluator_targets"
    attacker_records.mkdir(parents=True, exist_ok=True)
    evaluator_targets.mkdir(parents=True, exist_ok=True)
    writers = {
        condition: ReconstructionComparisonWriter(
            root / "reconstruction" / condition,
            catalog.names,
            max_grid_images=args.max_grid_images,
            save_separate_images=True,
        )
        for condition in ATTACK_CONDITIONS
    }
    reconstruction_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    attacker_manifest: list[dict[str, object]] = []
    evaluator_manifest: list[dict[str, object]] = []
    true_labels: list[int] = []
    predictions: dict[str, list[int]] = {name: [] for name in ATTACK_CONDITIONS}
    criterion = nn.CrossEntropyLoss()
    model.eval()

    for images, labels, sample_ids in holdout_loader:
        images, labels = images.to(device), labels.to(device)
        exchange = observe_frozen_gradient_exchange(model, images, labels, criterion)
        targets = denormalize_image(images)
        true_labels.extend(labels.cpu().tolist())
        with torch.no_grad():
            batch_reconstructions: dict[str, torch.Tensor] = {}
            batch_predictions: dict[str, torch.Tensor] = {}
            batch_probabilities: dict[str, torch.Tensor] = {}
            batch_metrics: dict[str, dict[str, torch.Tensor]] = {}
            for condition in ATTACK_CONDITIONS:
                u, grad_z = _signals(
                    condition, exchange.server_output_u, exchange.grad_g_to_f
                )
                restored = attacks.reconstruction[condition](u, grad_z)
                logits = attacks.label[condition](u, grad_z)
                batch_reconstructions[condition] = restored
                batch_predictions[condition] = logits.argmax(1)
                batch_probabilities[condition] = logits.softmax(1)
                batch_metrics[condition] = per_sample_metrics(restored, targets)
                predictions[condition].extend(logits.argmax(1).cpu().tolist())

        for index, sample_id in enumerate(sample_ids):
            transcript_id = "transcript_" + hashlib.sha256(
                f"victim:{epoch}:{sample_id}".encode("utf-8")
            ).hexdigest()[:24]
            filename = f"{transcript_id}.npz"
            np.savez_compressed(
                attacker_records / filename,
                server_output_u=exchange.server_output_u[index].cpu().numpy(),
                grad_g_to_f=exchange.grad_g_to_f[index].cpu().numpy(),
            )
            np.savez_compressed(
                evaluator_targets / filename,
                target_image=targets[index].cpu().numpy(),
                true_label=np.int64(labels[index].item()),
            )
            attacker_manifest.append(
                {
                    "transcript_id": transcript_id,
                    "attacker_record": f"attacker_records/{filename}",
                }
            )
            evaluator_manifest.append(
                {
                    "transcript_id": transcript_id,
                    "evaluator_target": f"evaluator_targets/{filename}",
                }
            )
            for condition in ATTACK_CONDITIONS:
                predicted = int(batch_predictions[condition][index])
                probability = batch_probabilities[condition][index]
                prediction_row: dict[str, object] = {
                    "epoch": epoch,
                    "condition": condition,
                    "transcript_id": transcript_id,
                    "true_label": int(labels[index]),
                    "true_class": catalog.names[int(labels[index])],
                    "inferred_label": predicted,
                    "inferred_class": catalog.names[predicted],
                    "correct": int(predicted == int(labels[index])),
                    "confidence": float(probability[predicted]),
                }
                for class_index, class_name in enumerate(catalog.names):
                    prediction_row[f"prob_{class_name}"] = float(probability[class_index])
                prediction_rows.append(prediction_row)
                reconstruction_rows.append(
                    {
                        "epoch": epoch,
                        "condition": condition,
                        "transcript_id": transcript_id,
                        "true_label": int(labels[index]),
                        "true_class": catalog.names[int(labels[index])],
                        **{
                            metric: float(values[index])
                            for metric, values in batch_metrics[condition].items()
                        },
                    }
                )
                writers[condition].save(
                    transcript_id,
                    targets[index],
                    batch_reconstructions[condition][index],
                    int(labels[index]),
                    predicted,
                )

    _write_csv(root / "transcripts" / "attacker_manifest.csv", attacker_manifest)
    _write_csv(root / "transcripts" / "evaluator_manifest.csv", evaluator_manifest)
    _write_csv(root / "label_predictions.csv", prediction_rows)
    _write_csv(root / "reconstruction_metrics.csv", reconstruction_rows)
    for writer in writers.values():
        writer.finalize()

    label_summary: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    labels_range = list(range(catalog.num_classes))
    for condition in ATTACK_CONDITIONS:
        predicted = predictions[condition]
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels,
            predicted,
            labels=labels_range,
            zero_division=0,
        )
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            true_labels, predicted, average="macro", zero_division=0
        )
        label_summary.append(
            {
                "epoch": epoch,
                "condition": condition,
                "samples": len(true_labels),
                "accuracy": accuracy_score(true_labels, predicted),
                "balanced_accuracy": balanced_accuracy_score(true_labels, predicted),
                "macro_precision": macro_precision,
                "macro_recall": macro_recall,
                "macro_f1": macro_f1,
                "random_accuracy_baseline": 1.0 / catalog.num_classes,
            }
        )
        for label_index, class_name in enumerate(catalog.names):
            class_rows.append(
                {
                    "epoch": epoch,
                    "condition": condition,
                    "label": label_index,
                    "class_name": class_name,
                    "precision": precision[label_index],
                    "recall": recall[label_index],
                    "f1": f1[label_index],
                    "support": int(support[label_index]),
                }
            )
        matrix = confusion_matrix(true_labels, predicted, labels=labels_range)
        _write_csv(
            root / "label_inference" / condition / "confusion_matrix.csv",
            [
                {"true_class": catalog.names[row], **{
                    catalog.names[column]: int(matrix[row, column])
                    for column in labels_range
                }}
                for row in labels_range
            ],
        )
        _save_confusion_plot(
            matrix,
            catalog.names,
            root / "label_inference" / condition / "confusion_matrix.png",
            f"Victim epoch {epoch}: {condition}",
        )
    _write_csv(root / "label_summary.csv", label_summary)
    _write_csv(root / "label_class_metrics.csv", class_rows)

    reconstruction_summary = []
    for condition in ATTACK_CONDITIONS:
        rows = [row for row in reconstruction_rows if row["condition"] == condition]
        reconstruction_summary.append(
            {
                "epoch": epoch,
                "condition": condition,
                "samples": len(rows),
                **{
                    metric: float(np.mean([float(row[metric]) for row in rows]))
                    for metric in ("mse", "mae", "psnr", "ssim")
                },
            }
        )
    _write_csv(root / "reconstruction_summary.csv", reconstruction_summary)
    return label_summary, reconstruction_summary


def _file_hashes(roots: list[Path]) -> set[str]:
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    result: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in suffixes:
                result.add(hashlib.sha256(path.read_bytes()).hexdigest())
    return result


def run(args: argparse.Namespace) -> Path:
    capture_epochs = sorted(
        set(args.capture_epochs) | {0, args.victim_finetune_epochs}
    )
    if capture_epochs[0] < 0 or capture_epochs[-1] > args.victim_finetune_epochs:
        raise ValueError("capture epochs must be within victim fine-tuning epochs")
    if min(
        args.warmup_epochs,
        args.attacker_finetune_epochs,
        args.victim_finetune_epochs,
        args.attack_epochs,
    ) < 1:
        raise ValueError("all training epoch counts must be positive")
    seed_everything(args.seed)
    device = _device(args.device)
    checkpoint = torch.load(
        args.pretrained_autoencoder, map_location=device, weights_only=False
    )
    if int(checkpoint["image_size"]) != args.image_size:
        raise ValueError("pretrained autoencoder image size differs from --image-size")
    catalog = ClassCatalog.discover(args.pretrain_data)
    attacker_catalog = ClassCatalog.discover(args.attacker_data)
    victim_catalog = ClassCatalog.discover(args.victim_data)
    if not (
        catalog.names == attacker_catalog.names == victim_catalog.names
        and list(catalog.names) == checkpoint["class_names"]
    ):
        raise ValueError(
            "pretraining checkpoint, pretrain, attacker, and victim classes must match"
        )
    if catalog.num_classes != 5:
        raise ValueError(
            f"this experiment requires five animal classes, found {catalog.num_classes}"
        )

    holdout_loader, holdout_rows = _balanced_holdout_loader(
        args.victim_data,
        catalog,
        args.image_size,
        args.batch_size,
        args.num_workers,
        args.holdout_per_class,
        args.seed + 99,
    )
    if len(holdout_rows) != 100 or args.holdout_per_class != 20:
        raise ValueError("the evaluation holdout must contain 20 x 5 = 100 images")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "holdout_manifest.csv", holdout_rows)
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
        raise RuntimeError(f"holdout leakage detected: {len(overlap)} duplicate files")
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
    initial_server_state = copy.deepcopy(base_model.g_model.state_dict())
    initial_tail_state = copy.deepcopy(base_model.h_model.state_dict())

    attacker_model = SplitLearningModel(
        copy.deepcopy(base_model.f_model),
        copy.deepcopy(base_model.g_model),
        copy.deepcopy(base_model.h_model),
        cut_config,
    ).to(device)
    attacker_finetune_loader = make_loader(
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
        torch.optim.Adam(attacker_model.f_model.parameters(), lr=args.learning_rate),
        torch.optim.Adam(attacker_model.g_model.parameters(), lr=args.learning_rate),
        torch.optim.Adam(attacker_model.h_model.parameters(), lr=args.learning_rate),
    )
    for epoch in range(1, args.attacker_finetune_epochs + 1):
        stats = _fine_tune_epoch(
            attacker_model, attacker_finetune_loader, attacker_optimizers, device
        )
        validation_accuracy = _classification_accuracy(
            attacker_model, attacker_validation_loader, device
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
        print(
            f"Attacker fine-tune {epoch:03d}/{args.attacker_finetune_epochs:03d}: "
            f"train_acc={stats['accuracy']:.2%}, val_acc={validation_accuracy:.2%}",
            flush=True,
        )

    attack_train_loader = make_loader(
        args.attacker_data,
        "train",
        args.image_size,
        args.attack_batch_size,
        args.num_workers,
        shuffle=True,
        augment=False,
        class_names=catalog.names,
    )
    attack_validation_loader = make_loader(
        args.attacker_data,
        "val",
        args.image_size,
        args.attack_batch_size,
        args.num_workers,
        shuffle=False,
        class_names=catalog.names,
    )
    print("Training C_psi and D_phi from the attacker's (u, dL/dz)...", flush=True)
    attacks, attack_history = _train_joint_attacks(
        attacker_model,
        attack_train_loader,
        attack_validation_loader,
        catalog,
        args,
        device,
    )
    _write_csv(output / "attack_training_history.csv", attack_history)
    checkpoint_root = output / "attack_checkpoints"
    checkpoint_root.mkdir(exist_ok=True)
    for condition in ATTACK_CONDITIONS:
        torch.save(
            {
                "model": attacks.reconstruction[condition].state_dict(),
                "config": attacks.reconstruction[condition].config.to_dict(),
                "condition": condition,
                "input": "u and/or dL/dz",
            },
            checkpoint_root / f"reconstruction_{condition}.pt",
        )
        torch.save(
            {
                "model": attacks.label[condition].state_dict(),
                "config": attacks.label[condition].config.to_dict(),
                "condition": condition,
                "input": "u and/or dL/dz",
            },
            checkpoint_root / f"label_{condition}.pt",
        )

    # The central server continues from the malicious client's legitimate session.
    # A different victim receives the original deployed encoder and tail.
    victim_front = ClientFrontFModel(cut_config).to(device)
    victim_front.load_state_dict(initial_encoder_state)
    victim_server = ServerMiddleGModel(cut_config).to(device)
    victim_server.load_state_dict(attacker_model.g_model.state_dict())
    victim_tail = ClientTailHModel(catalog.num_classes).to(device)
    victim_tail.load_state_dict(initial_tail_state)
    victim_model = SplitLearningModel(
        victim_front, victim_server, victim_tail, cut_config
    ).to(device)
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
        torch.optim.Adam(victim_model.f_model.parameters(), lr=args.learning_rate),
        torch.optim.Adam(victim_model.g_model.parameters(), lr=args.learning_rate),
        torch.optim.Adam(victim_model.h_model.parameters(), lr=args.learning_rate),
    )
    all_label_summary: list[dict[str, object]] = []
    all_reconstruction_summary: list[dict[str, object]] = []
    victim_task_rows: list[dict[str, object]] = []

    def capture(epoch: int) -> None:
        task_accuracy = _classification_accuracy(victim_model, holdout_loader, device)
        label_rows, reconstruction_rows = _evaluate(
            victim_model,
            attacks,
            holdout_loader,
            catalog,
            epoch,
            output,
            args,
            device,
        )
        all_label_summary.extend(label_rows)
        all_reconstruction_summary.extend(reconstruction_rows)
        victim_task_rows.append(
            {
                "epoch": epoch,
                "victim_holdout_task_accuracy": task_accuracy,
            }
        )
        print(
            f"Captured victim epoch {epoch:03d}: task_accuracy={task_accuracy:.2%}",
            flush=True,
        )

    capture(0)
    for epoch in range(1, args.victim_finetune_epochs + 1):
        stats = _fine_tune_epoch(victim_model, victim_train_loader, victim_optimizers, device)
        training_history.append(
            {
                "stage": "victim_finetune",
                "epoch": epoch,
                "loss": stats["loss"],
                "accuracy": stats["accuracy"],
            }
        )
        print(
            f"Victim fine-tune {epoch:03d}/{args.victim_finetune_epochs:03d}: "
            f"accuracy={stats['accuracy']:.2%}",
            flush=True,
        )
        if epoch in capture_epochs:
            capture(epoch)

    _write_csv(output / "training_history.csv", training_history)
    _write_csv(output / "label_inference_summary.csv", all_label_summary)
    _write_csv(output / "reconstruction_summary.csv", all_reconstruction_summary)
    _write_csv(output / "victim_task_accuracy.csv", victim_task_rows)
    torch.save(
        {
            "initial_server": initial_server_state,
            "attacker": attacker_model.state_dict(),
            "victim_final": victim_model.state_dict(),
        },
        output / "experiment_models.pt",
    )
    (output / "run_config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "resolved_device": str(device),
                "cut_config": cut_config,
                "class_names": list(catalog.names),
                "capture_epochs": capture_epochs,
                "attack_inputs": ["server_output_u", "grad_g_to_f=dL/dz"],
                "attack_targets": ["private_label_y", "private_image_x"],
                "attacker_knows_victim_labels": False,
                "label_usage_note": (
                    "Victim labels are used inside the simulated client loss only to "
                    "produce dL/dz and are withheld from C_psi and D_phi."
                ),
                "server_timeline": (
                    "warmup -> malicious-client fine-tuning -> victim fine-tuning"
                ),
                "victim_initial_encoder": "the originally distributed pretrained encoder",
                "holdout_used_for_training": False,
                "holdout_samples": len(holdout_rows),
                "holdout_per_class": args.holdout_per_class,
                "random_label_accuracy_baseline": 1.0 / catalog.num_classes,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"Joint attack results: {output.resolve()}")
    return output


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()


__all__ = ["ATTACK_CONDITIONS", "build_parser", "run"]
