from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from ...decoder.data.image_scaling import denormalize_image
from ...decoder.evaluation.image_comparison_writer import ReconstructionComparisonWriter
from ...decoder.evaluation.reconstruction_metrics import per_sample_metrics
from ...shared.data.class_catalog import ClassCatalog
from ...shared.data.image_dataset import ImageFolderWithID, image_transform, make_loader, validate_class_mapping
from ...shared.reproducibility.random_seed import seed_everything
from ...split_learning.architecture.split_learning_model import SplitLearningModel
from ...split_learning.f_model.client_front_f_model import ClientFrontFModel
from ...split_learning.g_model.server_middle_g_model import ServerMiddleGModel
from ...split_learning.gradient_flow.gradient_exchange import (
    observe_frozen_gradient_exchange,
    run_gradient_exchange_step,
)
from ...split_learning.h_model.client_tail_h_model import ClientTailHModel
from ..models import ReconstructionDecoder, decoder_for_condition


ATTACK_CONDITIONS = ("z_only", "gradient_only", "z_gradient")


@dataclass
class ClientState:
    name: str
    model: SplitLearningModel
    optimizer_f: torch.optim.Optimizer
    optimizer_h: torch.optim.Optimizer
    loader: DataLoader


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Warm up a shared server, train attacks at the common encoder state, "
            "then measure reconstruction and encoder drift across client epochs."
        )
    )
    parser.add_argument("--pretrained-autoencoder", required=True)
    parser.add_argument("--victim-data", default="workspace/data/dataset")
    parser.add_argument("--aux-data", default="workspace/data/dataset_aux_10k")
    parser.add_argument(
        "--output",
        default="workspace/results/shared_pretrained_encoder_drift_attack/animal",
    )
    parser.add_argument("--num-clients", type=int, default=3)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--client-epochs", type=int, default=20)
    parser.add_argument("--capture-epochs", nargs="+", type=int, default=[0, 1, 5, 10, 20])
    parser.add_argument("--attack-epochs", type=int, default=30)
    parser.add_argument("--oracle-epochs", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--attack-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--attack-learning-rate", type=float, default=1e-3)
    parser.add_argument("--signal-spatial-size", type=int, default=16)
    parser.add_argument("--signal-channels", type=int, default=64)
    parser.add_argument("--decoder-base-channels", type=int, default=256)
    parser.add_argument("--decoder-min-channels", type=int, default=32)
    parser.add_argument("--max-aux-train-samples", type=int, default=None)
    parser.add_argument("--max-aux-validation-samples", type=int, default=None)
    parser.add_argument("--max-holdout-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-grid-images", type=int, default=12)
    parser.add_argument("--device", default="auto")
    return parser


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _limit(loader: DataLoader, maximum: int | None) -> DataLoader:
    if maximum is None or maximum >= len(loader.dataset):
        return loader
    if maximum < 1:
        raise ValueError("sample limits must be positive")
    return DataLoader(
        Subset(loader.dataset, range(maximum)),
        batch_size=loader.batch_size,
        shuffle=False,
        num_workers=loader.num_workers,
    )


def _partition_loaders(args, catalog: ClassCatalog) -> list[DataLoader]:
    dataset = ImageFolderWithID(
        Path(args.victim_data) / "train",
        transform=image_transform(args.image_size, augment=True),
        allow_empty=True,
    )
    validate_class_mapping(dataset, catalog.names)
    rng = np.random.default_rng(args.seed)
    partitions = [[] for _ in range(args.num_clients)]
    targets = np.asarray(dataset.targets)
    for label in range(catalog.num_classes):
        indices = np.flatnonzero(targets == label)
        rng.shuffle(indices)
        for position, index in enumerate(indices.tolist()):
            partitions[position % args.num_clients].append(index)
    return [
        DataLoader(
            Subset(dataset, sorted(indices)),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        for indices in partitions
    ]


def _warmup(
    encoder: ClientFrontFModel,
    server: ServerMiddleGModel,
    tail: ClientTailHModel,
    loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: torch.device,
) -> None:
    optimizer = torch.optim.AdamW(
        [*server.parameters(), *tail.parameters()], lr=learning_rate
    )
    criterion = nn.CrossEntropyLoss()
    encoder.eval()
    for epoch in range(1, epochs + 1):
        server.train()
        tail.train()
        correct = count = 0
        for images, labels, _ in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                z = encoder(images)
            logits = tail(server(z))
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            correct += int((logits.argmax(1) == labels).sum())
            count += images.shape[0]
        print(f"Classifier warmup {epoch:03d}/{epochs:03d}: accuracy={correct/count:.2%}")


def _attack_loss(reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return nn.functional.l1_loss(reconstruction, target) + 0.5 * nn.functional.mse_loss(
        reconstruction, target
    )


def _train_attacks(
    provider: SplitLearningModel,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    conditions: tuple[str, ...],
    epochs: int,
    args,
    device: torch.device,
) -> dict[str, ReconstructionDecoder]:
    sample_images, sample_labels, _ = next(iter(train_loader))
    sample_exchange = observe_frozen_gradient_exchange(
        provider, sample_images[:1].to(device), sample_labels[:1].to(device), nn.CrossEntropyLoss()
    )
    z_channels = int(sample_exchange.smashed_z.shape[1])
    models = {
        condition: decoder_for_condition(
            "z_only" if condition == "oracle" else condition,
            z_channels,
            args.image_size,
            args.signal_spatial_size,
            args.signal_channels,
            args.decoder_base_channels,
            args.decoder_min_channels,
        ).to(device)
        for condition in conditions
    }
    optimizers = {
        name: torch.optim.AdamW(model.parameters(), lr=args.attack_learning_rate)
        for name, model in models.items()
    }
    best = {name: (float("inf"), copy.deepcopy(model.state_dict())) for name, model in models.items()}
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        for model in models.values():
            model.train()
        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            exchange = observe_frozen_gradient_exchange(provider, images, labels, criterion)
            target = denormalize_image(images)
            for name, model in models.items():
                optimizers[name].zero_grad(set_to_none=True)
                reconstruction = model(exchange.smashed_z, exchange.grad_g_to_f)
                loss = _attack_loss(reconstruction, target)
                loss.backward()
                optimizers[name].step()
        totals = {name: 0.0 for name in models}
        samples = 0
        for images, labels, _ in validation_loader:
            images, labels = images.to(device), labels.to(device)
            exchange = observe_frozen_gradient_exchange(provider, images, labels, criterion)
            target = denormalize_image(images)
            with torch.no_grad():
                for name, model in models.items():
                    model.eval()
                    totals[name] += float(_attack_loss(model(exchange.smashed_z, exchange.grad_g_to_f), target)) * images.shape[0]
            samples += images.shape[0]
        status = []
        for name, model in models.items():
            value = totals[name] / samples
            status.append(f"{name}={value:.4f}")
            if value < best[name][0]:
                best[name] = (value, copy.deepcopy(model.state_dict()))
        print(f"Attack epoch {epoch:03d}/{epochs:03d}: " + ", ".join(status), flush=True)
    for name, model in models.items():
        model.load_state_dict(best[name][1])
        model.eval()
    return models


def _weight_distance(current: ClientFrontFModel, initial_state: dict) -> float:
    squared = 0.0
    for name, value in current.state_dict().items():
        if value.is_floating_point():
            reference = initial_state[name].to(value.device)
            squared += float((value - reference).float().pow(2).sum())
    return squared ** 0.5


def _write_manifest(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _evaluate_client(
    client: ClientState,
    initial_encoder: ClientFrontFModel,
    initial_state: dict,
    attack_models: dict[str, ReconstructionDecoder],
    holdout_loader: DataLoader,
    epoch: int,
    output: Path,
    catalog: ClassCatalog,
    args,
    device: torch.device,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    root = output / "clients" / client.name / f"epoch_{epoch:03d}"
    transcript_root = root / "transcripts"
    attacker_records = transcript_root / "attacker_records"
    evaluator_records = transcript_root / "evaluator_targets"
    attacker_records.mkdir(parents=True, exist_ok=True)
    evaluator_records.mkdir(parents=True, exist_ok=True)
    attacker_rows: list[dict[str, str]] = []
    evaluator_rows: list[dict[str, str]] = []
    metric_rows: list[dict[str, object]] = []
    drift_cosines: list[float] = []
    drift_l2: list[float] = []
    writers = {
        name: ReconstructionComparisonWriter(
            root / "reconstruction" / name,
            catalog.names,
            max_grid_images=args.max_grid_images,
            save_separate_images=False,
        )
        for name in attack_models
    }
    criterion = nn.CrossEntropyLoss()
    client.model.eval()
    initial_encoder.eval()
    for images, labels, sample_ids in holdout_loader:
        images, labels = images.to(device), labels.to(device)
        exchange = observe_frozen_gradient_exchange(client.model, images, labels, criterion)
        with torch.no_grad():
            initial_z = initial_encoder(images)
            current_flat = exchange.smashed_z.flatten(1)
            initial_flat = initial_z.flatten(1)
            drift_cosines.extend(
                nn.functional.cosine_similarity(current_flat, initial_flat, dim=1).cpu().tolist()
            )
            drift_l2.extend((current_flat - initial_flat).norm(dim=1).cpu().tolist())
            target = denormalize_image(images)
            reconstructions = {
                name: model(exchange.smashed_z, exchange.grad_g_to_f)
                for name, model in attack_models.items()
            }
            metrics = {
                name: per_sample_metrics(reconstruction, target)
                for name, reconstruction in reconstructions.items()
            }
        for index, sample_id in enumerate(sample_ids):
            transcript_id = "transcript_" + hashlib.sha256(
                f"{client.name}:{epoch}:{sample_id}".encode("utf-8")
            ).hexdigest()[:24]
            filename = f"{transcript_id}.npz"
            np.savez_compressed(
                attacker_records / filename,
                smashed_z=exchange.smashed_z[index].cpu().numpy(),
                grad_g_to_f=exchange.grad_g_to_f[index].cpu().numpy(),
            )
            np.savez_compressed(
                evaluator_records / filename,
                target_image=target[index].cpu().numpy(),
                true_label=np.int64(labels[index].item()),
            )
            attacker_rows.append({"transcript_id": transcript_id, "attacker_record": f"attacker_records/{filename}"})
            evaluator_rows.append({"transcript_id": transcript_id, "evaluator_target": f"evaluator_targets/{filename}"})
            for name, reconstruction in reconstructions.items():
                metric_rows.append(
                    {
                        "client": client.name,
                        "epoch": epoch,
                        "condition": name,
                        "transcript_id": transcript_id,
                        **{key: float(value[index]) for key, value in metrics[name].items()},
                    }
                )
                writers[name].save(
                    transcript_id, target[index], reconstruction[index], int(labels[index]), -1
                )
    _write_manifest(transcript_root / "attacker_manifest.csv", ["transcript_id", "attacker_record"], attacker_rows)
    _write_manifest(transcript_root / "evaluator_manifest.csv", ["transcript_id", "evaluator_target"], evaluator_rows)
    for writer in writers.values():
        writer.finalize()
    drift = {
        "client": client.name,
        "epoch": epoch,
        "encoder_weight_l2": _weight_distance(client.model.f_model, initial_state),
        "representation_cosine": float(np.mean(drift_cosines)),
        "representation_l2": float(np.mean(drift_l2)),
    }
    snapshot = root / "snapshot.pt"
    torch.save(
        {
            "client_front": client.model.f_model.state_dict(),
            "client_tail": client.model.h_model.state_dict(),
            "shared_server_middle": client.model.g_model.state_dict(),
            "epoch": epoch,
        }, snapshot,
    )
    return metric_rows, drift


def run(args: argparse.Namespace) -> Path:
    if args.num_clients < 2:
        raise ValueError("num_clients must be at least two")
    capture_epochs = sorted(set(args.capture_epochs) | {0, args.client_epochs})
    if capture_epochs[0] < 0 or capture_epochs[-1] > args.client_epochs:
        raise ValueError("capture epochs must be between zero and client_epochs")
    seed_everything(args.seed)
    device = _device(args.device)
    checkpoint = torch.load(args.pretrained_autoencoder, map_location=device, weights_only=False)
    if int(checkpoint["image_size"]) != args.image_size:
        raise ValueError("pretrained autoencoder image size differs from --image-size")
    cut_config = str(checkpoint["cut_config"])
    victim_catalog = ClassCatalog.discover(args.victim_data)
    aux_catalog = ClassCatalog.discover(args.aux_data)
    if victim_catalog.names != aux_catalog.names or list(victim_catalog.names) != checkpoint["class_names"]:
        raise ValueError("pretraining, victim, and auxiliary class catalogs must match")
    encoder0 = ClientFrontFModel(cut_config).to(device)
    encoder0.load_state_dict(checkpoint["encoder"])
    initial_state = copy.deepcopy(encoder0.state_dict())
    server = ServerMiddleGModel(cut_config).to(device)
    common_tail = ClientTailHModel(victim_catalog.num_classes).to(device)
    warmup_loader = make_loader(
        args.victim_data, "train", args.image_size, args.batch_size, args.num_workers,
        shuffle=True, augment=True, class_names=victim_catalog.names,
    )
    _warmup(encoder0, server, common_tail, warmup_loader, args.warmup_epochs, args.learning_rate, device)
    server_initial = copy.deepcopy(server.state_dict())
    tail_initial = copy.deepcopy(common_tail.state_dict())

    aux_train = _limit(make_loader(
        args.aux_data, "train", args.image_size, args.attack_batch_size, args.num_workers,
        shuffle=True, class_names=aux_catalog.names,
    ), args.max_aux_train_samples)
    aux_validation = _limit(make_loader(
        args.aux_data, "val", args.image_size, args.attack_batch_size, args.num_workers,
        shuffle=False, class_names=aux_catalog.names,
    ), args.max_aux_validation_samples)
    attacker_provider = SplitLearningModel(
        copy.deepcopy(encoder0), copy.deepcopy(server), copy.deepcopy(common_tail), cut_config
    ).to(device)
    print("Training fixed attacks from the malicious client's epoch-0 transcript...")
    attacks = _train_attacks(
        attacker_provider, aux_train, aux_validation, ATTACK_CONDITIONS,
        args.attack_epochs, args, device,
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    attack_dir = output / "attack_checkpoints"
    attack_dir.mkdir(exist_ok=True)
    for name, model in attacks.items():
        torch.save(
            {"model": model.state_dict(), "config": model.config.to_dict(), "condition": name, "trained_at_client_epoch": 0},
            attack_dir / f"{name}.pt",
        )

    partition_loaders = _partition_loaders(args, victim_catalog)
    clients: list[ClientState] = []
    for index, loader in enumerate(partition_loaders, start=1):
        front = ClientFrontFModel(cut_config).to(device)
        front.load_state_dict(initial_state)
        tail = ClientTailHModel(victim_catalog.num_classes).to(device)
        tail.load_state_dict(tail_initial)
        model = SplitLearningModel(front, server, tail, cut_config)
        clients.append(
            ClientState(
                f"client_{index}", model,
                torch.optim.Adam(front.parameters(), lr=args.learning_rate),
                torch.optim.Adam(tail.parameters(), lr=args.learning_rate),
                loader,
            )
        )
    server.load_state_dict(server_initial)
    optimizer_g = torch.optim.Adam(server.parameters(), lr=args.learning_rate)
    holdout_loader = _limit(make_loader(
        args.victim_data, "new_holdout", args.image_size, args.batch_size, args.num_workers,
        shuffle=False, class_names=victim_catalog.names,
    ), args.max_holdout_samples)
    criterion = nn.CrossEntropyLoss()
    all_metrics: list[dict[str, object]] = []
    all_drift: list[dict[str, object]] = []

    def capture(epoch: int) -> None:
        for client in clients:
            models = attacks
            if epoch == args.client_epochs and args.oracle_epochs > 0:
                print(f"Training final oracle for {client.name}...")
                oracle = _train_attacks(
                    client.model, aux_train, aux_validation, ("oracle",),
                    args.oracle_epochs, args, device,
                )
                models = {**attacks, **oracle}
            rows, drift = _evaluate_client(
                client, encoder0, initial_state, models, holdout_loader,
                epoch, output, victim_catalog, args, device,
            )
            all_metrics.extend(rows)
            all_drift.append(drift)

    capture(0)
    for epoch in range(1, args.client_epochs + 1):
        for client in clients:
            client.model.train()
            for images, labels, _ in client.loader:
                run_gradient_exchange_step(
                    client.model, images.to(device), labels.to(device), criterion,
                    client.optimizer_f, optimizer_g, client.optimizer_h, update=True,
                )
        print(f"Completed shared-server client epoch {epoch:03d}/{args.client_epochs:03d}", flush=True)
        if epoch in capture_epochs:
            capture(epoch)

    with (output / "reconstruction_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_metrics[0]))
        writer.writeheader()
        writer.writerows(all_metrics)
    with (output / "drift_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_drift[0]))
        writer.writeheader()
        writer.writerows(all_drift)
    summary: list[dict[str, object]] = []
    groups = sorted({(row["client"], row["epoch"], row["condition"]) for row in all_metrics})
    for client_name, epoch, condition in groups:
        rows = [row for row in all_metrics if (row["client"], row["epoch"], row["condition"]) == (client_name, epoch, condition)]
        summary.append({
            "client": client_name, "epoch": epoch, "condition": condition, "samples": len(rows),
            **{metric: float(np.mean([float(row[metric]) for row in rows])) for metric in ("mse", "mae", "psnr", "ssim")},
        })
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (output / "run_config.json").write_text(
        json.dumps({
            **vars(args), "resolved_device": str(device), "cut_config": cut_config,
            "class_names": list(victim_catalog.names), "capture_epochs": capture_epochs,
            "attack_training_state": "common encoder epoch 0 after frozen-encoder classifier warmup",
            "victim_targets_used_for_attack_training": False,
            "server_weight_recovery_claimed": False,
        }, indent=2, default=str), encoding="utf-8",
    )
    print(f"Drift attack results: {output.resolve()}")
    return output


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()


__all__ = ["ATTACK_CONDITIONS", "build_parser", "run"]
