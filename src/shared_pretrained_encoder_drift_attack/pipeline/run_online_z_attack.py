from __future__ import annotations

import argparse
import copy
import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from ...shared.data.class_catalog import ClassCatalog
from ...shared.data.image_dataset import make_loader
from ...shared.reproducibility.random_seed import seed_everything
from ...split_learning.architecture.split_learning_model import SplitLearningModel
from ...split_learning.f_model.client_front_f_model import ClientFrontFModel
from ...split_learning.g_model.server_middle_g_model import ServerMiddleGModel
from ...split_learning.gradient_flow.gradient_exchange import run_gradient_exchange_step
from ...split_learning.h_model.client_tail_h_model import ClientTailHModel
from .run_joint_transcript_attack import _classification_accuracy, _device, _warmup_classifier
from .run_online_transcript_label_attack import (
    AUGMENTATION_MODES,
    ClientRuntime,
    _SignalAdapter,
    _aggregate,
    _attack_validation_ids,
    _attacker_dataset,
    _build_world,
    _loader,
    _metrics_from_predictions,
    _partition_victim_dataset,
    _write_csv,
)


@dataclass
class ZTranscriptBundle:
    z: Tensor
    labels: Tensor
    sample_ids: tuple[str, ...]
    rounds: Tensor
    server_steps: Tensor

    def __len__(self) -> int:
        return len(self.labels)

    def select(self, indices: Tensor | Sequence[int]) -> "ZTranscriptBundle":
        if not isinstance(indices, Tensor):
            indices = torch.tensor(indices, dtype=torch.long)
        index_list = (
            indices.nonzero(as_tuple=False).flatten().tolist()
            if indices.dtype == torch.bool
            else indices.flatten().tolist()
        )
        return ZTranscriptBundle(
            self.z[indices],
            self.labels[indices],
            tuple(self.sample_ids[index] for index in index_list),
            self.rounds[indices],
            self.server_steps[indices],
        )


class ZTranscriptAccumulator:
    def __init__(self) -> None:
        self._z: list[Tensor] = []
        self._labels: list[Tensor] = []
        self._ids: list[str] = []
        self._rounds: list[Tensor] = []
        self._steps: list[Tensor] = []

    def add(
        self,
        z: Tensor,
        labels: Tensor,
        sample_ids: Sequence[str],
        round_index: int,
        server_step: int,
    ) -> None:
        count = len(labels)
        if len(sample_ids) != count:
            raise ValueError("z transcript batch IDs and labels must match")
        self._z.append(z.detach().cpu())
        self._labels.append(labels.detach().cpu())
        self._ids.extend(sample_ids)
        self._rounds.append(torch.full((count,), round_index, dtype=torch.long))
        self._steps.append(torch.full((count,), server_step, dtype=torch.long))

    def bundle(self) -> ZTranscriptBundle:
        if not self._labels:
            raise ValueError("no z transcripts were collected")
        return ZTranscriptBundle(
            torch.cat(self._z),
            torch.cat(self._labels),
            tuple(self._ids),
            torch.cat(self._rounds),
            torch.cat(self._steps),
        )


class ZLabelDataset(Dataset):
    def __init__(self, bundle: ZTranscriptBundle) -> None:
        self.bundle = bundle

    def __len__(self) -> int:
        return len(self.bundle)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.bundle.z[index], self.bundle.labels[index]


@dataclass(frozen=True)
class ZLabelClassifierConfig:
    z_channels: int
    num_classes: int
    signal_spatial_size: int = 8
    signal_channels: int = 64
    hidden_channels: int = 128
    dropout: float = 0.2

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class ZLabelClassifier(nn.Module):
    """Learned C_psi that receives only the transmitted smashed representation z."""

    def __init__(self, config: ZLabelClassifierConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = _SignalAdapter(
            config.z_channels, config.signal_channels, config.signal_spatial_size
        )
        self.classifier = nn.Sequential(
            nn.Linear(config.signal_channels, config.hidden_channels),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_channels, config.num_classes),
        )

    def forward(self, z: Tensor) -> Tensor:
        features = nn.functional.adaptive_avg_pool2d(self.encoder(z), 1).flatten(1)
        return self.classifier(features)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the z-only online label-inference baseline with Learned C_psi "
            "and Cosine Class Prototype."
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
            "animal5_online_z_only"
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
        "--matched-budget", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser


def _validate_args(args: argparse.Namespace) -> tuple[int, ...]:
    rounds = tuple(sorted(set(args.collection_rounds)))
    if not rounds or rounds[0] < 1 or rounds[-1] > args.global_rounds:
        raise ValueError("collection rounds must be within the global-round range")
    if args.num_victims < 1:
        raise ValueError("num-victims must be positive")
    if not 0 < args.attack_validation_fraction < 0.5:
        raise ValueError("attack-validation-fraction must be between 0 and 0.5")
    return rounds


def _temporal_z_bundles(
    bundle: ZTranscriptBundle,
    validation_ids: set[str],
    collection_rounds: tuple[int, ...],
    include_matched: bool,
    seed: int,
) -> dict[str, tuple[ZTranscriptBundle, ZTranscriptBundle, str]]:
    is_validation = torch.tensor(
        [sample_id in validation_ids for sample_id in bundle.sample_ids],
        dtype=torch.bool,
    )
    latest_mask = bundle.rounds == collection_rounds[-1]
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
    base, remainder = divmod(latest_train_count, len(by_group))
    selected: list[int] = []
    for group_index, (group, indices) in enumerate(sorted(by_group.items())):
        random.Random(f"{seed}:z-matched:{group[0]}:{group[1]}").shuffle(indices)
        selected.extend(indices[: base + int(group_index < remainder)])
    outputs["multi_matched"] = (
        bundle.select(sorted(selected)),
        bundle.select(is_validation),
        "matched_latest_window",
    )
    return outputs


def _train_local_epoch_with_z(
    client: ClientRuntime,
    optimizer_g: torch.optim.Optimizer,
    loader: DataLoader,
    device: torch.device,
    round_index: int,
    server_step: int,
    collector: ZTranscriptAccumulator | None,
) -> tuple[dict[str, float], int]:
    criterion = nn.CrossEntropyLoss()
    client.model.train()
    loss_sum = 0.0
    correct = samples = 0
    for images, labels, sample_ids in loader:
        images, labels = images.to(device), labels.to(device)
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
                exchange.smashed_z,
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


def _observe_z_bundle(
    model: SplitLearningModel,
    loader: DataLoader,
    device: torch.device,
    round_index: int,
    server_step: int,
) -> ZTranscriptBundle:
    collector = ZTranscriptAccumulator()
    model.eval()
    with torch.no_grad():
        for images, labels, sample_ids in loader:
            images, labels = images.to(device), labels.to(device)
            collector.add(
                model.f_model(images),
                labels,
                tuple(sample_ids),
                round_index,
                server_step,
            )
    return collector.bundle()


def _run_z_world(
    mode: str,
    seed: int,
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
) -> tuple[ZTranscriptBundle, list[ZTranscriptBundle], list[dict[str, object]]]:
    attacker_dataset = _attacker_dataset(
        mode, args.attacker_data, catalog, args.image_size, seed + 10_000
    )
    _, clients, optimizer_g = _build_world(
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
    attacker_records = ZTranscriptAccumulator()
    training_rows: list[dict[str, object]] = []
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
            stats, server_step = _train_local_epoch_with_z(
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
                    "client": client.name,
                    "loss": stats["loss"],
                    "accuracy": stats["accuracy"],
                    "server_step_after": server_step,
                }
            )
        print(
            f"seed={seed} mode={mode} z-world round={round_index:02d}/"
            f"{args.global_rounds:02d}",
            flush=True,
        )
    victim_bundles: list[ZTranscriptBundle] = []
    for victim_index, holdout_dataset in enumerate(victim_holdout, start=1):
        loader = _loader(
            holdout_dataset,
            args.batch_size,
            seed + 80_000 + victim_index,
            shuffle=False,
            num_workers=args.num_workers,
        )
        victim_bundles.append(
            _observe_z_bundle(
                clients[victim_index].model,
                loader,
                device,
                args.global_rounds,
                server_step,
            )
        )
        training_rows.append(
            {
                "seed": seed,
                "augmentation_mode": mode,
                "global_round": args.global_rounds,
                "client": f"victim_{victim_index}",
                "stage": "final_holdout",
                "task_accuracy": _classification_accuracy(
                    clients[victim_index].model, loader, device
                ),
                "server_step_after": server_step,
            }
        )
    return attacker_records.bundle(), victim_bundles, training_rows


def _train_z_classifier(
    train_bundle: ZTranscriptBundle,
    validation_bundle: ZTranscriptBundle,
    num_classes: int,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[ZLabelClassifier, list[dict[str, object]]]:
    seed_everything(seed)
    model = ZLabelClassifier(
        ZLabelClassifierConfig(
            z_channels=int(train_bundle.z.shape[1]),
            num_classes=num_classes,
            signal_spatial_size=args.signal_spatial_size,
            signal_channels=args.signal_channels,
            hidden_channels=args.hidden_channels,
            dropout=args.dropout,
        )
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.attack_learning_rate)
    criterion = nn.CrossEntropyLoss()
    train_loader = _loader(
        ZLabelDataset(train_bundle),
        args.attack_batch_size,
        seed + 1,
        shuffle=True,
        num_workers=args.num_workers,
    )
    validation_loader = _loader(
        ZLabelDataset(validation_bundle),
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
        for z, labels in train_loader:
            z, labels = z.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(z)
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
            for z, labels in validation_loader:
                z, labels = z.to(device), labels.to(device)
                logits = model(z)
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


def _evaluate_z_classifier(
    model: ZLabelClassifier,
    bundle: ZTranscriptBundle,
    catalog: ClassCatalog,
    device: torch.device,
):
    loader = _loader(
        ZLabelDataset(bundle), 64, 0, shuffle=False, num_workers=0
    )
    predictions: list[int] = []
    labels: list[int] = []
    with torch.no_grad():
        for z, target in loader:
            predictions.extend(model(z.to(device)).argmax(1).cpu().tolist())
            labels.extend(target.tolist())
    return _metrics_from_predictions(labels, predictions, catalog)


def _z_prototype_features(bundle: ZTranscriptBundle) -> Tensor:
    features = nn.functional.adaptive_avg_pool2d(bundle.z.float(), (4, 4)).flatten(1)
    return nn.functional.normalize(features, dim=1)


def _evaluate_z_prototype(
    train_bundle: ZTranscriptBundle,
    victim_bundle: ZTranscriptBundle,
    catalog: ClassCatalog,
):
    train_features = _z_prototype_features(train_bundle)
    victim_features = _z_prototype_features(victim_bundle)
    prototypes = torch.stack(
        [
            nn.functional.normalize(
                train_features[train_bundle.labels == label].mean(0), dim=0
            )
            for label in range(catalog.num_classes)
        ]
    )
    predictions = (victim_features @ prototypes.T).argmax(1).tolist()
    return _metrics_from_predictions(victim_bundle.labels.tolist(), predictions, catalog)


def _table(
    rows: Sequence[dict[str, object]], attack_type: str
) -> list[str]:
    lookup = {
        (str(row["augmentation_mode"]), str(row["temporal_mode"])): row
        for row in rows
        if row["attack_type"] == attack_type
    }
    lines = [
        "| 공격자 증강 | Single-Latest | Multi-Matched | Multi-Natural |",
        "|---|---:|---:|---:|",
    ]
    temporal_modes = ("single_latest", "multi_matched", "multi_natural")
    maxima = {
        temporal: max(
            float(lookup[(mode, temporal)]["mean_accuracy"])
            for mode in ("none", "fixed", "dynamic")
        )
        for temporal in temporal_modes
    }
    for mode, display in (("none", "None (무 증강)"), ("fixed", "Fixed"), ("dynamic", "Dynamic")):
        values: list[str] = []
        for temporal in temporal_modes:
            value = float(lookup[(mode, temporal)]["mean_accuracy"])
            formatted = f"{value:.1%}"
            values.append(f"**{formatted}**" if np.isclose(value, maxima[temporal]) else formatted)
        lines.append(f"| {display} | {' | '.join(values)} |")
    return lines


def _write_result_readme(
    output: Path,
    label_aggregate: Sequence[dict[str, object]],
    seed_count: int,
    victims_per_seed: int,
) -> None:
    lines = [
        "# Online `z`-only 공격 결과",
        "",
        "모든 라벨 추론은 서버가 실제로 수신한 `z=f(x)`만 사용한다.",
        "",
        "## Learned `C_psi`: `z`-only 라벨 추론 정확도",
        "",
        *_table(label_aggregate, "learned_cpsi"),
        "",
        "## Cosine Prototype: `z`-only 라벨 추론 정확도",
        "",
        *_table(label_aggregate, "cosine_class_prototype"),
        "",
        f"- 라벨 표는 {seed_count}개 seed와 seed별 Victim {victims_per_seed}명의 평균 정확도다.",
        "- 두 공격기 모두 `u`, `dL/du`, `dL/dz`를 입력받지 않는다.",
    ]
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
    all_label_rows: list[dict[str, object]] = []
    all_class_rows: list[dict[str, object]] = []
    all_training_rows: list[dict[str, object]] = []

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
        for row in _warmup_classifier(
            base_model,
            pretrain_loader,
            args.warmup_epochs,
            args.client_learning_rate,
            device,
        ):
            all_training_rows.append({"seed": seed, "stage": "warmup", **row})
        initial_f = copy.deepcopy(base_model.f_model.state_dict())
        initial_g = copy.deepcopy(base_model.g_model.state_dict())
        initial_h = copy.deepcopy(base_model.h_model.state_dict())
        victim_train, _ = _partition_victim_dataset(
            args.victim_data,
            "train",
            catalog,
            args.image_size,
            args.num_victims,
            2026,
            augment=True,
        )
        victim_validation, _ = _partition_victim_dataset(
            args.victim_data,
            "val",
            catalog,
            args.image_size,
            args.num_victims,
            2026,
            augment=False,
        )
        victim_holdout, _ = _partition_victim_dataset(
            args.victim_data,
            "new_holdout",
            catalog,
            args.image_size,
            args.num_victims,
            2026,
            augment=False,
        )
        for mode in args.augmentation_modes:
            attacker_bundle, victim_bundles, training_rows = _run_z_world(
                mode,
                seed,
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
            validation_ids = _attack_validation_ids(
                attacker_bundle,
                catalog.num_classes,
                args.attack_validation_fraction,
                seed,
            )
            temporal = _temporal_z_bundles(
                attacker_bundle,
                validation_ids,
                collection_rounds,
                args.matched_budget,
                seed,
            )
            for temporal_index, (temporal_mode, values) in enumerate(temporal.items()):
                train_bundle, validation_bundle, budget = values
                attack_seed = seed + 100_000 + temporal_index * 100
                classifier, classifier_history = _train_z_classifier(
                    train_bundle,
                    validation_bundle,
                    catalog.num_classes,
                    args,
                    device,
                    attack_seed,
                )
                condition_root = output / f"seed_{seed}" / mode / temporal_mode
                condition_root.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model": classifier.state_dict(),
                        "config": classifier.config.to_dict(),
                        "attack_input": ["z"],
                        "training_target": ["attacker_y"],
                    },
                    condition_root / "learned_cpsi_z_only.pt",
                )
                _write_csv(condition_root / "classifier_training_history.csv", classifier_history)
                for victim_index, victim_bundle in enumerate(victim_bundles, start=1):
                    common = {
                        "seed": seed,
                        "augmentation_mode": mode,
                        "temporal_mode": temporal_mode,
                        "signal_mode": "z_only",
                        "budget": budget,
                        "training_records": len(train_bundle),
                        "victim_index": victim_index,
                        "victim_round": args.global_rounds,
                    }
                    for attack_type in ("learned_cpsi", "cosine_class_prototype"):
                        if attack_type == "learned_cpsi":
                            summary, class_rows, matrix = _evaluate_z_classifier(
                                classifier, victim_bundle, catalog, device
                            )
                        else:
                            summary, class_rows, matrix = _evaluate_z_prototype(
                                train_bundle, victim_bundle, catalog
                            )
                        all_label_rows.append({**common, "attack_type": attack_type, **summary})
                        all_class_rows.extend(
                            {**common, "attack_type": attack_type, **row}
                            for row in class_rows
                        )
                        np.savetxt(
                            condition_root
                            / f"victim_{victim_index}_{attack_type}_confusion.csv",
                            matrix,
                            delimiter=",",
                            fmt="%d",
                        )
                print(
                    f"seed={seed} mode={mode} temporal={temporal_mode} z attacks complete",
                    flush=True,
                )

    label_aggregate = _aggregate(all_label_rows)
    _write_csv(output / "label_inference_summary.csv", all_label_rows)
    _write_csv(output / "aggregate_label_summary.csv", label_aggregate)
    _write_csv(output / "per_class_metrics.csv", all_class_rows)
    _write_csv(output / "split_training_history.csv", all_training_rows)
    (output / "run_config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "resolved_output": str(output),
                "resolved_device": str(device),
                "class_names": list(catalog.names),
                "signal_mode": "z_only",
                "collection_rounds": list(collection_rounds),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output / "threat_model_audit.json").write_text(
        json.dumps(
            {
                "attack_model_inputs": ["z"],
                "attack_training_inputs": ["attacker_z"],
                "attack_training_targets": ["attacker_y"],
                "attacker_owned_source_data": ["attacker_x", "attacker_y"],
                "victim_attack_access": ["victim_z"],
                "victim_private_evaluator_only": ["victim_x", "victim_y"],
                "forbidden_attack_inputs": ["u", "dL_du", "dL_dz", "victim_x", "victim_y"],
                "note": (
                    "A server observes z directly. A client-side attacker requires an "
                    "explicit relay/logging exposure to obtain another client's z."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_result_readme(
        output,
        label_aggregate,
        len(args.seeds),
        args.num_victims,
    )


if __name__ == "__main__":
    run(build_parser().parse_args())


__all__ = [
    "ZLabelClassifier",
    "ZLabelClassifierConfig",
    "ZTranscriptAccumulator",
    "ZTranscriptBundle",
    "_temporal_z_bundles",
    "build_parser",
    "run",
]
