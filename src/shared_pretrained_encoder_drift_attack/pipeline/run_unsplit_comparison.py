from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from statistics import mean

import torch
from PIL import Image, ImageDraw
from torch import Tensor, nn
from torch.nn import functional
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms.functional import to_pil_image

from ...decoder.evaluation.reconstruction_metrics import per_sample_metrics
from ...shared.reproducibility.random_seed import seed_everything
from ..unsplit_benchmark import (
    SPECS,
    BenchmarkReconstructionDecoder,
    LayerwiseBenchmarkNet,
    benchmark_spec,
    first_example_per_class_indices,
    load_benchmark_datasets,
    minmax_normalize,
    observe_transcript,
    run_unsplit_attack,
)


CONDITIONS = ("z_only", "gradient_only", "z_gradient")
PAPER_REPORTED_MSE = {"mnist": 0.087, "fashion_mnist": 0.13, "cifar10": 0.0733}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the shared-encoder transcript decoder with the public UnSplit "
            "attack on the same native-resolution benchmark targets."
        )
    )
    parser.add_argument("--dataset", choices=tuple(SPECS), required=True)
    parser.add_argument("--data-root", default="workspace/data/unsplit_benchmark")
    parser.add_argument(
        "--output",
        default="workspace/results/shared_pretrained_encoder_drift_attack/unsplit_comparison",
    )
    parser.add_argument("--split-depth", type=int, default=None)
    parser.add_argument("--pretrain-samples", type=int, default=20_000)
    parser.add_argument("--aux-samples", type=int, default=10_000)
    parser.add_argument("--victim-samples", type=int, default=20_000)
    parser.add_argument("--pretrain-epochs", type=int, default=10)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--victim-epochs", type=int, default=10)
    parser.add_argument("--attack-epochs", type=int, default=20)
    parser.add_argument("--oracle-epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--attack-learning-rate", type=float, default=1e-3)
    parser.add_argument("--target-indices", nargs="*", type=int, default=None)
    parser.add_argument("--max-targets", type=int, default=10)
    parser.add_argument("--unsplit-main-iters", type=int, default=1000)
    parser.add_argument("--unsplit-input-iters", type=int, default=100)
    parser.add_argument("--unsplit-model-iters", type=int, default=100)
    parser.add_argument("--unsplit-learning-rate", type=float, default=1e-3)
    parser.add_argument("--unsplit-lambda-tv", type=float, default=None)
    parser.add_argument("--unsplit-lambda-l2", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-download", action="store_true")
    return parser


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
    )


def _split_training_data(trainset: Dataset, args) -> tuple[Subset, Subset, Subset, Subset]:
    total = args.pretrain_samples + args.aux_samples + args.victim_samples
    if min(args.pretrain_samples, args.aux_samples, args.victim_samples) < 2:
        raise ValueError("pretrain, auxiliary, and victim sample counts must be at least two")
    if total > len(trainset):
        raise ValueError(
            f"requested {total} disjoint training samples but {len(trainset)} are available"
        )
    permutation = torch.randperm(
        len(trainset), generator=torch.Generator().manual_seed(args.seed)
    ).tolist()
    pretrain_end = args.pretrain_samples
    auxiliary_end = pretrain_end + args.aux_samples
    victim_end = auxiliary_end + args.victim_samples
    auxiliary_indices = permutation[pretrain_end:auxiliary_end]
    validation_count = max(1, round(0.1 * len(auxiliary_indices)))
    auxiliary_validation = auxiliary_indices[:validation_count]
    auxiliary_train = auxiliary_indices[validation_count:]
    if not auxiliary_train:
        raise ValueError("auxiliary training partition is empty")
    return (
        Subset(trainset, permutation[:pretrain_end]),
        Subset(trainset, auxiliary_train),
        Subset(trainset, auxiliary_validation),
        Subset(trainset, permutation[auxiliary_end:victim_end]),
    )


def _signal_channels(signal: Tensor) -> int:
    if signal.ndim not in {2, 4}:
        raise ValueError(f"unsupported cut tensor shape: {tuple(signal.shape)}")
    return int(signal.shape[1])


def _decoder(
    sample_z: Tensor,
    channels: int,
    image_size: int,
    condition: str,
    device: torch.device,
) -> BenchmarkReconstructionDecoder:
    return BenchmarkReconstructionDecoder(
        _signal_channels(sample_z), channels, image_size, condition
    ).to(device)


def _reconstruction_loss(reconstruction: Tensor, target: Tensor) -> Tensor:
    return functional.l1_loss(reconstruction, target) + 0.5 * functional.mse_loss(
        reconstruction, target
    )


def _pretrain_encoder(
    model: LayerwiseBenchmarkNet,
    loader: DataLoader,
    split_depth: int,
    channels: int,
    image_size: int,
    epochs: int,
    learning_rate: float,
    device: torch.device,
) -> None:
    sample_images, _ = next(iter(loader))
    with torch.no_grad():
        sample_z = model.forward_to(sample_images[:1].to(device), split_depth)
    head = _decoder(sample_z, channels, image_size, "z_only", device)
    optimizer = torch.optim.AdamW(
        [*model.prefix_parameters(split_depth), *head.parameters()], lr=learning_rate
    )
    for epoch in range(1, epochs + 1):
        model.train()
        head.train()
        total = samples = 0
        for images, _ in loader:
            images = images.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction = head(model.forward_to(images, split_depth), None)
            loss = _reconstruction_loss(reconstruction, images)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(images)
            samples += len(images)
        print(f"Encoder pretrain {epoch:03d}/{epochs:03d}: loss={total/samples:.6f}")


def _train_classifier(
    model: LayerwiseBenchmarkNet,
    loader: DataLoader,
    split_depth: int,
    epochs: int,
    learning_rate: float,
    device: torch.device,
    *,
    frozen_encoder: bool,
    label: str,
) -> None:
    parameters = (
        model.suffix_parameters(split_depth)
        if frozen_encoder
        else list(model.parameters())
    )
    optimizer = torch.optim.Adam(parameters, lr=learning_rate, amsgrad=True)
    for epoch in range(1, epochs + 1):
        model.train()
        total = correct = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            if frozen_encoder:
                with torch.no_grad():
                    z = model.forward_to(images, split_depth)
                logits = model.forward_from(z, split_depth)
            else:
                logits = model(images)
            loss = functional.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            total += len(images)
            correct += int((logits.argmax(1) == labels).sum())
        print(f"{label} {epoch:03d}/{epochs:03d}: accuracy={correct/total:.2%}")


def _classification_accuracy(
    model: LayerwiseBenchmarkNet,
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> float:
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            correct += int((model(images).argmax(1) == labels).sum())
            total += len(images)
    return correct / total


def _validation_loss(
    provider: LayerwiseBenchmarkNet,
    decoder: BenchmarkReconstructionDecoder,
    loader: DataLoader,
    split_depth: int,
    device: torch.device,
) -> float:
    decoder.eval()
    total = samples = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        z, gradient = observe_transcript(provider, images, labels, split_depth)
        with torch.no_grad():
            loss = _reconstruction_loss(decoder(z, gradient), images)
        total += float(loss) * len(images)
        samples += len(images)
    return total / samples


def _train_decoders(
    provider: LayerwiseBenchmarkNet,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    split_depth: int,
    channels: int,
    image_size: int,
    conditions: tuple[str, ...],
    epochs: int,
    learning_rate: float,
    device: torch.device,
) -> dict[str, BenchmarkReconstructionDecoder]:
    sample_images, sample_labels = next(iter(train_loader))
    sample_z, _ = observe_transcript(
        provider,
        sample_images[:1].to(device),
        sample_labels[:1].to(device),
        split_depth,
    )
    decoders = {
        condition: _decoder(sample_z, channels, image_size, condition, device)
        for condition in conditions
    }
    optimizers = {
        name: torch.optim.AdamW(decoder.parameters(), lr=learning_rate)
        for name, decoder in decoders.items()
    }
    best = {
        name: (float("inf"), copy.deepcopy(decoder.state_dict()))
        for name, decoder in decoders.items()
    }
    provider.eval()
    for epoch in range(1, epochs + 1):
        for decoder in decoders.values():
            decoder.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            z, gradient = observe_transcript(provider, images, labels, split_depth)
            for name, decoder in decoders.items():
                optimizers[name].zero_grad(set_to_none=True)
                loss = _reconstruction_loss(decoder(z, gradient), images)
                loss.backward()
                optimizers[name].step()
        status: list[str] = []
        for name, decoder in decoders.items():
            value = _validation_loss(
                provider, decoder, validation_loader, split_depth, device
            )
            status.append(f"{name}={value:.5f}")
            if value < best[name][0]:
                best[name] = (value, copy.deepcopy(decoder.state_dict()))
        print(f"Decoder {epoch:03d}/{epochs:03d}: " + ", ".join(status))
    for name, decoder in decoders.items():
        decoder.load_state_dict(best[name][1])
        decoder.eval()
    return decoders


def _target_batch(
    testset: Dataset,
    indices: list[int],
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    samples = [testset[index] for index in indices]
    images = torch.stack([sample[0] for sample in samples]).to(device)
    labels = torch.tensor([int(sample[1]) for sample in samples], device=device)
    return images, labels


def _decoder_reconstructions(
    model: LayerwiseBenchmarkNet,
    decoders: dict[str, BenchmarkReconstructionDecoder],
    images: Tensor,
    labels: Tensor,
    split_depth: int,
) -> dict[str, Tensor]:
    z, gradient = observe_transcript(model, images, labels, split_depth)
    with torch.no_grad():
        return {name: decoder(z, gradient) for name, decoder in decoders.items()}


def _save_images(
    output: Path,
    indices: list[int],
    labels: Tensor,
    references: Tensor,
    reconstructions: dict[str, Tensor],
) -> None:
    for method, images in {"reference": references, **reconstructions}.items():
        root = output / "images" / method
        root.mkdir(parents=True, exist_ok=True)
        for order, (index, label, image) in enumerate(zip(indices, labels, images)):
            to_pil_image(image.detach().cpu().clamp(0, 1)).save(
                root / f"{order:02d}_index_{index}_label_{int(label)}.png"
            )


def _save_grid(
    path: Path,
    references: Tensor,
    labels: Tensor,
    class_names: tuple[str, ...],
    reconstructions: dict[str, Tensor],
) -> None:
    methods = ["reference", *reconstructions]
    images_by_method = {"reference": references, **reconstructions}
    cell = 128
    header = 36
    label_width = 115
    row_height = cell + 6
    canvas = Image.new(
        "RGB", (label_width + cell * len(methods), header + row_height * len(references)), "white"
    )
    draw = ImageDraw.Draw(canvas)
    for column, method in enumerate(methods):
        draw.text((label_width + column * cell + 4, 8), method, fill="black")
    for row, label in enumerate(labels.tolist()):
        y = header + row * row_height
        draw.text((4, y + cell // 2 - 7), f"{label}: {class_names[label]}", fill="black")
        for column, method in enumerate(methods):
            image = to_pil_image(images_by_method[method][row].detach().cpu().clamp(0, 1))
            resampling = Image.Resampling.NEAREST if image.mode == "L" else Image.Resampling.BILINEAR
            image = image.convert("RGB").resize((cell, cell), resampling)
            canvas.paste(image, (label_width + column * cell, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _metric_rows(
    dataset: str,
    split_depth: int,
    stage: str,
    indices: list[int],
    labels: Tensor,
    class_names: tuple[str, ...],
    references: Tensor,
    reconstructions: dict[str, Tensor],
    runtimes: dict[str, list[float]] | None = None,
    feature_mses: dict[str, list[float]] | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method, reconstructed in reconstructions.items():
        values = per_sample_metrics(reconstructed, references)
        for order, index in enumerate(indices):
            label = int(labels[order])
            rows.append(
                {
                    "dataset": dataset,
                    "split_depth": split_depth,
                    "stage": stage,
                    "method": method,
                    "order": order,
                    "dataset_index": index,
                    "label": label,
                    "class_name": class_names[label],
                    "mse": float(values["mse"][order]),
                    "mae": float(values["mae"][order]),
                    "psnr": float(values["psnr"][order]),
                    "ssim": float(values["ssim"][order]),
                    "runtime_seconds": (
                        runtimes[method][order]
                        if runtimes is not None and method in runtimes
                        else ""
                    ),
                    "feature_mse": (
                        feature_mses[method][order]
                        if feature_mses is not None and method in feature_mses
                        else ""
                    ),
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["stage"]), str(row["method"])), []).append(row)
    result: list[dict[str, object]] = []
    for (stage, method), group in groups.items():
        runtime_values = [float(row["runtime_seconds"]) for row in group if row["runtime_seconds"] != ""]
        result.append(
            {
                "stage": stage,
                "method": method,
                "samples": len(group),
                "mse": mean(float(row["mse"]) for row in group),
                "mae": mean(float(row["mae"]) for row in group),
                "psnr": mean(float(row["psnr"]) for row in group),
                "ssim": mean(float(row["ssim"]) for row in group),
                "mean_runtime_seconds": mean(runtime_values) if runtime_values else "",
            }
        )
    return sorted(result, key=lambda row: (str(row["stage"]), str(row["method"])))


def run(args: argparse.Namespace) -> Path:
    seed_everything(args.seed)
    device = _device(args.device)
    spec = benchmark_spec(args.dataset)
    split_depth = spec.default_split_depth if args.split_depth is None else args.split_depth
    if not 1 <= split_depth <= spec.max_split_depth:
        raise ValueError(
            f"{args.dataset} split depth must be between 1 and {spec.max_split_depth}"
        )
    trainset, testset = load_benchmark_datasets(
        args.dataset, args.data_root, download=not args.no_download
    )
    pretrain_set, auxiliary_train, auxiliary_validation, victim_set = _split_training_data(
        trainset, args
    )
    loaders = {
        "pretrain": _loader(pretrain_set, args.batch_size, True, args.num_workers, args.seed + 1),
        "aux_train": _loader(auxiliary_train, args.batch_size, True, args.num_workers, args.seed + 2),
        "aux_validation": _loader(auxiliary_validation, args.batch_size, False, args.num_workers, args.seed + 3),
        "victim": _loader(victim_set, args.batch_size, True, args.num_workers, args.seed + 4),
    }
    model = LayerwiseBenchmarkNet(args.dataset).to(device)
    _pretrain_encoder(
        model,
        loaders["pretrain"],
        split_depth,
        spec.channels,
        spec.image_size,
        args.pretrain_epochs,
        args.learning_rate,
        device,
    )
    _train_classifier(
        model,
        loaders["victim"],
        split_depth,
        args.warmup_epochs,
        args.learning_rate,
        device,
        frozen_encoder=True,
        label="Classifier warmup",
    )
    epoch0_accuracy = _classification_accuracy(
        model, testset, args.batch_size, args.num_workers, device
    )
    epoch0_state = copy.deepcopy(model.state_dict())
    print("Training fixed shared-encoder transcript decoders...")
    decoders = _train_decoders(
        model,
        loaders["aux_train"],
        loaders["aux_validation"],
        split_depth,
        spec.channels,
        spec.image_size,
        CONDITIONS,
        args.attack_epochs,
        args.attack_learning_rate,
        device,
    )
    target_indices = (
        first_example_per_class_indices(testset)
        if not args.target_indices
        else list(args.target_indices)
    )[: args.max_targets]
    if not target_indices:
        raise ValueError("at least one target image is required")
    if min(target_indices) < 0 or max(target_indices) >= len(testset):
        raise ValueError("target index is outside the test set")
    targets, target_labels = _target_batch(testset, target_indices, device)
    epoch0_z, _ = observe_transcript(model, targets, target_labels, split_depth)
    epoch0_reconstructions = {
        f"ours_{name}": value
        for name, value in _decoder_reconstructions(
            model, decoders, targets, target_labels, split_depth
        ).items()
    }
    _train_classifier(
        model,
        loaders["victim"],
        split_depth,
        args.victim_epochs,
        args.learning_rate,
        device,
        frozen_encoder=False,
        label="Victim fine-tune",
    )
    final_accuracy = _classification_accuracy(
        model, testset, args.batch_size, args.num_workers, device
    )
    final_z_for_drift, _ = observe_transcript(model, targets, target_labels, split_depth)
    final_reconstructions = {
        f"ours_{name}": value
        for name, value in _decoder_reconstructions(
            model, decoders, targets, target_labels, split_depth
        ).items()
    }
    trained_decoders = dict(decoders)
    if args.oracle_epochs > 0:
        print("Training final victim-matched z-only oracle...")
        oracle = _train_decoders(
            model,
            loaders["aux_train"],
            loaders["aux_validation"],
            split_depth,
            spec.channels,
            spec.image_size,
            ("z_only",),
            args.oracle_epochs,
            args.attack_learning_rate,
            device,
        )["z_only"]
        trained_decoders["oracle_z_only"] = oracle
        final_reconstructions["ours_oracle_z_only"] = _decoder_reconstructions(
            model, {"oracle": oracle}, targets, target_labels, split_depth
        )["oracle"]

    print("Running UnSplit on the same final victim targets...")
    clone = LayerwiseBenchmarkNet(args.dataset).to(device)
    final_z, _ = observe_transcript(model, targets, target_labels, split_depth)
    unsplit_images: list[Tensor] = []
    unsplit_runtimes: list[float] = []
    unsplit_feature_mses: list[float] = []
    lambda_tv = (
        args.unsplit_lambda_tv
        if args.unsplit_lambda_tv is not None
        else (0.1 if split_depth <= 3 else 1.0)
    )
    for order in range(len(targets)):
        result = run_unsplit_attack(
            clone,
            split_depth,
            final_z[order : order + 1],
            targets[order : order + 1].shape,
            main_iters=args.unsplit_main_iters,
            input_iters=args.unsplit_input_iters,
            model_iters=args.unsplit_model_iters,
            learning_rate=args.unsplit_learning_rate,
            lambda_tv=lambda_tv,
            lambda_l2=args.unsplit_lambda_l2,
        )
        reconstruction = result.reconstruction
        if args.dataset == "cifar10":
            reconstruction = minmax_normalize(reconstruction)
        unsplit_images.append(reconstruction[0])
        unsplit_runtimes.append(result.runtime_seconds)
        unsplit_feature_mses.append(result.feature_mse)
        print(
            f"UnSplit target {order + 1:02d}/{len(targets):02d}: "
            f"runtime={result.runtime_seconds:.2f}s feature_mse={result.feature_mse:.6g}"
        )
    final_reconstructions["unsplit"] = torch.stack(unsplit_images)

    output = Path(args.output) / args.dataset / f"split_{split_depth}"
    output.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for order, (index, label) in enumerate(zip(target_indices, target_labels.tolist())):
        manifest_rows.append(
            {
                "order": order,
                "dataset": args.dataset,
                "split": "test",
                "dataset_index": index,
                "label": label,
                "class_name": spec.class_names[label],
                "selection": (
                    "official_demo_first_per_class"
                    if not args.target_indices
                    else "explicit_cli_indices"
                ),
            }
        )
    _write_csv(output / "target_manifest.csv", manifest_rows)
    metric_rows = _metric_rows(
        args.dataset,
        split_depth,
        "epoch_0",
        target_indices,
        target_labels,
        spec.class_names,
        targets,
        epoch0_reconstructions,
    )
    metric_rows.extend(
        _metric_rows(
            args.dataset,
            split_depth,
            "final",
            target_indices,
            target_labels,
            spec.class_names,
            targets,
            final_reconstructions,
            runtimes={"unsplit": unsplit_runtimes},
            feature_mses={"unsplit": unsplit_feature_mses},
        )
    )
    _write_csv(output / "per_sample_metrics.csv", metric_rows)
    _write_csv(output / "summary.csv", _summary(metric_rows))
    _write_csv(
        output / "paper_reference.csv",
        [
            {
                "dataset": args.dataset,
                "method": "UnSplit",
                "reported_mse_after_training": PAPER_REPORTED_MSE[args.dataset],
                "scope": "mean_over_split_depths_not_directly_comparable_to_single_run",
                "source": "https://arxiv.org/abs/2108.09033",
            }
        ],
    )
    flattened_epoch0 = epoch0_z.flatten(1)
    flattened_final = final_z_for_drift.flatten(1)
    drift_rows = [
        {
            "dataset": args.dataset,
            "split_depth": split_depth,
            "epoch0_test_accuracy": epoch0_accuracy,
            "final_test_accuracy": final_accuracy,
            "target_representation_cosine": float(
                functional.cosine_similarity(
                    flattened_epoch0, flattened_final, dim=1
                ).mean()
            ),
            "target_representation_l2": float(
                (flattened_final - flattened_epoch0).norm(dim=1).mean()
            ),
        }
    ]
    _write_csv(output / "victim_and_drift_metrics.csv", drift_rows)
    _save_images(output / "epoch_0", target_indices, target_labels, targets, epoch0_reconstructions)
    _save_images(output / "final", target_indices, target_labels, targets, final_reconstructions)
    _save_grid(
        output / "epoch_0" / "comparison_grid.png",
        targets,
        target_labels,
        spec.class_names,
        epoch0_reconstructions,
    )
    _save_grid(
        output / "final" / "comparison_grid.png",
        targets,
        target_labels,
        spec.class_names,
        final_reconstructions,
    )
    checkpoint_root = output / "checkpoints"
    checkpoint_root.mkdir(exist_ok=True)
    torch.save(
        {
            "dataset": args.dataset,
            "split_depth": split_depth,
            "epoch0_model": epoch0_state,
            "final_model": model.state_dict(),
        },
        checkpoint_root / "victim.pt",
    )
    for name, decoder in trained_decoders.items():
        torch.save(decoder.state_dict(), checkpoint_root / f"{name}.pt")
    run_config = vars(args) | {
        "resolved_device": str(device),
        "resolved_split_depth": split_depth,
        "image_size": spec.image_size,
        "channels": spec.channels,
        "target_selection": (
            "official_demo_first_per_class"
            if not args.target_indices
            else "explicit_cli_indices"
        ),
        "paper_reported_after_train_mse_mean_over_split_depths": PAPER_REPORTED_MSE[
            args.dataset
        ],
        "epoch0_test_accuracy": epoch0_accuracy,
        "final_test_accuracy": final_accuracy,
        "comparison_warning": (
            "The paper number is averaged over split depths; this run uses one split. "
            "Our decoder uses auxiliary data while UnSplit is data-oblivious."
        ),
    }
    (output / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )
    print(f"Results written to {output}")
    return output


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
