from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import torch

from ...client_received_transcript_attack.data.dataset import (
    ClientReceivedTranscriptDataset,
)
from ...client_received_transcript_attack.evaluation.evaluator import (
    evaluate_client_received_decoder,
)
from ...client_received_transcript_attack.models.decoder import (
    ClientReceivedDecoder,
    ClientReceivedDecoderConfig,
    ZUGradZDecoder,
)
from ...client_received_transcript_attack.training.trainer import (
    AttackTrainingConfig,
    train_client_received_decoder,
)
from ...shared.reproducibility.random_seed import seed_everything
from ..conditions import SignalCondition, condition_from_name
from ..data import ConditionedTranscriptDataset
from ..evaluation import add_label_report
from ..statistics import write_comparisons


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train A/B/C/D transcript ablations on one auxiliary client and "
            "evaluate the unchanged attacks on one or more victim clients."
        )
    )
    parser.add_argument("--train-attacker-manifest", required=True)
    parser.add_argument("--train-target-manifest", required=True)
    parser.add_argument("--validation-attacker-manifest", required=True)
    parser.add_argument("--validation-target-manifest", required=True)
    parser.add_argument(
        "--victim-manifests",
        nargs=3,
        action="append",
        metavar=("CLIENT", "ATTACKER_CSV", "EVALUATOR_CSV"),
        required=True,
        help="Repeat for each client evaluated with the same trained attack.",
    )
    parser.add_argument("--class-names", nargs="+", required=True)
    parser.add_argument("--conditions", nargs="+", default=["A", "B", "C"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--classification-weight", type=float, default=0.1)
    parser.add_argument("--disable-label-head", action="store_true")
    parser.add_argument("--signal-spatial-size", type=int, default=16)
    parser.add_argument("--signal-channels", type=int, default=64)
    parser.add_argument("--decoder-base-channels", type=int, default=256)
    parser.add_argument("--decoder-min-channels", type=int, default=32)
    parser.add_argument("--refinement-blocks", type=int, default=1)
    parser.add_argument("--l1-weight", type=float, default=1.0)
    parser.add_argument("--ssim-weight", type=float, default=0.75)
    parser.add_argument("--edge-weight", type=float, default=0.1)
    parser.add_argument("--perceptual-weight", type=float, default=0.1)
    parser.add_argument("--laplacian-weight", type=float, default=0.25)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--max-grid-images", type=int, default=20)
    parser.add_argument("--save-separate-images", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _unique_conditions(values: list[str]) -> list[SignalCondition]:
    result: list[SignalCondition] = []
    seen: set[str] = set()
    for value in values:
        condition = condition_from_name(value)
        if condition.code not in seen:
            result.append(condition)
            seen.add(condition.code)
    return result


def _safe_client_name(value: str) -> str:
    if not value or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in value):
        raise ValueError(
            f"invalid client name {value!r}; use only letters, digits, '_' and '-'"
        )
    return value


def _model(
    condition: SignalCondition,
    sample: dict,
    args: argparse.Namespace,
    device: torch.device,
):
    u = sample["server_output_u"]
    grad_z = sample["grad_g_to_f"]
    if not isinstance(u, torch.Tensor) or not isinstance(grad_z, torch.Tensor):
        raise TypeError("transcript observations must be tensors")
    use_label_head = condition.use_grad_z and not args.disable_label_head
    config = ClientReceivedDecoderConfig(
        u_channels=int(u.shape[0]),
        grad_z_channels=int(grad_z.shape[0]),
        z_channels=(
            int(sample["smashed_z"].shape[0]) if condition.use_z else None
        ),
        num_classes=len(args.class_names),
        image_size=args.image_size,
        signal_spatial_size=args.signal_spatial_size,
        signal_channels=args.signal_channels,
        decoder_base_channels=args.decoder_base_channels,
        decoder_min_channels=args.decoder_min_channels,
        refinement_blocks=args.refinement_blocks,
        use_label_head=use_label_head,
    )
    if condition.use_z:
        return ZUGradZDecoder(config).to(device), config
    return ClientReceivedDecoder(config).to(device), config


def _training_config(args: argparse.Namespace) -> AttackTrainingConfig:
    return AttackTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        l1_weight=args.l1_weight,
        ssim_weight=args.ssim_weight,
        edge_weight=args.edge_weight,
        perceptual_weight=args.perceptual_weight,
        laplacian_weight=args.laplacian_weight,
        classification_weight=args.classification_weight,
        gradient_clip_norm=args.gradient_clip_norm,
        num_workers=args.num_workers,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
    )


def _annotate_checkpoint(path: Path, condition: SignalCondition) -> None:
    state = torch.load(path, map_location="cpu", weights_only=False)
    state["signal_condition"] = condition.code
    state["visible_signals"] = list(condition.visible_signals)
    state["omitted_signals_are_zero_masked"] = True
    torch.save(state, path)


def run(args: argparse.Namespace) -> Path:
    device = _device(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    conditions = _unique_conditions(args.conditions)
    victims = [
        (_safe_client_name(name), Path(attacker), Path(evaluator))
        for name, attacker, evaluator in args.victim_manifests
    ]
    if len({name for name, _, _ in victims}) != len(victims):
        raise ValueError("victim client names must be unique")

    public_train = ClientReceivedTranscriptDataset(
        args.train_attacker_manifest, args.train_target_manifest
    )
    public_validation = ClientReceivedTranscriptDataset(
        args.validation_attacker_manifest, args.validation_target_manifest
    )
    training_config = _training_config(args)
    summary_rows: list[dict[str, object]] = []
    metrics_by_client: dict[str, dict[str, Path]] = {name: {} for name, _, _ in victims}

    for condition in conditions:
        print(
            f"[{condition.code}] Training with {', '.join(condition.visible_signals)}...",
            flush=True,
        )
        train_dataset = ConditionedTranscriptDataset(public_train, condition)
        validation_dataset = ConditionedTranscriptDataset(public_validation, condition)
        # Reset before every condition so A/B/C start from comparable initialization.
        seed_everything(args.seed)
        model, model_config = _model(condition, train_dataset[0], args, device)
        condition_root = output / "conditions" / condition.code
        checkpoint, _ = train_client_received_decoder(
            model,
            train_dataset,
            validation_dataset,
            condition_root / "checkpoints",
            training_config,
            device,
        )
        _annotate_checkpoint(checkpoint, condition)

        for client_name, attacker_manifest, evaluator_manifest in victims:
            print(f"[{condition.code}] Evaluating client {client_name}...", flush=True)
            victim_base = ClientReceivedTranscriptDataset(
                attacker_manifest, evaluator_manifest
            )
            victim_dataset = ConditionedTranscriptDataset(victim_base, condition)
            evaluation_root = condition_root / "clients" / client_name
            summary = evaluate_client_received_decoder(
                model,
                victim_dataset,
                evaluation_root,
                device,
                tuple(args.class_names),
                batch_size=args.batch_size,
                max_grid_images=args.max_grid_images,
                save_separate_images=args.save_separate_images,
            )
            label_report = add_label_report(
                evaluation_root / "reconstruction_metrics.csv",
                evaluation_root,
                tuple(args.class_names),
            )
            if label_report is not None:
                summary["macro_f1"] = label_report["macro_f1"]
            summary_rows.append(
                {
                    "condition": condition.code,
                    "condition_name": condition.name,
                    "client": client_name,
                    "visible_signals": "+".join(condition.visible_signals),
                    **summary,
                }
            )
            metrics_by_client[client_name][condition.code] = (
                evaluation_root / "reconstruction_metrics.csv"
            )

        with (condition_root / "condition_config.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(
                {
                    "condition": condition.code,
                    "condition_name": condition.name,
                    "visible_signals": list(condition.visible_signals),
                    "decoder_config": model_config.to_dict(),
                    "training_config": asdict(training_config),
                    "checkpoint": str(checkpoint),
                    "victim_clients": [name for name, _, _ in victims],
                },
                handle,
                indent=2,
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report_root = output / "reports"
    report_root.mkdir(parents=True, exist_ok=True)
    if summary_rows:
        fieldnames: list[str] = []
        for row in summary_rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with (report_root / "transfer_matrix.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(summary_rows)
    for client_name, condition_csvs in metrics_by_client.items():
        write_comparisons(
            condition_csvs,
            report_root / client_name,
            samples=args.bootstrap_samples,
            seed=args.seed,
        )

    with (output / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                **vars(args),
                "conditions": [condition.code for condition in conditions],
                "condition_signals": {
                    condition.code: list(condition.visible_signals)
                    for condition in conditions
                },
                "resolved_device": str(device),
                "public_training_samples": len(public_train),
                "public_validation_samples": len(public_validation),
                "attacker_loaded_split_learning_checkpoint": False,
                "victim_targets_used_for_training": False,
                "comparison_design": "paired by transcript_id within each client",
            },
            handle,
            indent=2,
            default=str,
        )
    print(f"Ablation and transfer reports: {output.resolve()}")
    return output


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()


__all__ = ["build_parser", "run"]
