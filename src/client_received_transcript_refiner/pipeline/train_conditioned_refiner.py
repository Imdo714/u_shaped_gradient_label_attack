from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from ...client_received_transcript_attack.models.decoder import (
    ClientReceivedDecoderConfig,
)
from ...shared.reproducibility.random_seed import seed_everything
from ..data.materialized_dataset import MaterializedCoarseTranscriptDataset
from ..models.refiner import (
    TranscriptConditionedRefinerConfig,
    TranscriptConditionedResidualUNetRefiner,
)
from ..training.trainer import RefinerTrainingConfig, train_conditioned_refiner
from .common import (
    device_from_name,
    file_sha256,
    require_disjoint_datasets,
    transcript_ids,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train an OOF transcript-conditioned refiner from coarse RGB, u, and dL/dz."
        )
    )
    parser.add_argument("--inference-decoder-checkpoint", required=True)
    parser.add_argument("--train-attacker-manifest", required=True)
    parser.add_argument("--train-target-manifest", required=True)
    parser.add_argument("--train-coarse-manifest", required=True)
    parser.add_argument("--validation-attacker-manifest", required=True)
    parser.add_argument("--validation-target-manifest", required=True)
    parser.add_argument("--validation-coarse-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--condition-channels", type=int, default=32)
    parser.add_argument("--condition-spatial-size", type=int, default=16)
    parser.add_argument("--bottleneck-blocks", type=int, default=2)
    parser.add_argument("--max-residual", type=float, default=0.1)
    parser.add_argument("--l1-weight", type=float, default=1.0)
    parser.add_argument("--ssim-weight", type=float, default=0.75)
    parser.add_argument("--edge-weight", type=float, default=0.2)
    parser.add_argument("--perceptual-weight", type=float, default=0.1)
    parser.add_argument("--residual-weight", type=float, default=0.05)
    parser.add_argument("--low-frequency-weight", type=float, default=0.05)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser


def run(args: argparse.Namespace) -> Path:
    seed_everything(args.seed)
    device = device_from_name(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    train_dataset = MaterializedCoarseTranscriptDataset(
        args.train_attacker_manifest,
        args.train_target_manifest,
        args.train_coarse_manifest,
    )
    validation_dataset = MaterializedCoarseTranscriptDataset(
        args.validation_attacker_manifest,
        args.validation_target_manifest,
        args.validation_coarse_manifest,
    )
    require_disjoint_datasets(
        train_dataset, validation_dataset, "conditioned refiner train/validation"
    )
    invalid_folds = [
        str(row.get("fold", ""))
        for row in train_dataset.rows
        if not str(row.get("fold", "")).startswith("fold_")
    ]
    if invalid_folds:
        raise ValueError(
            "train coarse manifest must contain OOF fold assignments created by "
            "prepare_oof_data"
        )

    decoder_checkpoint = torch.load(
        args.inference_decoder_checkpoint, map_location="cpu", weights_only=False
    )
    decoder_config = ClientReceivedDecoderConfig(
        **decoder_checkpoint["decoder_config"]
    )
    sample = train_dataset[0]
    u = sample["server_output_u"]
    grad_z = sample["grad_g_to_f"]
    target = sample["target_image"]
    if not all(isinstance(value, torch.Tensor) for value in (u, grad_z, target)):
        raise TypeError("conditioned refiner inputs must be tensors")
    if int(u.shape[0]) != decoder_config.u_channels:
        raise ValueError("u channel count does not match inference decoder")
    if int(grad_z.shape[0]) != decoder_config.grad_z_channels:
        raise ValueError("dL/dz channel count does not match inference decoder")

    refiner_config = TranscriptConditionedRefinerConfig(
        u_channels=int(u.shape[0]),
        grad_z_channels=int(grad_z.shape[0]),
        image_channels=int(target.shape[0]),
        base_channels=args.base_channels,
        condition_channels=args.condition_channels,
        condition_spatial_size=args.condition_spatial_size,
        bottleneck_blocks=args.bottleneck_blocks,
        max_residual=args.max_residual,
    )
    training_config = RefinerTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        l1_weight=args.l1_weight,
        ssim_weight=args.ssim_weight,
        edge_weight=args.edge_weight,
        perceptual_weight=args.perceptual_weight,
        residual_weight=args.residual_weight,
        low_frequency_weight=args.low_frequency_weight,
        gradient_clip_norm=args.gradient_clip_norm,
        num_workers=args.num_workers,
    )
    inference_decoder_path = Path(args.inference_decoder_checkpoint).resolve()
    metadata: dict[str, object] = {
        "decoder_checkpoint": str(inference_decoder_path),
        "decoder_sha256": file_sha256(inference_decoder_path),
        "train_transcript_ids": sorted(transcript_ids(train_dataset)),
        "validation_transcript_ids": sorted(transcript_ids(validation_dataset)),
        "train_coarse_manifest": str(Path(args.train_coarse_manifest).resolve()),
        "validation_coarse_manifest": str(
            Path(args.validation_coarse_manifest).resolve()
        ),
        "oof_coarse_training": True,
        "public_auxiliary_targets_only": True,
        "loaded_split_learning_checkpoint": False,
        "refiner_inputs": ["coarse_reconstruction", "u", "dL/dz"],
    }
    refiner = TranscriptConditionedResidualUNetRefiner(refiner_config).to(device)
    checkpoint, history = train_conditioned_refiner(
        refiner,
        train_dataset,
        validation_dataset,
        output / "checkpoints",
        training_config,
        device,
        checkpoint_metadata=metadata,
    )
    with (output / "training_run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                **vars(args),
                "device": str(device),
                "refiner_type": refiner.refiner_type,
                "refiner_config": refiner_config.to_dict(),
                "training_config": asdict(training_config),
                "refiner_checkpoint": str(checkpoint),
                "best_epoch": min(
                    history,
                    key=lambda row: float(row["validation_loss"]),
                )["epoch"],
                "decoder_sha256": metadata["decoder_sha256"],
                "oof_coarse_training": True,
                "public_auxiliary_targets_only": True,
                "loaded_split_learning_checkpoint": False,
                "holdout_used_for_training": False,
            },
            handle,
            indent=2,
            default=str,
        )
    print(f"Conditioned refiner checkpoint: {checkpoint.resolve()}")
    return checkpoint


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()


__all__ = ["build_parser", "run"]
