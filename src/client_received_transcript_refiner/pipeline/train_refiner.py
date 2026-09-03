from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from ...client_received_transcript_attack.data.dataset import (
    ClientReceivedTranscriptDataset,
)
from ...shared.reproducibility.random_seed import seed_everything
from ..models.refiner import ResidualUNetRefiner, ResidualUNetRefinerConfig
from ..training.trainer import RefinerTrainingConfig, train_refiner
from .common import (
    device_from_name,
    file_sha256,
    load_frozen_decoder,
    require_disjoint_datasets,
    transcript_ids,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a residual image refiner on public auxiliary transcripts. "
            "The coarse decoder is loaded and frozen; no Split Learning checkpoint "
            "is loaded."
        )
    )
    parser.add_argument("--decoder-checkpoint", required=True)
    parser.add_argument("--train-attacker-manifest", required=True)
    parser.add_argument("--train-target-manifest", required=True)
    parser.add_argument("--validation-attacker-manifest", required=True)
    parser.add_argument("--validation-target-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--base-channels", type=int, default=32)
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
    decoder = load_frozen_decoder(args.decoder_checkpoint, device)
    train_dataset = ClientReceivedTranscriptDataset(
        args.train_attacker_manifest, args.train_target_manifest
    )
    validation_dataset = ClientReceivedTranscriptDataset(
        args.validation_attacker_manifest, args.validation_target_manifest
    )
    require_disjoint_datasets(
        train_dataset, validation_dataset, "refiner train/validation datasets"
    )

    target = train_dataset[0]["target_image"]
    image_channels = int(target.shape[0])
    refiner_config = ResidualUNetRefinerConfig(
        image_channels=image_channels,
        base_channels=args.base_channels,
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
    decoder_checkpoint = Path(args.decoder_checkpoint).resolve()
    metadata: dict[str, object] = {
        "decoder_checkpoint": str(decoder_checkpoint),
        "decoder_sha256": file_sha256(decoder_checkpoint),
        "train_attacker_manifest": str(Path(args.train_attacker_manifest).resolve()),
        "train_target_manifest": str(Path(args.train_target_manifest).resolve()),
        "validation_attacker_manifest": str(
            Path(args.validation_attacker_manifest).resolve()
        ),
        "validation_target_manifest": str(
            Path(args.validation_target_manifest).resolve()
        ),
        "train_transcript_ids": sorted(transcript_ids(train_dataset)),
        "validation_transcript_ids": sorted(transcript_ids(validation_dataset)),
        "public_auxiliary_targets_only": True,
        "loaded_split_learning_checkpoint": False,
        "coarse_decoder_frozen": True,
        "refiner_input": ["coarse_reconstruction"],
    }
    refiner = ResidualUNetRefiner(refiner_config).to(device)
    checkpoint, _ = train_refiner(
        refiner,
        decoder,
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
                "refiner_config": refiner_config.to_dict(),
                "training_config": asdict(training_config),
                "refiner_checkpoint": str(checkpoint),
                "decoder_sha256": metadata["decoder_sha256"],
                "public_auxiliary_targets_only": True,
                "loaded_split_learning_checkpoint": False,
                "coarse_decoder_frozen": True,
                "holdout_used_for_training": False,
            },
            handle,
            indent=2,
            default=str,
        )
    print(f"Refiner checkpoint: {checkpoint.resolve()}")
    return checkpoint


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()


__all__ = ["build_parser", "run"]
