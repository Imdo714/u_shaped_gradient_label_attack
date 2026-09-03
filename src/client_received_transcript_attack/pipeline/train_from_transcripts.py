from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from ...shared.reproducibility.random_seed import seed_everything
from ..data.dataset import ClientReceivedTranscriptDataset
from ..models.decoder import (
    ClientReceivedDecoder,
    ClientReceivedDecoderConfig,
    ZUGradZDecoder,
)
from ..models.multiscale_decoder import (
    MultiscaleDecoderConfig,
    MultiscalePixelShuffleDecoder,
)
from ..models.residual_detail_decoder import (
    ResidualDetailDecoder,
    ResidualDetailDecoderConfig,
)
from ..training.trainer import AttackTrainingConfig, train_client_received_decoder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the attack decoder from public u/dL-dz manifests without loading "
            "any Split Learning checkpoint."
        )
    )
    parser.add_argument("--train-attacker-manifest", required=True)
    parser.add_argument("--train-target-manifest", required=True)
    parser.add_argument("--validation-attacker-manifest", required=True)
    parser.add_argument("--validation-target-manifest", required=True)
    parser.add_argument("--class-names", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument(
        "--decoder-architecture",
        choices=(
            "baseline_bilinear",
            "multiscale_pixelshuffle",
            "residual_detail",
            "z_u_grad_z_bilinear",
        ),
        default="baseline_bilinear",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--classification-weight", type=float, default=0.1)
    parser.add_argument("--use-label-head", action="store_true")
    parser.add_argument("--signal-spatial-size", type=int, default=16)
    parser.add_argument("--signal-channels", type=int, default=64)
    parser.add_argument("--decoder-base-channels", type=int, default=256)
    parser.add_argument("--decoder-min-channels", type=int, default=32)
    parser.add_argument("--refinement-blocks", type=int, default=1)
    parser.add_argument("--film-strength", type=float, default=0.1)
    parser.add_argument("--detail-condition-channels", type=int, default=8)
    parser.add_argument("--detail-channels", type=int, default=16)
    parser.add_argument("--detail-scale", type=float, default=0.25)
    parser.add_argument("--initialize-from-decoder-checkpoint", default=None)
    parser.add_argument("--l1-weight", type=float, default=1.0)
    parser.add_argument("--ssim-weight", type=float, default=0.75)
    parser.add_argument("--edge-weight", type=float, default=0.15)
    parser.add_argument("--perceptual-weight", type=float, default=0.25)
    parser.add_argument("--laplacian-weight", type=float, default=0.0)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--preserve-initial-state", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def run(args: argparse.Namespace) -> Path:
    seed_everything(args.seed)
    device = _device(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    train_dataset = ClientReceivedTranscriptDataset(
        args.train_attacker_manifest, args.train_target_manifest
    )
    validation_dataset = ClientReceivedTranscriptDataset(
        args.validation_attacker_manifest, args.validation_target_manifest
    )
    sample = train_dataset[0]
    u = sample["server_output_u"]
    grad_z = sample["grad_g_to_f"]
    if not isinstance(u, torch.Tensor) or not isinstance(grad_z, torch.Tensor):
        raise TypeError("public transcript signals must be tensors")
    common_config = {
        "u_channels": int(u.shape[0]),
        "grad_z_channels": int(grad_z.shape[0]),
        "num_classes": len(args.class_names),
        "image_size": args.image_size,
        "signal_spatial_size": args.signal_spatial_size,
        "signal_channels": args.signal_channels,
        "decoder_base_channels": args.decoder_base_channels,
        "decoder_min_channels": args.decoder_min_channels,
        "refinement_blocks": args.refinement_blocks,
        "use_label_head": args.use_label_head,
    }
    if args.decoder_architecture == "baseline_bilinear":
        model_config = ClientReceivedDecoderConfig(**common_config)
        decoder = ClientReceivedDecoder(model_config).to(device)
    elif args.decoder_architecture == "multiscale_pixelshuffle":
        model_config = MultiscaleDecoderConfig(
            **common_config,
            film_strength=args.film_strength,
        )
        decoder = MultiscalePixelShuffleDecoder(model_config).to(device)
    elif args.decoder_architecture == "residual_detail":
        model_config = ResidualDetailDecoderConfig(
            **common_config,
            detail_condition_channels=args.detail_condition_channels,
            detail_channels=args.detail_channels,
            detail_scale=args.detail_scale,
        )
        decoder = ResidualDetailDecoder(model_config).to(device)
    else:
        smashed_z = sample.get("smashed_z")
        if not isinstance(smashed_z, torch.Tensor):
            raise ValueError(
                "z_u_grad_z_bilinear requires manifests captured with --capture-z"
            )
        model_config = ClientReceivedDecoderConfig(
            **common_config,
            z_channels=int(smashed_z.shape[0]),
        )
        decoder = ZUGradZDecoder(model_config).to(device)

    initialization_summary: dict[str, object] | None = None
    if args.initialize_from_decoder_checkpoint:
        if args.decoder_architecture != "residual_detail":
            raise ValueError(
                "--initialize-from-decoder-checkpoint is only supported by "
                "residual_detail"
            )
        initialization_path = Path(args.initialize_from_decoder_checkpoint)
        state = torch.load(initialization_path, map_location=device, weights_only=False)
        source_type = str(state.get("decoder_type", "baseline_bilinear"))
        if source_type != "baseline_bilinear":
            raise ValueError(
                "residual_detail initialization requires a baseline_bilinear checkpoint"
            )
        incompatible = decoder.load_state_dict(state["model"], strict=False)
        allowed_prefixes = (
            "u_detail_projection.",
            "grad_z_detail_projection.",
            "detail_head.",
        )
        unexpected_missing = [
            key
            for key in incompatible.missing_keys
            if not key.startswith(allowed_prefixes)
        ]
        if incompatible.unexpected_keys or unexpected_missing:
            raise ValueError(
                "baseline checkpoint is not compatible with residual_detail: "
                f"missing={unexpected_missing}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        initialization_summary = {
            "checkpoint": str(initialization_path),
            "source_decoder_type": source_type,
            "loaded_base_parameters": True,
            "new_zero_initialized_output": "detail_head.4",
        }
    training_config = AttackTrainingConfig(
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
        preserve_initial_state=args.preserve_initial_state,
    )
    checkpoint, _ = train_client_received_decoder(
        decoder,
        train_dataset,
        validation_dataset,
        output / "checkpoints",
        training_config,
        device,
    )
    with (output / "training_run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                **vars(args),
                "device": str(device),
                "decoder_config": model_config.to_dict(),
                "decoder_type": decoder.decoder_type,
                "training_config": asdict(training_config),
                "decoder_checkpoint": str(checkpoint),
                "loaded_split_learning_checkpoint": False,
                "attacker_visible_signals": (
                    ["z", "u", "dL/dz"]
                    if args.decoder_architecture == "z_u_grad_z_bilinear"
                    else ["u", "dL/dz"]
                ),
                "initialization": initialization_summary,
            },
            handle,
            indent=2,
            default=str,
        )
    print(f"Attack decoder checkpoint: {checkpoint.resolve()}")
    return checkpoint


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()


__all__ = ["build_parser", "run"]
