from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch

from ...decoder.data.holdout_selection import (
    HoldoutRecord,
    select_holdout_records,
    write_holdout_records,
)
from ...shared.data.class_catalog import checkpoint_class_catalog
from ...shared.data.image_dataset import make_loader
from ...shared.reproducibility.random_seed import seed_everything
from ...split_learning.architecture.split_learning_model import load_split_learning_model
from ..data.collector import (
    CollectionManifests,
    FrozenSplitLearningProvider,
    collect_received_transcripts,
)
from ..data.dataset import ClientReceivedTranscriptDataset
from ..evaluation.evaluator import evaluate_client_received_decoder
from ..models.decoder import ClientReceivedDecoder, ClientReceivedDecoderConfig
from ..training.trainer import AttackTrainingConfig, train_client_received_decoder


DEFAULT_CHECKPOINT = Path("workspace/results/runs/cut_middle/checkpoints/model.pt")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "LOCAL-EXACT BASELINE: load one full checkpoint, collect u/dL-dz, train "
            "an architecture-agnostic decoder, and evaluate unseen holdouts. Use "
            "run_rpc_experiment for process-separated black-box evidence."
        )
    )
    parser.add_argument("--data", default="workspace/data/dataset")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output", default=None)
    parser.add_argument("--aux-train-split", default="train")
    parser.add_argument("--aux-validation-split", default="val")
    parser.add_argument(
        "--victim-split",
        choices=("test", "new_holdout"),
        default="test",
    )
    parser.add_argument("--holdout-count", type=int, default=20)
    parser.add_argument("--holdout-start-index", type=int, default=0)
    parser.add_argument("--holdout-labels", nargs="+", default=None)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--classification-weight", type=float, default=0.1)
    parser.add_argument("--use-label-head", action="store_true")
    parser.add_argument("--max-aux-train-samples", type=int, default=None)
    parser.add_argument("--max-aux-validation-samples", type=int, default=None)
    parser.add_argument("--signal-spatial-size", type=int, default=16)
    parser.add_argument("--signal-channels", type=int, default=64)
    parser.add_argument("--decoder-base-channels", type=int, default=256)
    parser.add_argument("--decoder-min-channels", type=int, default=32)
    parser.add_argument("--refinement-blocks", type=int, default=1)
    parser.add_argument("--l1-weight", type=float, default=1.0)
    parser.add_argument("--ssim-weight", type=float, default=0.75)
    parser.add_argument("--edge-weight", type=float, default=0.15)
    parser.add_argument("--perceptual-weight", type=float, default=0.25)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-grid-images", type=int, default=20)
    parser.add_argument("--reuse-observations", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _balanced_holdouts(
    data_dir: str | Path,
    split: str,
    class_names: tuple[str, ...],
    labels: tuple[str, ...],
    count: int,
    start_index: int,
) -> list[HoldoutRecord]:
    if count < 1:
        raise ValueError("holdout_count must be positive")
    unknown = sorted(set(labels) - set(class_names))
    if unknown:
        raise ValueError(f"unknown holdout labels: {unknown}")
    base_count, remainder = divmod(count, len(labels))
    records: list[HoldoutRecord] = []
    for label_index, label in enumerate(labels):
        class_count = base_count + int(label_index < remainder)
        if class_count:
            records.extend(
                select_holdout_records(
                    data_dir,
                    split,
                    class_names,
                    holdouts_per_class=class_count,
                    labels=(label,),
                    start_index=start_index,
                )
            )
    return records


def _manifests(root: Path) -> CollectionManifests:
    manifests = CollectionManifests(
        attacker=root / "attacker_manifest.csv",
        evaluator=root / "evaluator_manifest.csv",
    )
    missing = [path for path in (manifests.attacker, manifests.evaluator) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"observation manifests not found: {missing}")
    return manifests


def run(args: argparse.Namespace) -> dict[str, float | int | bool]:
    print(
        "[LOCAL-EXACT BASELINE] This process loads the full victim checkpoint. "
        "Use run_rpc_experiment for checkpoint-separated evidence."
    )
    seed_everything(args.seed)
    device = _device(args.device)
    if args.aux_train_split == args.aux_validation_split:
        raise ValueError("auxiliary train and validation splits must be different")
    if args.victim_split in (args.aux_train_split, args.aux_validation_split):
        raise ValueError("victim split must be separate from auxiliary train/validation")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = (
        Path(args.output)
        if args.output
        else Path("workspace/results/client_received_transcript_attack") / stamp
    )
    observations = output / "observations"
    output.mkdir(parents=True, exist_ok=True)

    split_model, metadata = load_split_learning_model(args.checkpoint, device)
    catalog = checkpoint_class_catalog(metadata, split_model.h_model.num_classes)
    selected_labels = tuple(args.holdout_labels) if args.holdout_labels else catalog.names
    holdouts = _balanced_holdouts(
        args.data,
        args.victim_split,
        catalog.names,
        selected_labels,
        args.holdout_count,
        args.holdout_start_index,
    )
    holdout_ids = {record.sample_id for record in holdouts}
    write_holdout_records(holdouts, output / "holdout_records.csv")

    if args.reuse_observations:
        print("[1/4] Reusing previously collected u and dL/dz records...")
        train_manifests = _manifests(observations / "aux_train")
        validation_manifests = _manifests(observations / "aux_validation")
        victim_manifests = _manifests(observations / "victim_holdout")
    else:
        print("[1/4] Collecting public auxiliary u and dL/dz pairs...")
        provider = FrozenSplitLearningProvider(split_model)
        train_manifests = collect_received_transcripts(
            provider,
            make_loader(
                args.data,
                args.aux_train_split,
                args.image_size,
                batch_size=1,
                num_workers=args.num_workers,
                shuffle=False,
                class_names=catalog.names,
            ),
            observations / "aux_train",
            "aux_train",
            device,
            max_samples=args.max_aux_train_samples,
        )
        validation_manifests = collect_received_transcripts(
            provider,
            make_loader(
                args.data,
                args.aux_validation_split,
                args.image_size,
                batch_size=1,
                num_workers=args.num_workers,
                shuffle=False,
                class_names=catalog.names,
            ),
            observations / "aux_validation",
            "aux_validation",
            device,
            max_samples=args.max_aux_validation_samples,
        )
        print("[2/4] Collecting evaluator-only victim holdout transcripts...")
        victim_manifests = collect_received_transcripts(
            provider,
            make_loader(
                args.data,
                args.victim_split,
                args.image_size,
                batch_size=1,
                num_workers=args.num_workers,
                shuffle=False,
                class_names=catalog.names,
                include_sample_ids=holdout_ids,
            ),
            observations / "victim_holdout",
            "victim_holdout",
            device,
        )
        del provider

    del split_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    train_dataset = ClientReceivedTranscriptDataset(
        train_manifests.attacker, train_manifests.evaluator
    )
    validation_dataset = ClientReceivedTranscriptDataset(
        validation_manifests.attacker, validation_manifests.evaluator
    )
    victim_dataset = ClientReceivedTranscriptDataset(
        victim_manifests.attacker, victim_manifests.evaluator
    )
    sample = train_dataset[0]
    server_output_u = sample["server_output_u"]
    grad_g_to_f = sample["grad_g_to_f"]
    if not isinstance(server_output_u, torch.Tensor) or not isinstance(
        grad_g_to_f, torch.Tensor
    ):
        raise TypeError("transcript dataset returned non-tensor signals")

    model_config = ClientReceivedDecoderConfig(
        u_channels=int(server_output_u.shape[0]),
        grad_z_channels=int(grad_g_to_f.shape[0]),
        num_classes=catalog.num_classes,
        image_size=args.image_size,
        signal_spatial_size=args.signal_spatial_size,
        signal_channels=args.signal_channels,
        decoder_base_channels=args.decoder_base_channels,
        decoder_min_channels=args.decoder_min_channels,
        refinement_blocks=args.refinement_blocks,
        use_label_head=args.use_label_head,
    )
    decoder = ClientReceivedDecoder(model_config).to(device)
    training_config = AttackTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        l1_weight=args.l1_weight,
        ssim_weight=args.ssim_weight,
        edge_weight=args.edge_weight,
        perceptual_weight=args.perceptual_weight,
        classification_weight=args.classification_weight,
        gradient_clip_norm=args.gradient_clip_norm,
        num_workers=args.num_workers,
    )

    print("[3/4] Training u + dL/dz encoders and image decoder end-to-end...")
    checkpoint, _ = train_client_received_decoder(
        decoder,
        train_dataset,
        validation_dataset,
        output / "checkpoints",
        training_config,
        device,
    )
    print("[4/4] Evaluating unseen victim holdouts...")
    summary = evaluate_client_received_decoder(
        decoder,
        victim_dataset,
        output / "evaluation",
        device,
        catalog.names,
        batch_size=args.batch_size,
        max_grid_images=args.max_grid_images,
    )
    with (output / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                **vars(args),
                "resolved_output": str(output),
                "device": str(device),
                "class_names": list(catalog.names),
                "holdout_samples": len(holdouts),
                "attacker_visible_signals": ["u", "dL/dz"],
                "experiment_mode": "local_exact_checkpoint_provider",
                "final_black_box_evidence": False,
                "decoder_config": model_config.to_dict(),
                "training_config": asdict(training_config),
                "decoder_checkpoint": str(checkpoint),
                "evaluation_summary": summary,
            },
            handle,
            indent=2,
            default=str,
        )
    print(json.dumps(summary, indent=2))
    print(f"Results: {output.resolve()}")
    return summary


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()


__all__ = ["build_parser", "run"]
