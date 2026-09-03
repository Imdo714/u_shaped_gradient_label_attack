from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from ...client_received_transcript_attack.data.dataset import (
    ClientReceivedTranscriptDataset,
)
from ...client_received_transcript_attack.models.factory import (
    DecoderModel,
    decoder_from_config,
)
from ...client_received_transcript_attack.training.trainer import (
    AttackTrainingConfig,
    train_client_received_decoder,
)
from ...shared.reproducibility.random_seed import seed_everything
from .common import device_from_name, file_sha256, load_frozen_decoder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create out-of-fold coarse reconstructions for refiner training. "
            "Every auxiliary-train image is reconstructed by a decoder that did "
            "not optimize on that image."
        )
    )
    parser.add_argument("--reference-decoder-checkpoint", required=True)
    parser.add_argument("--train-attacker-manifest", required=True)
    parser.add_argument("--train-target-manifest", required=True)
    parser.add_argument("--validation-attacker-manifest", required=True)
    parser.add_argument("--validation-target-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--decoder-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser


def _tensor(batch: dict, key: str, device: torch.device) -> Tensor:
    value = batch[key]
    if not isinstance(value, Tensor):
        raise TypeError(f"batch[{key!r}] must be a tensor")
    return value.to(device)


def _fold_indices(samples: int, folds: int, seed: int) -> list[list[int]]:
    if folds < 2 or folds > samples:
        raise ValueError("folds must be between 2 and the number of train samples")
    generator = torch.Generator().manual_seed(seed)
    shuffled = torch.randperm(samples, generator=generator).tolist()
    return [shuffled[index::folds] for index in range(folds)]


def _record_filename(transcript_id: str) -> str:
    digest = hashlib.sha256(transcript_id.encode("utf-8")).hexdigest()[:24]
    return f"coarse_{digest}.npz"


def _materialize(
    decoder: DecoderModel,
    dataset,
    output_dir: Path,
    device: torch.device,
    batch_size: int,
    fold_name: str,
    num_workers: int,
) -> list[dict[str, str]]:
    records_dir = output_dir / "coarse_records"
    records_dir.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    rows: list[dict[str, str]] = []
    decoder.eval()
    with torch.no_grad():
        for batch in loader:
            coarse, _ = decoder(
                _tensor(batch, "server_output_u", device),
                _tensor(batch, "grad_g_to_f", device),
            )
            for index, transcript_id in enumerate(batch["transcript_id"]):
                transcript_id = str(transcript_id)
                filename = _record_filename(transcript_id)
                np.savez_compressed(
                    records_dir / filename,
                    coarse_reconstruction=coarse[index].cpu().numpy(),
                )
                rows.append(
                    {
                        "transcript_id": transcript_id,
                        "coarse_record": f"coarse_records/{filename}",
                        "fold": fold_name,
                    }
                )
    return rows


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty coarse manifest")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("transcript_id", "coarse_record", "fold")
        )
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["transcript_id"]))


def _training_config(
    checkpoint: dict,
    epochs: int,
    batch_size: int,
    learning_rate: float | None,
    num_workers: int,
) -> AttackTrainingConfig:
    saved = checkpoint.get("training_config", {})
    return AttackTrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=(
            float(learning_rate)
            if learning_rate is not None
            else float(saved.get("learning_rate", 1e-3))
        ),
        weight_decay=float(saved.get("weight_decay", 1e-5)),
        l1_weight=float(saved.get("l1_weight", 1.0)),
        ssim_weight=float(saved.get("ssim_weight", 0.75)),
        edge_weight=float(saved.get("edge_weight", 0.15)),
        perceptual_weight=float(saved.get("perceptual_weight", 0.25)),
        laplacian_weight=float(saved.get("laplacian_weight", 0.0)),
        classification_weight=float(saved.get("classification_weight", 0.1)),
        gradient_clip_norm=float(saved.get("gradient_clip_norm", 5.0)),
        num_workers=num_workers,
    )


def run(args: argparse.Namespace) -> Path:
    seed_everything(args.seed)
    device = device_from_name(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    train_output = output / "oof_train"
    validation_output = output / "reference_validation"
    train_output.mkdir(parents=True, exist_ok=True)
    validation_output.mkdir(parents=True, exist_ok=True)

    train_dataset = ClientReceivedTranscriptDataset(
        args.train_attacker_manifest, args.train_target_manifest
    )
    validation_dataset = ClientReceivedTranscriptDataset(
        args.validation_attacker_manifest, args.validation_target_manifest
    )
    train_ids = {str(row["transcript_id"]) for row in train_dataset.rows}
    validation_ids = {str(row["transcript_id"]) for row in validation_dataset.rows}
    if train_ids & validation_ids:
        raise ValueError("auxiliary train and validation transcript IDs overlap")

    reference_state = torch.load(
        args.reference_decoder_checkpoint, map_location="cpu", weights_only=False
    )
    decoder_type = str(
        reference_state.get("decoder_type", "baseline_bilinear")
    )
    model_config_dict = reference_state["decoder_config"]
    model_config = decoder_from_config(decoder_type, model_config_dict).config
    training_config = _training_config(
        reference_state,
        args.decoder_epochs,
        args.batch_size,
        args.learning_rate,
        args.num_workers,
    )
    folds = _fold_indices(len(train_dataset), args.folds, args.seed)
    all_indices = set(range(len(train_dataset)))
    oof_rows: list[dict[str, str]] = []
    assignments: list[dict[str, str | int]] = []

    for fold_index, held_out_indices in enumerate(folds):
        fold_number = fold_index + 1
        seed_everything(args.seed + fold_index)
        fold_train_indices = sorted(all_indices - set(held_out_indices))
        fold_decoder = decoder_from_config(decoder_type, model_config_dict).to(device)
        fold_root = output / "fold_decoders" / f"fold_{fold_number:02d}"
        print(
            f"[OOF] fold {fold_number}/{args.folds}: "
            f"train={len(fold_train_indices)}, predict={len(held_out_indices)}",
            flush=True,
        )
        train_client_received_decoder(
            fold_decoder,
            Subset(train_dataset, fold_train_indices),
            validation_dataset,
            fold_root,
            training_config,
            device,
        )
        fold_subset = Subset(train_dataset, held_out_indices)
        oof_rows.extend(
            _materialize(
                fold_decoder,
                fold_subset,
                train_output,
                device,
                args.batch_size,
                f"fold_{fold_number:02d}",
                args.num_workers,
            )
        )
        assignments.extend(
            {
                "transcript_id": str(train_dataset.rows[index]["transcript_id"]),
                "fold": fold_number,
            }
            for index in held_out_indices
        )

    if len(oof_rows) != len(train_dataset):
        raise RuntimeError("OOF generation did not produce exactly one coarse image per sample")
    if len({row["transcript_id"] for row in oof_rows}) != len(oof_rows):
        raise RuntimeError("OOF generation produced duplicate transcript IDs")
    _write_manifest(train_output / "coarse_manifest.csv", oof_rows)
    with (output / "fold_assignments.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=("transcript_id", "fold"))
        writer.writeheader()
        writer.writerows(sorted(assignments, key=lambda row: str(row["transcript_id"])))

    reference_decoder = load_frozen_decoder(
        args.reference_decoder_checkpoint, device
    )
    validation_rows = _materialize(
        reference_decoder,
        validation_dataset,
        validation_output,
        device,
        args.batch_size,
        "reference_decoder",
        args.num_workers,
    )
    _write_manifest(validation_output / "coarse_manifest.csv", validation_rows)

    with (output / "oof_data_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                **vars(args),
                "device": str(device),
                "train_samples": len(train_dataset),
                "validation_samples": len(validation_dataset),
                "fold_sizes": [len(fold) for fold in folds],
                "decoder_config": model_config.to_dict(),
                "decoder_type": decoder_type,
                "fold_training_config": asdict(training_config),
                "reference_decoder_sha256": file_sha256(
                    args.reference_decoder_checkpoint
                ),
                "oof_train_manifest": str(train_output / "coarse_manifest.csv"),
                "validation_manifest": str(
                    validation_output / "coarse_manifest.csv"
                ),
                "victim_holdout_used": False,
            },
            handle,
            indent=2,
            default=str,
        )
    print(f"OOF refiner data: {output.resolve()}")
    return output


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()


__all__ = ["build_parser", "run"]
