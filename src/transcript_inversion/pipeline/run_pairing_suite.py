from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from torch.utils.data import DataLoader

from ..data.pairing import PairingAblationDataset, PairingMode
from ..data.transcript_dataset import TranscriptDataset
from ..evaluation.evaluator import ReconstructionEvaluator
from ..training.paired_decoder_trainer import PairedDecoderConfig, PairedDecoderTrainer
from .common import make_decoder, resolve_device, seed_everything


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run P0-P3 reconstruction pairing controls.")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--limited-fractions", default="0.01,0.05,0.1,0.25")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser


def _conditions(fractions: list[float]):
    yield "P0_exact", PairingMode.EXACT, 1.0
    for fraction in fractions:
        yield f"P1_limited_{fraction:g}", PairingMode.EXACT, fraction
    yield "P2_class_shuffled", PairingMode.CLASS_SHUFFLED, 1.0
    yield "P3_global_shuffled", PairingMode.GLOBAL_SHUFFLED, 1.0


def run(args: argparse.Namespace) -> list[dict[str, object]]:
    seed_everything(args.seed)
    device = resolve_device(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    train_source = TranscriptDataset(args.train_manifest, include_target=True)
    validation_source = TranscriptDataset(args.validation_manifest, include_target=True)
    test_source = TranscriptDataset(args.test_manifest, include_target=True)
    validation_loader = DataLoader(
        PairingAblationDataset(validation_source),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(test_source, batch_size=args.batch_size, num_workers=args.num_workers)
    fractions = [float(value) for value in args.limited_fractions.split(",") if value]
    rows: list[dict[str, object]] = []
    for name, mode, fraction in _conditions(fractions):
        condition_dir = output / name
        train_dataset = PairingAblationDataset(
            train_source, mode=mode, paired_fraction=fraction, seed=args.seed
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        decoder = make_decoder(train_source[0], args.num_classes, args.image_size).to(device)
        trainer = PairedDecoderTrainer(
            PairedDecoderConfig(epochs=args.epochs, learning_rate=args.learning_rate), device
        )
        trainer.fit(decoder, train_loader, validation_loader, condition_dir / "training")
        summary = ReconstructionEvaluator(device).evaluate(
            decoder, test_loader, condition_dir / "evaluation"
        )
        row: dict[str, object] = {
            "condition": name,
            "pairing_mode": mode.value,
            "paired_fraction": fraction,
            "train_pairs": len(train_dataset),
        }
        row.update(summary)
        rows.append(row)
    with (output / "pairing_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)
    return rows


def main() -> None:
    rows = run(build_parser().parse_args())
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()


__all__ = ["build_parser", "run"]
