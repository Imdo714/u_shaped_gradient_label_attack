from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader

from ..data.pairing import PairingAblationDataset
from ..data.transcript_dataset import PublicTargetDataset, TranscriptDataset
from ..evaluation.evaluator import ReconstructionEvaluator
from ..models.simulators import FrontSimulator, TailGradientSimulator
from ..training.abtr_trainer import ABTRConfig, ABTRTrainer
from ..training.tail_simulator_trainer import TailSimulatorConfig, TailSimulatorTrainer
from .common import load_server_middle_only, make_decoder, resolve_device, seed_everything


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run strict or semi-paired ABTR (P4/P5).")
    parser.add_argument("--real-manifest", required=True, help="Attacker transcript manifest")
    parser.add_argument("--public-manifest", required=True, help="Independent public image manifest")
    parser.add_argument("--test-manifest", required=True, help="Evaluator-only paired test manifest")
    parser.add_argument("--server-checkpoint", required=True, help="Checkpoint containing server-owned g")
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--tail-epochs", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--paired-manifest", default=None, help="Enable P5 semi-paired training")
    parser.add_argument("--paired-fraction", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser


def run(args: argparse.Namespace) -> dict[str, float | int]:
    seed_everything(args.seed)
    device = resolve_device(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    real_dataset = TranscriptDataset(args.real_manifest, include_target=False)
    public_dataset = PublicTargetDataset(args.public_manifest)
    test_dataset = TranscriptDataset(args.test_manifest, include_target=True)
    real_loader = DataLoader(
        real_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    public_loader = DataLoader(
        public_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    sample = real_dataset[0]
    server, cut_config = load_server_middle_only(args.server_checkpoint, device)
    front = FrontSimulator(
        int(sample["smashed_z"].shape[0]), tuple(sample["smashed_z"].shape[-2:])
    )
    tail = TailGradientSimulator(int(sample["server_output_u"].shape[0]), args.num_classes)
    decoder = make_decoder(sample, args.num_classes, args.image_size)
    TailSimulatorTrainer(
        TailSimulatorConfig(epochs=args.tail_epochs, learning_rate=args.learning_rate), device
    ).fit(tail, real_loader, output / "tail_warmup")

    paired_loader = None
    condition = "P4_strict_unpaired"
    if args.paired_manifest:
        condition = "P5_semi_paired"
        paired_source = TranscriptDataset(args.paired_manifest, include_target=True)
        paired_loader = DataLoader(
            PairingAblationDataset(
                paired_source, paired_fraction=args.paired_fraction, seed=args.seed
            ),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
    trainer = ABTRTrainer(
        ABTRConfig(epochs=args.epochs, learning_rate=args.learning_rate),
        device,
        args.num_classes,
    )
    trainer.fit(
        front, server, tail, decoder, real_loader, public_loader, output / "training", paired_loader
    )
    summary = ReconstructionEvaluator(device).evaluate(
        decoder, test_loader, output / "evaluation"
    )
    config = vars(args).copy()
    config.update({"condition": condition, "server_cut_config": cut_config})
    with (output / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()


__all__ = ["build_parser", "run"]
