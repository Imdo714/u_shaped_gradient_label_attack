from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ...shared.configuration.workspace_paths import DEFAULT_WORKSPACE_PATHS
from ...shared.data.class_catalog import checkpoint_class_catalog
from ...shared.data.image_dataset import make_loader
from ...split_learning.architecture.split_learning_model import load_split_learning_model
from ...split_learning.logging.gradient_transcript_logger import (
    EvaluatorGroundTruthLogger,
    ServerGradientTranscriptLogger,
)
from ...split_learning.training.split_learning_trainer import SplitLearningTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect frozen-checkpoint training loss gradients")
    parser.add_argument("--data", default=str(DEFAULT_WORKSPACE_PATHS.dataset))
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_WORKSPACE_PATHS.checkpoints / "model.pt"),
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_WORKSPACE_PATHS.results_root))
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    model, metadata = load_split_learning_model(args.checkpoint, device)
    catalog = checkpoint_class_catalog(metadata, model.h_model.num_classes)
    config = metadata.get("config", {})
    epoch = args.epoch or int(metadata.get("epoch", 1))
    loader = make_loader(
        args.data,
        args.split,
        int(config.get("image_size", 64)),
        args.batch_size,
        int(config.get("num_workers", 0)),
        shuffle=False,
        class_names=catalog.names,
    )
    trainer = SplitLearningTrainer(model, device, 1e-3, Path(args.output_dir) / "checkpoints")
    count = trainer.collect_frozen_transcript(
        loader,
        epoch,
        ServerGradientTranscriptLogger(Path(args.output_dir) / "transcripts"),
        EvaluatorGroundTruthLogger(Path(args.output_dir) / "transcripts"),
        args.max_samples,
    )
    print(f"Collected {count} per-sample server transcripts for epoch {epoch}.")
    print("Attacker records contain no labels; evaluator truth is in a separate directory.")


if __name__ == "__main__":
    main()
