from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from ...shared.data.class_catalog import ClassCatalog
from ...shared.data.image_dataset import make_loader
from ...shared.reproducibility.random_seed import seed_everything
from ...split_learning.architecture.split_learning_model import (
    build_split_learning_model,
)
from ...split_learning.training.split_learning_trainer import SplitLearningTrainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the letter victim f-g-h model without horizontal-flip "
            "augmentation, which would change character geometry."
        )
    )
    parser.add_argument(
        "--data", default="workspace/data/letter_experiment/victim"
    )
    parser.add_argument(
        "--output",
        default=(
            "workspace/results/shared_server_gradient_transcript_inversion/"
            "letter_c/victim_model"
        ),
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--cut-config", choices=("early", "middle", "late"), default="middle")
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def run(args: argparse.Namespace) -> Path:
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch size must be positive")
    if args.early_stopping_patience < 0:
        raise ValueError("early stopping patience must be non-negative")
    seed_everything(args.seed)
    device = _device(args.device)
    catalog = ClassCatalog.discover(args.data)
    train_loader = make_loader(
        args.data,
        "train",
        args.image_size,
        args.batch_size,
        args.num_workers,
        shuffle=True,
        augment=False,
        class_names=catalog.names,
    )
    validation_loader = make_loader(
        args.data,
        "val",
        args.image_size,
        args.batch_size,
        args.num_workers,
        shuffle=False,
        augment=False,
        class_names=catalog.names,
    )
    test_loader = make_loader(
        args.data,
        "test",
        args.image_size,
        args.batch_size,
        args.num_workers,
        shuffle=False,
        augment=False,
        class_names=catalog.names,
    )
    output = Path(args.output)
    checkpoint_dir = output / "checkpoints"
    model = build_split_learning_model(
        args.cut_config, num_classes=catalog.num_classes
    )
    trainer = SplitLearningTrainer(
        model, device, args.learning_rate, checkpoint_dir
    )
    best_state = copy.deepcopy(model.state_dict())
    best_accuracy = -1.0
    best_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, object]] = []
    config = {
        **vars(args),
        "resolved_device": str(device),
        "class_names": list(catalog.names),
        "train_horizontal_flip": False,
    }

    for epoch in range(1, args.epochs + 1):
        train_metrics = trainer.train_epoch(
            train_loader, epoch, debug_samples=0, debug_values=0
        )
        validation_metrics = trainer.evaluate(validation_loader)
        trainer.save_checkpoint(
            epoch,
            config=config,
            class_names=list(catalog.names),
            train_accuracy=train_metrics["accuracy"],
            validation_accuracy=validation_metrics["accuracy"],
        )
        history.append(
            {"epoch": epoch, "train": train_metrics, "validation": validation_metrics}
        )
        print(
            f"Synthetic victim epoch {epoch:03d}/{args.epochs:03d}: "
            f"train_acc={train_metrics['accuracy']:.2%}, "
            f"validation_acc={validation_metrics['accuracy']:.2%}, "
            f"validation_loss={validation_metrics['loss']:.4f}",
            flush=True,
        )
        improved = validation_metrics["accuracy"] > best_accuracy or (
            validation_metrics["accuracy"] == best_accuracy
            and validation_metrics["loss"] < best_loss
        )
        if improved:
            best_accuracy = validation_metrics["accuracy"]
            best_loss = validation_metrics["loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if (
                args.early_stopping_patience > 0
                and epochs_without_improvement >= args.early_stopping_patience
            ):
                print(f"Early stopping at epoch {epoch}; best epoch={best_epoch}")
                break

    model.load_state_dict(best_state)
    test_metrics = trainer.evaluate(test_loader)
    best_checkpoint = checkpoint_dir / "model_best.pt"
    model.save(
        best_checkpoint,
        epoch=best_epoch,
        config=config,
        class_names=list(catalog.names),
        validation_accuracy=best_accuracy,
        validation_loss=best_loss,
        selection="highest validation accuracy, then lowest validation loss",
    )
    output.mkdir(parents=True, exist_ok=True)
    with (output / "training_history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    with (output / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                **config,
                "best_epoch": best_epoch,
                "best_validation_accuracy": best_accuracy,
                "best_validation_loss": best_loss,
                "test_metrics": test_metrics,
                "best_checkpoint": str(best_checkpoint),
            },
            handle,
            indent=2,
            default=str,
        )
    print(f"Best synthetic victim checkpoint: {best_checkpoint.resolve()}")
    print(f"Unseen test accuracy: {test_metrics['accuracy']:.2%}")
    return best_checkpoint


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()


__all__ = ["build_parser", "run"]
