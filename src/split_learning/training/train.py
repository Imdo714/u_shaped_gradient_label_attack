from __future__ import annotations

import argparse
from pathlib import Path

from ...shared.configuration.experiment_config import ExperimentConfig
from ...shared.configuration.workspace_paths import DEFAULT_WORKSPACE_PATHS
from ...shared.data.class_catalog import ClassCatalog
from ...shared.data.image_dataset import make_loader
from ...shared.reproducibility.random_seed import seed_everything
from ..architecture.split_learning_model import build_split_learning_model
from ..gradient_flow.gradient_exchange import COMMUNICATION_DIAGRAM
from .split_learning_trainer import SplitLearningTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train explicit U-shaped Split Learning")
    parser.add_argument("--data", default=str(DEFAULT_WORKSPACE_PATHS.dataset))
    parser.add_argument("--output-dir", default=str(DEFAULT_WORKSPACE_PATHS.results_root))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cut-config", choices=("early", "middle", "late"), default="middle")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--debug-samples",
        type=int,
        default=3,
        help="첫 epoch에서 상세 중간값/gradient를 출력할 샘플 수 (0이면 끔)",
    )
    parser.add_argument(
        "--debug-values",
        type=int,
        default=8,
        help="각 텐서에서 터미널에 표시할 앞부분 값의 개수",
    )
    return parser.parse_args()


def train_from_config(config: ExperimentConfig):
    seed_everything(config.random_seed)
    print(COMMUNICATION_DIAGRAM)
    catalog = ClassCatalog.discover(config.data_dir)
    train_loader = make_loader(
        config.data_dir,
        "train",
        config.image_size,
        config.batch_size,
        config.num_workers,
        shuffle=True,
        augment=True,
        class_names=catalog.names,
    )
    val_split = "val" if (Path(config.data_dir) / "val").exists() else "train"
    val_loader = make_loader(
        config.data_dir,
        val_split,
        config.image_size,
        config.batch_size,
        config.num_workers,
        shuffle=False,
        class_names=catalog.names,
    )
    trainer = SplitLearningTrainer(
        build_split_learning_model(config.cut_config, num_classes=catalog.num_classes),
        config.resolved_device(),
        config.learning_rate,
        config.output_path("checkpoints"),
    )
    history = []
    for epoch in range(1, config.epochs + 1):
        train_metrics = trainer.train_epoch(
            train_loader,
            epoch,
            debug_samples=config.debug_samples,
            debug_values=config.debug_values,
        )  # 모델 f, g, h 학습, gradient 계산, optimizer 업데이트
        val_metrics = trainer.evaluate(val_loader)  # Validate all discovered classes.
        trainer.save_checkpoint(
            epoch,
            config=config.to_dict(),
            class_names=list(catalog.names),
            train_accuracy=train_metrics["accuracy"],
            validation_accuracy=val_metrics["accuracy"],
        )
        row = {"epoch": epoch, "train": train_metrics, "validation": val_metrics}
        history.append(row)
        print(
            f"Epoch {epoch}/{config.epochs}: loss={train_metrics['loss']:.4f}, "
            f"train_acc={train_metrics['accuracy']:.2%}, val_acc={val_metrics['accuracy']:.2%}"
        )
    return trainer, train_loader, val_loader, history


def main() -> None:
    args = parse_args()
    train_from_config(
        ExperimentConfig(
            data_dir=args.data,
            output_dir=args.output_dir,
            image_size=args.image_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            random_seed=args.random_seed,
            num_workers=args.num_workers,
            cut_config=args.cut_config,
            device=args.device,
            debug_samples=args.debug_samples,
            debug_values=args.debug_values,
        )
    )


if __name__ == "__main__":
    main()
