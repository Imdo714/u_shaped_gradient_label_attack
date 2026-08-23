from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .attacks.anchor_mapping import (
    extract_anchor_matrix,
    map_anchors_to_clusters,
    print_anchor_mapping,
)
from .cluster_gradients import cluster_epoch
from .epoch_analysis import analyze_epochs
from .evaluate_attack import evaluate_files
from .same_image_different_label import experiment_same_image_different_label
from ..shared.configuration.experiment_config import ExperimentConfig
from ..shared.configuration.workspace_paths import DEFAULT_WORKSPACE_PATHS
from ..shared.data.class_catalog import ClassCatalog
from ..shared.data.image_dataset import load_image, make_loader
from ..split_learning.architecture.split_learning_model import (
    load_split_learning_model,
)
from ..split_learning.logging.gradient_transcript_logger import (
    EvaluatorGroundTruthLogger,
    ServerGradientTranscriptLogger,
)
from ..split_learning.training.split_learning_trainer import SplitLearningTrainer
from ..split_learning.training.train import train_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete controlled security experiment")
    parser.add_argument("--data", default=str(DEFAULT_WORKSPACE_PATHS.dataset))
    parser.add_argument("--output-dir", default=str(DEFAULT_WORKSPACE_PATHS.results_root))
    parser.add_argument("--anchor-dir", default=str(DEFAULT_WORKSPACE_PATHS.anchors))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--cut-config", choices=("early", "middle", "late"), default="middle")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-attack-samples", type=int)
    parser.add_argument("--debug-samples", type=int, default=3)
    parser.add_argument("--debug-values", type=int, default=8)
    args = parser.parse_args()
    config = ExperimentConfig(
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
        max_attack_samples=args.max_attack_samples,
        debug_samples=args.debug_samples,
        debug_values=args.debug_values,
    )
    output = Path(args.output_dir)
    catalog = ClassCatalog.discover(args.data)
    transcripts = output / "transcripts"
    results = output / "reports"
    checkpoints = output / "checkpoints"
    anchor_dir = Path(args.anchor_dir)
    anchor_paths = catalog.anchor_paths(anchor_dir)
    missing_anchors = [path for path in anchor_paths if not path.is_file()]
    if missing_anchors:
        raise FileNotFoundError(f"Missing class anchors: {[str(path) for path in missing_anchors]}")

    print("[1/7] Training U-shaped Split Learning")
    trainer, train_loader, val_loader, history = train_from_config(config)
    print("[2/7] Collecting server-visible gradients")
    for epoch in range(1, config.epochs + 1):
        epoch_model, _ = load_split_learning_model(
            checkpoints / f"epoch_{epoch:03d}.pt",
            config.resolved_device(),
        )
        collector = SplitLearningTrainer(epoch_model, config.resolved_device(), config.learning_rate, checkpoints)
        collector.collect_frozen_transcript(
            make_loader(
                args.data,
                "train",
                args.image_size,
                args.batch_size,
                args.num_workers,
                False,
                class_names=catalog.names,
            ),
            epoch,
            ServerGradientTranscriptLogger(transcripts),
            EvaluatorGroundTruthLogger(transcripts),
            args.max_attack_samples,
        )
    print("[3/7] Verifying label-dependent gradients")
    experiment_same_image_different_label(
        trainer.model,
        load_image(anchor_paths[0], args.image_size).to(config.resolved_device()),
        catalog.names,
        anchor_paths[0].name,
    )
    print("[4/7] Normalizing gradient vectors")
    print("Features are flattened and L2-normalized with epsilon=1e-12 on transcript load.")
    print(f"[5/7] KMeans clustering K={catalog.num_classes}")
    final_data, final_clusters, final_centroids = cluster_epoch(
        transcripts, results, config.epochs, catalog.num_classes, config.random_seed
    )
    print("[6/7] Mapping clusters using one anchor per class")
    final_model, _ = load_split_learning_model(
        checkpoints / f"epoch_{config.epochs:03d}.pt",
        config.resolved_device(),
    )
    anchors = extract_anchor_matrix(
        final_model, anchor_paths, args.image_size, config.resolved_device()
    )
    mapping = map_anchors_to_clusters(anchors, final_centroids)
    print_anchor_mapping(mapping, list(catalog.names))
    (results / "cluster_mapping.json").write_text(
        json.dumps({str(k): v for k, v in mapping.cluster_to_label.items()}, indent=2),
        encoding="utf-8",
    )
    print("[7/7] Evaluating inferred labels")
    metrics, matrix, _, _ = evaluate_files(
        results / f"gradient_clusters_epoch_{config.epochs:03d}.csv",
        transcripts / "evaluator_ground_truth" / "ground_truth.csv",
        mapping.cluster_to_label,
        results,
        transcripts,
        config.epochs,
        catalog.names,
    )
    epoch_rows = analyze_epochs(
        transcripts,
        checkpoints,
        results,
        anchor_dir,
        catalog.names,
        config.resolved_device(),
        config.random_seed,
    )
    test_split = "test" if (Path(args.data) / "test").exists() else "val"
    test_loader = make_loader(
        args.data,
        test_split,
        args.image_size,
        args.batch_size,
        args.num_workers,
        False,
        class_names=catalog.names,
    )
    test_metrics = trainer.evaluate(test_loader)
    counts = np.bincount(final_clusters, minlength=catalog.num_classes)
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Victim training accuracy:    {history[-1]['train']['accuracy']:.2%}")
    print(f"Victim {test_split} accuracy: {test_metrics['accuracy']:.2%}")
    print(f"Number of attack samples:   {len(final_clusters)}")
    print(f"Gradient cluster counts:    {counts.tolist()}")
    for cluster, label in sorted(mapping.cluster_to_label.items()):
        print(f"Recovered mapping:          Cluster {cluster} = {catalog.names[label].upper()}")
    for key in ("purity", "ARI", "NMI", "attack_accuracy", "F1"):
        print(f"{key:27s}{metrics[key]:.6f}")
    print(f"Files: {results / 'pca_gradient_clusters.png'}, {results / 'confusion_matrix.png'},")
    print(f"       {results / 'attack_f1_by_epoch.png'}, {results / 'gradient_clusters.csv'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
