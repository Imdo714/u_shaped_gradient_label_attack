from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from ...shared.configuration.workspace_paths import DEFAULT_WORKSPACE_PATHS
from ..attacks.anchor_mapping import extract_anchor_matrix, map_anchors_to_clusters
from ..clustering.cluster_gradients import cluster_epoch
from .evaluate_attack import evaluate_files
from ...shared.data.class_catalog import ClassCatalog
from ...shared.evaluation.visualization import plot_epoch_f1
from ...split_learning.architecture.split_learning_model import load_split_learning_model


def analyze_epochs(
    transcripts: Path,
    checkpoints: Path,
    results: Path,
    anchor_dir: Path,
    class_names: tuple[str, ...],
    device: torch.device,
    seed: int = 42,
) -> list[dict]:
    catalog = ClassCatalog.from_names(class_names)
    anchor_paths = catalog.anchor_paths(anchor_dir)
    with (transcripts / "attacker_transcript" / "index.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        index_epochs = sorted({int(row["epoch"]) for row in csv.DictReader(handle)})
    rows: list[dict] = []
    truth_path = transcripts / "evaluator_ground_truth" / "ground_truth.csv"
    for epoch in index_epochs:
        data, cluster_ids, centroids = cluster_epoch(
            transcripts, results, epoch, catalog.num_classes, seed
        )
        model, metadata = load_split_learning_model(
            checkpoints / f"epoch_{epoch:03d}.pt",
            device,
        )
        image_size = int(metadata.get("config", {}).get("image_size", 64))
        anchors = extract_anchor_matrix(model, anchor_paths, image_size, device)
        mapping_result = map_anchors_to_clusters(anchors, centroids)
        cluster_file = results / f"gradient_clusters_epoch_{epoch:03d}.csv"
        metrics, _, _, _ = evaluate_files(
            cluster_file,
            truth_path,
            mapping_result.cluster_to_label,
            results,
            transcripts,
            epoch,
            catalog.names,
        )
        rows.append({"epoch": epoch, "num_samples": len(cluster_ids), **metrics})
    csv_path = results / "attack_f1_by_epoch.csv"
    fields = ("epoch", "num_samples", "purity", "ARI", "NMI", "attack_accuracy", "F1")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    plot_epoch_f1(rows, results / "attack_f1_by_epoch.png")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcripts", default=str(DEFAULT_WORKSPACE_PATHS.transcripts))
    parser.add_argument("--checkpoints", default=str(DEFAULT_WORKSPACE_PATHS.checkpoints))
    parser.add_argument("--results", default=str(DEFAULT_WORKSPACE_PATHS.reports))
    parser.add_argument("--anchor-dir", default=str(DEFAULT_WORKSPACE_PATHS.anchors))
    parser.add_argument("--data", default=str(DEFAULT_WORKSPACE_PATHS.dataset))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    catalog = ClassCatalog.discover(args.data)
    rows = analyze_epochs(
        Path(args.transcripts),
        Path(args.checkpoints),
        Path(args.results),
        Path(args.anchor_dir),
        catalog.names,
        torch.device(args.device),
    )
    for row in rows:
        print(f"Epoch {row['epoch']}: n={row['num_samples']}, F1={row['F1']:.4f}")


if __name__ == "__main__":
    main()
