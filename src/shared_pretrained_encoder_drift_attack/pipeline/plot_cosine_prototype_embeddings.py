from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA

from ...shared.data.class_catalog import ClassCatalog
from ...shared.data.image_dataset import make_loader
from ...shared.reproducibility.random_seed import seed_everything
from ...split_learning.architecture.split_learning_model import SplitLearningModel
from ...split_learning.f_model.client_front_f_model import ClientFrontFModel
from ...split_learning.g_model.server_middle_g_model import ServerMiddleGModel
from ...split_learning.h_model.client_tail_h_model import ClientTailHModel
from .run_joint_transcript_attack import _device, _warmup_classifier
from .run_online_transcript_label_attack import (
    _attack_validation_ids,
    _partition_victim_dataset,
    _prototype_features,
    _run_world,
    _temporal_bundles,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot the actual gradient-only cosine-prototype feature geometry."
    )
    parser.add_argument(
        "--result-root",
        default=(
            "workspace/results/shared_pretrained_encoder_drift_attack/"
            "animal5_online_transcript_core"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--victim-index", type=int, default=1)
    parser.add_argument("--expected-accuracy", type=float, default=0.82)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None)
    return parser


def _runtime_args(config: dict[str, object], device_override: str | None) -> argparse.Namespace:
    values = {
        key: value
        for key, value in config.items()
        if key not in {"resolved_output", "resolved_device", "class_names", "signal_modes"}
    }
    if device_override is not None:
        values["device"] = device_override
    return argparse.Namespace(**values)


def run(cli: argparse.Namespace) -> Path:
    result_root = Path(cli.result_root).resolve()
    config = json.loads((result_root / "run_config.json").read_text(encoding="utf-8"))
    args = _runtime_args(config, cli.device)
    device = _device(args.device)
    collection_rounds = tuple(sorted(set(args.collection_rounds)))
    catalog = ClassCatalog.discover(args.pretrain_data)
    checkpoint = torch.load(
        args.pretrained_autoencoder, map_location=device, weights_only=False
    )
    cut_config = str(checkpoint["cut_config"])

    seed_everything(cli.seed)
    encoder = ClientFrontFModel(cut_config).to(device)
    encoder.load_state_dict(checkpoint["encoder"])
    base_model = SplitLearningModel(
        encoder,
        ServerMiddleGModel(cut_config).to(device),
        ClientTailHModel(catalog.num_classes).to(device),
        cut_config,
    )
    pretrain_loader = make_loader(
        args.pretrain_data,
        "train",
        args.image_size,
        args.batch_size,
        args.num_workers,
        shuffle=True,
        augment=True,
        class_names=catalog.names,
    )
    _warmup_classifier(
        base_model,
        pretrain_loader,
        args.warmup_epochs,
        args.client_learning_rate,
        device,
    )
    initial_f = copy.deepcopy(base_model.f_model.state_dict())
    initial_g = copy.deepcopy(base_model.g_model.state_dict())
    initial_h = copy.deepcopy(base_model.h_model.state_dict())

    victim_train, _ = _partition_victim_dataset(
        args.victim_data,
        "train",
        catalog,
        args.image_size,
        args.num_victims,
        2026,
        augment=True,
    )
    victim_validation, _ = _partition_victim_dataset(
        args.victim_data,
        "val",
        catalog,
        args.image_size,
        args.num_victims,
        2026,
        augment=False,
    )
    victim_holdout, _ = _partition_victim_dataset(
        args.victim_data,
        "new_holdout",
        catalog,
        args.image_size,
        args.num_victims,
        2026,
        augment=False,
    )
    attacker_bundle, victim_bundles, _, _ = _run_world(
        "dynamic",
        cli.seed,
        result_root / f"seed_{cli.seed}" / "dynamic",
        initial_f,
        initial_g,
        initial_h,
        cut_config,
        catalog,
        victim_train,
        victim_validation,
        victim_holdout,
        collection_rounds,
        args,
        device,
    )
    validation_ids = _attack_validation_ids(
        attacker_bundle,
        catalog.num_classes,
        args.attack_validation_fraction,
        cli.seed,
    )
    train_bundle, _, _ = _temporal_bundles(
        attacker_bundle,
        validation_ids,
        collection_rounds,
        args.matched_budget,
        cli.seed,
    )["multi_natural"]
    victim_bundle = victim_bundles[cli.victim_index - 1]

    train_features = _prototype_features(train_bundle, "gradient_only")
    victim_features = _prototype_features(victim_bundle, "gradient_only")
    raw_means = torch.stack(
        [
            train_features[train_bundle.labels == label].mean(0)
            for label in range(catalog.num_classes)
        ]
    )
    prototypes = torch.nn.functional.normalize(raw_means, dim=1)
    similarities = victim_features @ prototypes.T
    predictions = similarities.argmax(1)
    accuracy = float((predictions == victim_bundle.labels).float().mean())
    if not np.isclose(accuracy, cli.expected_accuracy):
        raise RuntimeError(
            f"replayed accuracy {accuracy:.4f} does not match expected "
            f"{cli.expected_accuracy:.4f}"
        )

    train_np = train_features.numpy()
    victim_np = victim_features.numpy()
    raw_mean_np = raw_means.numpy()
    prototype_np = prototypes.numpy()
    combined = np.concatenate((train_np, victim_np, raw_mean_np, prototype_np), axis=0)
    pca = PCA(n_components=2, random_state=cli.seed)
    projected = pca.fit_transform(combined)
    n_train = len(train_np)
    n_victim = len(victim_np)
    train_xy = projected[:n_train]
    victim_xy = projected[n_train : n_train + n_victim]
    mean_xy = projected[n_train + n_victim : n_train + n_victim + catalog.num_classes]
    prototype_xy = projected[-catalog.num_classes :]

    colors = plt.get_cmap("tab10")(np.arange(catalog.num_classes))
    figure, (scatter_ax, heatmap_ax) = plt.subplots(
        1,
        2,
        figsize=(15.5, 7.6),
        gridspec_kw={"width_ratios": [1.15, 1]},
        constrained_layout=True,
    )
    train_labels = train_bundle.labels.numpy()
    victim_labels = victim_bundle.labels.numpy()
    for label, class_name in enumerate(catalog.names):
        train_mask = train_labels == label
        victim_mask = victim_labels == label
        scatter_ax.scatter(
            train_xy[train_mask, 0],
            train_xy[train_mask, 1],
            s=13,
            alpha=0.24,
            color=colors[label],
            linewidths=0,
            label=class_name,
        )
        scatter_ax.scatter(
            victim_xy[victim_mask, 0],
            victim_xy[victim_mask, 1],
            s=30,
            marker="s",
            facecolors="none",
            edgecolors=colors[label],
            linewidths=0.9,
        )
        scatter_ax.plot(
            [mean_xy[label, 0], prototype_xy[label, 0]],
            [mean_xy[label, 1], prototype_xy[label, 1]],
            color=colors[label],
            linewidth=1.1,
            alpha=0.8,
        )
        scatter_ax.scatter(
            mean_xy[label, 0],
            mean_xy[label, 1],
            s=75,
            marker="D",
            color=colors[label],
            edgecolors="black",
            linewidths=0.7,
            zorder=5,
        )
        scatter_ax.scatter(
            prototype_xy[label, 0],
            prototype_xy[label, 1],
            s=260,
            marker="*",
            color=colors[label],
            edgecolors="black",
            linewidths=0.9,
            zorder=6,
        )
    explained = 100.0 * pca.explained_variance_ratio_
    scatter_ax.set_xlabel(f"PCA 1 ({explained[0]:.1f}% variance)")
    scatter_ax.set_ylabel(f"PCA 2 ({explained[1]:.1f}% variance)")
    scatter_ax.set_title("Feature projection and class prototypes (PCA view only)")
    scatter_ax.grid(alpha=0.18)
    class_legend = scatter_ax.legend(title="True class", loc="upper right", fontsize=8)
    scatter_ax.add_artist(class_legend)
    scatter_ax.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="none", color="gray", alpha=0.5, label="Attacker record"),
            Line2D([], [], marker="s", linestyle="none", markerfacecolor="none", markeredgecolor="gray", label="Victim record"),
            Line2D([], [], marker="D", linestyle="none", color="gray", label="Class mean"),
            Line2D([], [], marker="*", linestyle="none", color="gray", markersize=13, label="Normalized prototype"),
        ],
        loc="lower left",
        fontsize=8,
    )

    order = np.lexsort((predictions.numpy(), victim_labels))
    sorted_similarities = similarities.numpy()[order]
    sorted_predictions = predictions.numpy()[order]
    sorted_labels = victim_labels[order]
    image = heatmap_ax.imshow(
        sorted_similarities,
        aspect="auto",
        cmap="viridis",
        vmin=float(similarities.min()),
        vmax=float(similarities.max()),
    )
    rows = np.arange(len(order))
    heatmap_ax.scatter(
        sorted_predictions,
        rows,
        marker="s",
        s=22,
        facecolors="none",
        edgecolors="white",
        linewidths=0.75,
        label="Predicted (row maximum)",
    )
    centers: list[float] = []
    for label in range(catalog.num_classes):
        indices = np.flatnonzero(sorted_labels == label)
        centers.append(float(indices.mean()))
        if label:
            heatmap_ax.axhline(indices.min() - 0.5, color="white", linewidth=1.0)
    heatmap_ax.set_xticks(np.arange(catalog.num_classes), catalog.names, rotation=35, ha="right")
    heatmap_ax.set_yticks(centers, [f"true: {name}" for name in catalog.names])
    heatmap_ax.set_xlabel("Class prototype (white square = predicted row maximum)")
    heatmap_ax.set_ylabel("Victim records grouped by true class")
    heatmap_ax.set_title(f"Original {train_features.shape[1]}D cosine similarities")
    heatmap_ax.legend(loc="lower right", fontsize=8)
    colorbar = figure.colorbar(image, ax=heatmap_ax, shrink=0.82)
    colorbar.set_label("Cosine similarity")

    figure.suptitle(
        "Cosine Prototype · Gradient-only · Dynamic · Multi-Natural\n"
        f"seed {cli.seed}, victim {cli.victim_index} | "
        f"attacker records {len(train_bundle)} | accuracy {accuracy:.1%}",
        fontsize=14,
        fontweight="bold",
    )
    output = (
        Path(cli.output).resolve()
        if cli.output
        else result_root
        / "figures"
        / f"cosine_prototype_gradient_dynamic_multi_natural_seed{cli.seed}.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(
        json.dumps(
            {
                "output": str(output),
                "seed": cli.seed,
                "victim_index": cli.victim_index,
                "train_records": len(train_bundle),
                "victim_records": len(victim_bundle),
                "feature_dimensions": int(train_features.shape[1]),
                "accuracy": accuracy,
                "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
            },
            indent=2,
        )
    )
    return output


if __name__ == "__main__":
    run(build_parser().parse_args())
