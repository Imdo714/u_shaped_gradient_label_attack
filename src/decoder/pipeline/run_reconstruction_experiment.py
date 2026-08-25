from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch

from ...shared.data.class_catalog import checkpoint_class_catalog
from ...shared.data.image_dataset import make_loader
from ...shared.reproducibility.random_seed import seed_everything
from ...split_learning.architecture.split_learning_model import load_split_learning_model
from ..data.gradient_label_predictor import GradientLabelPredictor
from ..data.observation_dataset import (
    ConditionedObservationDataset,
    ObservationDataset,
    collect_observations,
    collect_surrogate_observations,
)
from ..data.holdout_selection import assert_holdouts_excluded
from ..evaluation.reconstruction_evaluator import evaluate_reconstructions
from ..models.label_conditioned_decoder import DecoderConfig, LabelConditionedDecoder
from ..surrogate_models.surrogate_f_model import SurrogateFModel
from ..surrogate_models.surrogate_h_model import SurrogateHModel
from ..training.decoder_trainer import DecoderTrainingConfig, train_decoder
from ..training.surrogate_trainer import train_surrogate_f, train_surrogate_h


DEFAULT_RUN = Path("workspace/results/runs/cut_middle")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a label-conditioned split-learning image decoder."
    )
    parser.add_argument("--data", default="workspace/data/dataset")
    parser.add_argument("--checkpoint", default=str(DEFAULT_RUN / "checkpoints/model.pt"))
    parser.add_argument(
        "--centroids",
        default=str(DEFAULT_RUN / "results/gradient_centroids_epoch_010.npy"),
    )
    parser.add_argument(
        "--cluster-mapping", default=str(DEFAULT_RUN / "results/cluster_mapping.json")
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--decoder-epochs", type=int, default=20)
    parser.add_argument("--surrogate-epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument(
        "--victim-label-mode",
        choices=("inferred-hard", "inferred-soft", "oracle"),
        default="inferred-soft",
    )
    parser.add_argument(
        "--signals", default="z,u,gradient", help="Comma-separated: z,u,gradient"
    )
    parser.add_argument("--train-surrogates", action="store_true")
    parser.add_argument(
        "--decoder-observation-source",
        choices=("exact", "surrogate"),
        default="exact",
        help="Use victim observations or f-hat/g/h-hat generated auxiliary observations.",
    )
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--decoder-preset", choices=("baseline", "strong"), default="baseline"
    )
    parser.add_argument("--signal-spatial-size", type=int, default=None)
    parser.add_argument("--signal-channels", type=int, default=None)
    parser.add_argument("--label-channels", type=int, default=None)
    parser.add_argument("--decoder-base-channels", type=int, default=None)
    parser.add_argument("--decoder-min-channels", type=int, default=None)
    parser.add_argument("--decoder-refinement-blocks", type=int, default=None)
    parser.add_argument("--l1-weight", type=float, default=None)
    parser.add_argument("--ssim-weight", type=float, default=None)
    parser.add_argument("--edge-weight", type=float, default=None)
    parser.add_argument("--perceptual-weight", type=float, default=None)
    parser.add_argument("--max-grid-images", type=int, default=12)
    parser.add_argument(
        "--no-save-separate-images",
        action="store_false",
        dest="save_separate_images",
        help="Do not save individual originals/ and reconstructions/ PNG files.",
    )
    parser.set_defaults(save_separate_images=True)
    return parser


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _signal_selection(value: str) -> set[str]:
    selected = {part.strip() for part in value.split(",") if part.strip()}
    allowed = {"z", "u", "gradient"}
    if not selected or not selected.issubset(allowed):
        raise ValueError(f"--signals must contain one or more of {sorted(allowed)}")
    return selected


def _loader(
    args,
    split: str,
    class_names: tuple[str, ...],
    include_sample_ids: set[str] | None = None,
    exclude_sample_ids: set[str] | None = None,
):
    return make_loader(
        args.data,
        split,
        args.image_size,
        batch_size=1,
        num_workers=args.num_workers,
        shuffle=False,
        class_names=class_names,
        include_sample_ids=include_sample_ids,
        exclude_sample_ids=exclude_sample_ids,
    )


DECODER_PRESETS = {
    "baseline": {
        "signal_spatial_size": 8,
        "signal_channels": 32,
        "label_channels": 16,
        "decoder_base_channels": 128,
        "decoder_min_channels": 16,
        "decoder_refinement_blocks": 0,
        "l1_weight": 1.0,
        "ssim_weight": 0.5,
        "edge_weight": 0.0,
        "perceptual_weight": 0.0,
    },
    "strong": {
        "signal_spatial_size": 16,
        "signal_channels": 64,
        "label_channels": 32,
        "decoder_base_channels": 256,
        "decoder_min_channels": 32,
        "decoder_refinement_blocks": 1,
        "l1_weight": 1.0,
        "ssim_weight": 0.75,
        "edge_weight": 0.15,
        "perceptual_weight": 0.25,
    },
}


def resolved_decoder_options(args: argparse.Namespace) -> dict[str, int | float]:
    preset_name = getattr(args, "decoder_preset", "baseline")
    options = dict(DECODER_PRESETS[preset_name])
    for name in options:
        override = getattr(args, name, None)
        if override is not None:
            options[name] = override
    return options


def run(
    args: argparse.Namespace,
    *,
    evaluation_split: str = "test",
    evaluation_sample_ids: set[str] | None = None,
    excluded_sample_ids: set[str] | None = None,
    condition_mode: str = "stored",
    shared_observations_dir: str | Path | None = None,
    reuse_observations: bool = False,
) -> dict[str, float | int]:
    seed_everything(args.seed)
    device = _device(args.device)
    victim_model, metadata = load_split_learning_model(args.checkpoint, device)
    catalog = checkpoint_class_catalog(metadata, victim_model.h_model.num_classes)
    selected = _signal_selection(args.signals)
    if args.output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = Path("workspace/results/decoder") / stamp
    else:
        output = Path(args.output)
    observations = (
        Path(shared_observations_dir)
        if shared_observations_dir is not None
        else output / "observations"
    )
    output.mkdir(parents=True, exist_ok=True)

    predictor = None
    if args.victim_label_mode != "oracle":
        predictor = GradientLabelPredictor.from_files(
            args.centroids,
            args.cluster_mapping,
            catalog.num_classes,
            args.temperature,
        )

    excluded_sample_ids = excluded_sample_ids or set()
    if reuse_observations:
        train_manifest = observations / "train" / "manifest.csv"
        validation_manifest = observations / "val" / "manifest.csv"
        test_manifest = observations / "test" / "manifest.csv"
        missing = [
            path
            for path in (train_manifest, validation_manifest, test_manifest)
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(f"shared observation manifests not found: {missing}")
        print("[1/4] Reusing shared train/validation/victim observations...")
    else:
        print("[1/4] Collecting auxiliary train/validation observations...")
        train_manifest = collect_observations(
            victim_model,
            _loader(args, "train", catalog.names, exclude_sample_ids=excluded_sample_ids),
            observations / "train",
            device, catalog.num_classes, "oracle", max_samples=args.max_train_samples,
        )
        validation_manifest = collect_observations(
            victim_model,
            _loader(args, "val", catalog.names, exclude_sample_ids=excluded_sample_ids),
            observations / "val",
            device, catalog.num_classes, "oracle", max_samples=args.max_val_samples,
        )
        print("[2/4] Collecting victim observations with gradient-inferred labels...")
        test_manifest = collect_observations(
            victim_model,
            _loader(
                args,
                evaluation_split,
                catalog.names,
                include_sample_ids=evaluation_sample_ids,
            ),
            observations / "test",
            device, catalog.num_classes, args.victim_label_mode, predictor=predictor,
            max_samples=args.max_test_samples,
        )
    train_dataset = ObservationDataset(train_manifest)
    validation_dataset = ObservationDataset(validation_manifest)
    test_dataset = ObservationDataset(test_manifest)
    if excluded_sample_ids:
        assert_holdouts_excluded(
            (train_manifest, validation_manifest), excluded_sample_ids
        )

    train_clones = args.train_surrogates or args.decoder_observation_source == "surrogate"
    if train_clones:
        print("[optional] Training attacker-owned f-hat and h-hat clones...")
        surrogate_dir = output / "surrogates"
        surrogate_f = SurrogateFModel(victim_model.cut_config)
        surrogate_h = SurrogateHModel(catalog.num_classes)
        train_surrogate_f(
            surrogate_f, train_dataset,
            surrogate_dir / "surrogate_f.pt", device, epochs=args.surrogate_epochs,
            batch_size=args.batch_size, learning_rate=args.learning_rate,
        )
        train_surrogate_h(
            surrogate_h, train_dataset,
            surrogate_dir / "surrogate_h.pt", device, epochs=args.surrogate_epochs,
            batch_size=args.batch_size, learning_rate=args.learning_rate,
        )
        if args.decoder_observation_source == "surrogate":
            print("[optional] Rebuilding auxiliary observations through f-hat/g/h-hat...")
            train_dataset = ObservationDataset(
                collect_surrogate_observations(
                    train_dataset, surrogate_f, victim_model.g_model, surrogate_h,
                    observations / "surrogate_train", device, catalog.num_classes,
                )
            )
            validation_dataset = ObservationDataset(
                collect_surrogate_observations(
                    validation_dataset, surrogate_f, victim_model.g_model, surrogate_h,
                    observations / "surrogate_val", device, catalog.num_classes,
                )
            )

    if condition_mode != "stored":
        train_dataset = ConditionedObservationDataset(
            train_dataset, condition_mode, catalog.num_classes
        )
        validation_dataset = ConditionedObservationDataset(
            validation_dataset, condition_mode, catalog.num_classes
        )
        test_dataset = ConditionedObservationDataset(
            test_dataset, condition_mode, catalog.num_classes
        )

    sample = train_dataset[0]
    decoder_options = resolved_decoder_options(args)
    decoder_config = DecoderConfig(
        z_channels=int(sample["smashed_z"].shape[0]),
        u_channels=int(sample["server_output_u"].shape[0]),
        gradient_channels=int(sample["grad_h_to_g"].shape[0]),
        num_classes=catalog.num_classes,
        image_size=args.image_size,
        use_z="z" in selected,
        use_u="u" in selected,
        use_gradient="gradient" in selected,
        signal_spatial_size=int(decoder_options["signal_spatial_size"]),
        signal_channels=int(decoder_options["signal_channels"]),
        label_channels=int(decoder_options["label_channels"]),
        decoder_base_channels=int(decoder_options["decoder_base_channels"]),
        decoder_min_channels=int(decoder_options["decoder_min_channels"]),
        decoder_refinement_blocks=int(decoder_options["decoder_refinement_blocks"]),
    )
    decoder = LabelConditionedDecoder(decoder_config).to(device)
    training_config = DecoderTrainingConfig(
        epochs=args.decoder_epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        l1_weight=float(decoder_options["l1_weight"]),
        ssim_weight=float(decoder_options["ssim_weight"]),
        edge_weight=float(decoder_options["edge_weight"]),
        perceptual_weight=float(decoder_options["perceptual_weight"]),
    )
    print("[3/4] Training the label-conditioned decoder...")
    decoder_checkpoint, _ = train_decoder(
        decoder, train_dataset, validation_dataset, output / "checkpoints",
        training_config, device,
    )
    print("[4/4] Evaluating victim reconstructions...")
    summary = evaluate_reconstructions(
        decoder, test_dataset, victim_model, output / "evaluation", device,
        batch_size=args.batch_size, class_names=catalog.names,
        max_grid_images=getattr(args, "max_grid_images", 12),
        save_separate_images=getattr(args, "save_separate_images", True),
    )
    run_config = vars(args).copy()
    run_config.update({
        "resolved_output": str(output),
        "device": str(device),
        "class_names": list(catalog.names),
        "decoder_config": decoder_config.to_dict(),
        "training_config": asdict(training_config),
        "decoder_checkpoint": str(decoder_checkpoint),
    })
    with (output / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Results: {output.resolve()}")
    return summary


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()


__all__ = [
    "DECODER_PRESETS",
    "build_parser",
    "resolved_decoder_options",
    "run",
]
