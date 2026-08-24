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
    ObservationDataset,
    collect_observations,
    collect_surrogate_observations,
)
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


def _loader(args, split: str, class_names: tuple[str, ...]):
    return make_loader(
        args.data,
        split,
        args.image_size,
        batch_size=1,
        num_workers=args.num_workers,
        shuffle=False,
        class_names=class_names,
    )


def run(args: argparse.Namespace) -> dict[str, float | int]:
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
    observations = output / "observations"
    output.mkdir(parents=True, exist_ok=True)

    predictor = None
    if args.victim_label_mode != "oracle":
        predictor = GradientLabelPredictor.from_files(
            args.centroids,
            args.cluster_mapping,
            catalog.num_classes,
            args.temperature,
        )

    print("[1/4] Collecting auxiliary train/validation observations...")
    train_manifest = collect_observations(
        victim_model, _loader(args, "train", catalog.names), observations / "train",
        device, catalog.num_classes, "oracle", max_samples=args.max_train_samples,
    )
    validation_manifest = collect_observations(
        victim_model, _loader(args, "val", catalog.names), observations / "val",
        device, catalog.num_classes, "oracle", max_samples=args.max_val_samples,
    )
    print("[2/4] Collecting victim observations with gradient-inferred labels...")
    test_manifest = collect_observations(
        victim_model, _loader(args, "test", catalog.names), observations / "test",
        device, catalog.num_classes, args.victim_label_mode, predictor=predictor,
        max_samples=args.max_test_samples,
    )
    train_dataset = ObservationDataset(train_manifest)
    validation_dataset = ObservationDataset(validation_manifest)
    test_dataset = ObservationDataset(test_manifest)

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

    sample = train_dataset[0]
    decoder_config = DecoderConfig(
        z_channels=int(sample["smashed_z"].shape[0]),
        u_channels=int(sample["server_output_u"].shape[0]),
        gradient_channels=int(sample["grad_h_to_g"].shape[0]),
        num_classes=catalog.num_classes,
        image_size=args.image_size,
        use_z="z" in selected,
        use_u="u" in selected,
        use_gradient="gradient" in selected,
    )
    decoder = LabelConditionedDecoder(decoder_config).to(device)
    training_config = DecoderTrainingConfig(
        epochs=args.decoder_epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate,
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
