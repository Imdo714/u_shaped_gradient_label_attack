from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


PIPELINE_MODULE = (
    "src.shared_pretrained_encoder_drift_attack.pipeline."
    "run_label_inference_improvements"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the controlled no/shared/independent dynamic-augmentation "
            "comparison over seeds and victim environments, then aggregate results."
        )
    )
    base = "workspace/data/shared_pretrained_encoder_joint_attack/animal5"
    parser.add_argument("--pretrained-autoencoder", required=True)
    parser.add_argument("--pretrain-data", default=f"{base}/pretrain")
    parser.add_argument("--attacker-data", default=f"{base}/attacker")
    parser.add_argument("--victim-data", default=f"{base}/victim")
    parser.add_argument(
        "--output",
        default=(
            "workspace/results/shared_pretrained_encoder_drift_attack/"
            "animal5_dynamic_env"
        ),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument(
        "--victim-learning-rates", nargs="+", type=float, default=[1e-3]
    )
    parser.add_argument(
        "--victim-observation-batch-sizes", nargs="+", type=int, default=[16]
    )
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--attacker-finetune-epochs", type=int, default=20)
    parser.add_argument(
        "--attacker-snapshot-epochs",
        nargs="+",
        type=int,
        default=[0, 1, 5, 10, 20],
    )
    parser.add_argument("--victim-finetune-epochs", type=int, default=20)
    parser.add_argument(
        "--capture-epochs", nargs="+", type=int, default=[0, 1, 5, 10, 20]
    )
    parser.add_argument("--attack-epochs", type=int, default=30)
    parser.add_argument("--holdout-per-class", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--attack-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--attack-learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser


def _float_token(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sample_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _aggregate(output: Path, manifest: list[dict[str, object]]) -> None:
    run_summaries: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    epoch_summaries: dict[
        tuple[str, str, int], list[dict[str, str]]
    ] = defaultdict(list)

    for run in manifest:
        run_root = Path(str(run["run_output"]))
        environment = str(run["environment"])
        for row in _read_csv(run_root / "dynamic_comparison_summary.csv"):
            run_summaries[(environment, row["variant"])].append(row)
        for row in _read_csv(run_root / "label_inference_improvement_summary.csv"):
            epoch_summaries[
                (environment, row["variant"], int(row["epoch"]))
            ].append(row)

    environment_lookup = {
        str(row["environment"]): row for row in manifest
    }
    summary_rows: list[dict[str, object]] = []
    for (environment, variant), rows in sorted(run_summaries.items()):
        metadata = environment_lookup[environment]
        mean_accuracy = [float(row["mean_accuracy"]) for row in rows]
        min_accuracy = [float(row["min_accuracy"]) for row in rows]
        final_accuracy = [float(row["final_accuracy"]) for row in rows]
        mean_macro_f1 = [float(row["mean_macro_f1"]) for row in rows]
        summary_rows.append(
            {
                "environment": environment,
                "victim_learning_rate": metadata["victim_learning_rate"],
                "victim_observation_batch_size": metadata[
                    "victim_observation_batch_size"
                ],
                "variant": variant,
                "seed_count": len(rows),
                "mean_accuracy": statistics.mean(mean_accuracy),
                "std_accuracy_between_seeds": _sample_std(mean_accuracy),
                "mean_worst_case_accuracy": statistics.mean(min_accuracy),
                "mean_final_accuracy": statistics.mean(final_accuracy),
                "std_final_accuracy_between_seeds": _sample_std(final_accuracy),
                "mean_macro_f1": statistics.mean(mean_macro_f1),
            }
        )

    epoch_rows: list[dict[str, object]] = []
    for (environment, variant, epoch), rows in sorted(epoch_summaries.items()):
        metadata = environment_lookup[environment]
        accuracy = [float(row["accuracy"]) for row in rows]
        macro_f1 = [float(row["macro_f1"]) for row in rows]
        epoch_rows.append(
            {
                "environment": environment,
                "victim_learning_rate": metadata["victim_learning_rate"],
                "victim_observation_batch_size": metadata[
                    "victim_observation_batch_size"
                ],
                "variant": variant,
                "victim_epoch": epoch,
                "seed_count": len(rows),
                "mean_accuracy": statistics.mean(accuracy),
                "std_accuracy_between_seeds": _sample_std(accuracy),
                "mean_macro_f1": statistics.mean(macro_f1),
                "std_macro_f1_between_seeds": _sample_std(macro_f1),
            }
        )

    _write_csv(output / "environment_summary.csv", summary_rows)
    _write_csv(output / "environment_epoch_summary.csv", epoch_rows)


def run(args: argparse.Namespace) -> Path:
    if not args.seeds:
        raise ValueError("at least one seed is required")
    if any(value <= 0 for value in args.victim_learning_rates):
        raise ValueError("victim learning rates must be positive")
    if any(value < 1 for value in args.victim_observation_batch_sizes):
        raise ValueError("victim observation batch sizes must be positive")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    for victim_learning_rate in args.victim_learning_rates:
        for observation_batch_size in args.victim_observation_batch_sizes:
            environment = (
                f"lr_{_float_token(victim_learning_rate)}__b{observation_batch_size}"
            )
            for seed in args.seeds:
                run_output = output / environment / f"seed_{seed}"
                complete = run_output / "dynamic_comparison_summary.csv"
                if complete.exists():
                    print(f"Skipping completed run: {run_output.resolve()}", flush=True)
                else:
                    if run_output.exists() and any(run_output.iterdir()):
                        raise RuntimeError(
                            f"incomplete output already exists: {run_output.resolve()}"
                        )
                    command = [
                        sys.executable,
                        "-m",
                        PIPELINE_MODULE,
                        "--pretrained-autoencoder",
                        args.pretrained_autoencoder,
                        "--pretrain-data",
                        args.pretrain_data,
                        "--attacker-data",
                        args.attacker_data,
                        "--victim-data",
                        args.victim_data,
                        "--output",
                        str(run_output),
                        "--warmup-epochs",
                        str(args.warmup_epochs),
                        "--attacker-finetune-epochs",
                        str(args.attacker_finetune_epochs),
                        "--attacker-snapshot-epochs",
                        *map(str, args.attacker_snapshot_epochs),
                        "--victim-finetune-epochs",
                        str(args.victim_finetune_epochs),
                        "--capture-epochs",
                        *map(str, args.capture_epochs),
                        "--attack-epochs",
                        str(args.attack_epochs),
                        "--holdout-per-class",
                        str(args.holdout_per_class),
                        "--image-size",
                        str(args.image_size),
                        "--batch-size",
                        str(args.batch_size),
                        "--attack-batch-size",
                        str(args.attack_batch_size),
                        "--learning-rate",
                        str(args.learning_rate),
                        "--victim-learning-rate",
                        str(victim_learning_rate),
                        "--victim-observation-batch-size",
                        str(observation_batch_size),
                        "--attack-learning-rate",
                        str(args.attack_learning_rate),
                        "--seed",
                        str(seed),
                        "--num-workers",
                        str(args.num_workers),
                        "--device",
                        args.device,
                        "--controlled-dynamic-ablation",
                    ]
                    print(
                        f"Running {environment}, seed={seed}: {run_output.resolve()}",
                        flush=True,
                    )
                    subprocess.run(command, check=True)
                manifest.append(
                    {
                        "environment": environment,
                        "victim_learning_rate": victim_learning_rate,
                        "victim_observation_batch_size": observation_batch_size,
                        "seed": seed,
                        "run_output": str(run_output),
                    }
                )

    _write_csv(output / "environment_run_manifest.csv", manifest)
    _aggregate(output, manifest)
    (output / "sweep_config.json").write_text(
        json.dumps(vars(args), indent=2, default=str), encoding="utf-8"
    )
    print(f"Environment sweep results: {output.resolve()}", flush=True)
    return output


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()


__all__ = ["build_parser", "run"]
