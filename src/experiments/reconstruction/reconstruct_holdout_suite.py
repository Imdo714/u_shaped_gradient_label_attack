from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from src.decoder.data.holdout_selection import (
    select_holdout_records,
    write_holdout_records,
)
from src.decoder.pipeline.run_reconstruction_experiment import (
    build_parser as build_reconstruction_parser,
    run as run_reconstruction,
)
from src.shared.data.class_catalog import ClassCatalog


CONDITIONS = {
    "strong_all_soft": ("strong", "z,u,gradient", "stored"),
    "strong_all_oracle": ("strong", "z,u,gradient", "oracle"),
    "strong_no_label": ("strong", "z,u,gradient", "zero"),
    "strong_z_only": ("strong", "z", "stored"),
    "strong_z_u": ("strong", "z,u", "stored"),
    "strong_z_gradient": ("strong", "z,gradient", "stored"),
    "strong_gradient_only": ("strong", "gradient", "stored"),
    "strong_u_grad_z": ("strong", "u,grad_z", "stored"),
    "baseline_all_soft": ("baseline", "z,u,gradient", "stored"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = build_reconstruction_parser()
    parser.description = "Run the multi-holdout decoder ablation suite."
    parser.add_argument("--holdouts-per-class", type=int, default=20)
    parser.add_argument(
        "--holdout-count",
        type=int,
        default=None,
        help="Select this many holdouts in total, balanced across all classes.",
    )
    parser.add_argument(
        "--holdout-start-index",
        type=int,
        default=0,
        help="Start at this zero-based index within every selected class.",
    )
    parser.add_argument(
        "--holdout-labels",
        nargs="+",
        default=None,
        help="Restrict holdout selection to these class names.",
    )
    parser.add_argument(
        "--holdout-split",
        choices=("train", "val", "test", "new_holdout"),
        default="test",
    )
    parser.add_argument(
        "--conditions",
        default=None,
        help=(
            "Comma-separated condition names to run. By default, run the full suite. "
            f"Available: {','.join(CONDITIONS)}"
        ),
    )
    parser.set_defaults(max_grid_images=60)
    return parser


def _write_condition_summary(rows: list[dict[str, object]], path: Path) -> None:
    fieldnames = ["condition"]
    fieldnames.extend(
        sorted({key for row in rows for key in row if key != "condition"})
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> list[dict[str, object]]:
    if args.output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = str(Path("workspace/results/decoder") / f"holdout_suite_{stamp}")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    catalog = ClassCatalog.discover(args.data)
    selected_labels = (
        tuple(args.holdout_labels) if args.holdout_labels else catalog.names
    )
    unknown_labels = sorted(set(selected_labels) - set(catalog.names))
    if unknown_labels:
        raise ValueError(
            f"unknown holdout labels: {unknown_labels}; available: {list(catalog.names)}"
        )
    if args.holdout_count is None:
        records = select_holdout_records(
            args.data,
            args.holdout_split,
            catalog.names,
            holdouts_per_class=args.holdouts_per_class,
            labels=selected_labels,
            start_index=args.holdout_start_index,
        )
    else:
        if args.holdout_count < 1:
            raise ValueError("--holdout-count must be positive")
        base_count, extra = divmod(args.holdout_count, len(selected_labels))
        records = []
        for class_index, label in enumerate(selected_labels):
            class_count = base_count + int(class_index < extra)
            if class_count:
                records.extend(
                    select_holdout_records(
                        args.data,
                        args.holdout_split,
                        catalog.names,
                        holdouts_per_class=class_count,
                        labels=(label,),
                        start_index=args.holdout_start_index,
                    )
                )
    write_holdout_records(records, output / "holdout_records.csv")
    sample_ids = {record.sample_id for record in records}
    shared_observations = output / "observations"

    requested = (
        [name.strip() for name in args.conditions.split(",") if name.strip()]
        if args.conditions
        else list(CONDITIONS)
    )
    unknown = sorted(set(requested) - set(CONDITIONS))
    if unknown:
        raise ValueError(f"unknown conditions: {unknown}; available: {list(CONDITIONS)}")
    if not requested:
        raise ValueError("--conditions must select at least one condition")
    selected_conditions = [(name, CONDITIONS[name]) for name in requested]

    summaries: list[dict[str, object]] = []
    for index, (name, (preset, signals, condition_mode)) in enumerate(
        selected_conditions
    ):
        print(f"\n=== Condition {index + 1}/{len(selected_conditions)}: {name} ===")
        condition_args = argparse.Namespace(**vars(args))
        condition_args.output = str(output / "conditions" / name)
        condition_args.decoder_preset = preset
        condition_args.signals = signals
        condition_args.victim_label_mode = "inferred-soft"
        condition_args.train_surrogates = False
        condition_args.decoder_observation_source = "exact"
        condition_args.max_test_samples = None
        summary = run_reconstruction(
            condition_args,
            evaluation_split=args.holdout_split,
            evaluation_sample_ids=sample_ids,
            excluded_sample_ids=sample_ids,
            condition_mode=condition_mode,
            shared_observations_dir=shared_observations,
            reuse_observations=index > 0,
        )
        summaries.append({"condition": name, **summary})
        _write_condition_summary(summaries, output / "condition_summary.csv")

    with (output / "suite_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                **vars(args),
                "class_names": list(catalog.names),
                "conditions": requested,
                "holdout_samples": len(records),
            },
            handle,
            indent=2,
            default=str,
        )
    return summaries


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()


__all__ = ["CONDITIONS", "build_parser", "run"]
