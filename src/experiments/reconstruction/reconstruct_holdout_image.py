from __future__ import annotations

import argparse
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


def build_parser() -> argparse.ArgumentParser:
    parser = build_reconstruction_parser()
    parser.description = (
        "Train without one selected image and reconstruct that image only at evaluation."
    )
    parser.add_argument("--holdout-label", required=True)
    parser.add_argument(
        "--holdout-split",
        choices=("train", "val", "test", "new_holdout"),
        default="test",
    )
    parser.add_argument("--holdout-index", type=int, default=0)
    parser.set_defaults(decoder_preset="strong")
    return parser


def run(args: argparse.Namespace) -> dict[str, float | int]:
    if args.output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = str(Path("workspace/results/decoder") / f"holdout_{stamp}")
    catalog = ClassCatalog.discover(args.data)
    records = select_holdout_records(
        args.data,
        args.holdout_split,
        catalog.names,
        holdouts_per_class=1,
        labels=(args.holdout_label,),
        start_index=args.holdout_index,
    )
    write_holdout_records(records, Path(args.output) / "holdout_records.csv")
    sample_ids = {record.sample_id for record in records}
    return run_reconstruction(
        args,
        evaluation_split=args.holdout_split,
        evaluation_sample_ids=sample_ids,
        excluded_sample_ids=sample_ids,
    )


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()


__all__ = ["build_parser", "run"]
