from __future__ import annotations

import argparse
import json
from pathlib import Path

from ...client_received_transcript_attack.data.dataset import (
    ClientReceivedTranscriptDataset,
)
from ..evaluation.evaluator import evaluate_refiner
from .common import (
    device_from_name,
    file_sha256,
    load_frozen_decoder,
    load_refiner,
    transcript_ids,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Original/Coarse/Refined outputs on an unseen evaluator dataset."
        )
    )
    parser.add_argument("--refiner-checkpoint", required=True)
    parser.add_argument(
        "--decoder-checkpoint",
        help="Defaults to the frozen decoder recorded in the refiner checkpoint.",
    )
    parser.add_argument("--attacker-manifest", required=True)
    parser.add_argument("--evaluator-manifest", required=True)
    parser.add_argument("--class-names", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-grid-images", type=int, default=20)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--allow-seen-transcripts",
        action="store_true",
        help="Allow train/evaluation transcript overlap for debugging only.",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    device = device_from_name(args.device)
    refiner, checkpoint = load_refiner(args.refiner_checkpoint, device)
    metadata = checkpoint.get("metadata", {})
    decoder_checkpoint = args.decoder_checkpoint or metadata.get("decoder_checkpoint")
    if not decoder_checkpoint:
        raise ValueError(
            "--decoder-checkpoint is required because the refiner checkpoint does not "
            "record one"
        )
    expected_hash = metadata.get("decoder_sha256")
    actual_hash = file_sha256(decoder_checkpoint)
    if expected_hash and actual_hash != expected_hash:
        raise ValueError(
            "decoder checkpoint does not match the decoder used to train the refiner"
        )
    decoder = load_frozen_decoder(decoder_checkpoint, device)
    if decoder.config.num_classes != len(args.class_names):
        raise ValueError("decoder class count does not match --class-names")

    dataset = ClientReceivedTranscriptDataset(
        args.attacker_manifest, args.evaluator_manifest
    )
    seen_ids = set(metadata.get("train_transcript_ids", [])) | set(
        metadata.get("validation_transcript_ids", [])
    )
    overlap = transcript_ids(dataset) & seen_ids
    if overlap and not args.allow_seen_transcripts:
        examples = ", ".join(sorted(overlap)[:3])
        raise ValueError(
            f"evaluation contains {len(overlap)} refiner train/validation transcripts; "
            f"examples: {examples}"
        )

    summary = evaluate_refiner(
        decoder,
        refiner,
        dataset,
        args.output,
        device,
        tuple(args.class_names),
        batch_size=args.batch_size,
        max_grid_images=args.max_grid_images,
    )
    output = Path(args.output)
    with (output / "evaluation_run_config.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                **vars(args),
                "device": str(device),
                "decoder_checkpoint": str(decoder_checkpoint),
                "decoder_sha256": actual_hash,
                "refiner_type": checkpoint.get("refiner_type", "image_only"),
                "refiner_best_epoch": checkpoint.get("best_epoch"),
                "loaded_split_learning_checkpoint": False,
                "holdout_used_for_training": bool(overlap),
                "evaluation_summary": summary,
            },
            handle,
            indent=2,
            default=str,
        )
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()


__all__ = ["build_parser", "run"]
