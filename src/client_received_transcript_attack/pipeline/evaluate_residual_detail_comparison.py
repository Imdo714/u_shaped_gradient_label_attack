from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from ..data.dataset import ClientReceivedTranscriptDataset
from ..evaluation.residual_detail_comparison import (
    evaluate_residual_detail_comparison,
)
from ..models.factory import load_decoder_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Original / bilinear baseline / residual-detail decoder on "
            "the same unseen transcript holdout."
        )
    )
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--residual-detail-checkpoint", required=True)
    parser.add_argument("--attacker-manifest", required=True)
    parser.add_argument("--evaluator-manifest", required=True)
    parser.add_argument("--class-names", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-grid-images", type=int, default=20)
    parser.add_argument("--device", default="auto")
    return parser


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict[str, object]:
    device = _device(args.device)
    baseline, baseline_state = load_decoder_checkpoint(
        args.baseline_checkpoint, device
    )
    residual_detail, detail_state = load_decoder_checkpoint(
        args.residual_detail_checkpoint, device
    )
    baseline_type = str(
        baseline_state.get("decoder_type", "baseline_bilinear")
    )
    detail_type = str(detail_state.get("decoder_type", "baseline_bilinear"))
    if baseline_type != "baseline_bilinear":
        raise ValueError("--baseline-checkpoint must be baseline_bilinear")
    if detail_type != "residual_detail":
        raise ValueError("--residual-detail-checkpoint must be residual_detail")
    if baseline.config.num_classes != len(args.class_names):
        raise ValueError("baseline class count does not match --class-names")
    if residual_detail.config.num_classes != len(args.class_names):
        raise ValueError("residual-detail class count does not match --class-names")
    if baseline.config.image_size != residual_detail.config.image_size:
        raise ValueError("decoder image sizes differ")

    dataset = ClientReceivedTranscriptDataset(
        args.attacker_manifest, args.evaluator_manifest
    )
    summary = evaluate_residual_detail_comparison(
        baseline,
        residual_detail,
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
                "baseline_decoder_type": baseline_type,
                "candidate_decoder_type": detail_type,
                "baseline_sha256": _sha256(args.baseline_checkpoint),
                "residual_detail_sha256": _sha256(
                    args.residual_detail_checkpoint
                ),
                "loaded_split_learning_checkpoint": False,
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
